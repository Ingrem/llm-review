"""
GitLab MR Diff Retrieval Module
===============================

Purpose
-------
Provide a high-level workflow to retrieve *complete* diffs for a GitLab Merge
Request (MR). The module:
- Parses an MR URL to extract the project path and MR IID.
- Queries GitLab API for MR metadata and the list of changed files.
- Uses server-provided diffs when available.
- Self-generates unified diffs for files whose diffs are truncated or omitted by the API.
- Gracefully marks binary/unavailable files.

Algorithm (step-by-step)
------------------------
1) Inputs:
   - MR URL (e.g., https://gitlab.com/group/project/-/merge_requests/42)
   - Personal access token (used via `PRIVATE-TOKEN` header)
   - Optional explicit GitLab API base URL (otherwise derived from MR URL)

2) Parse MR URL:
   a) `urlparse(mr_url)` → split `path` on '/'.
   b) Find the "merge_requests" segment; the next segment is the MR IID (int).
   c) Project path is everything before "merge_requests", handling the `/-/` sentinel:
      - If the segment before "merge_requests" equals "-", drop it.
      - Join the remainder with '/' to form "group/subgroup/.../project".
   d) Derive API base as `{scheme}://{netloc}/api/v4` unless already set.

3) Fetch MR metadata:
   a) GET `/projects/{url-quoted project_path}/merge_requests/{iid}`
   b) Read `diff_refs.base_sha` and `diff_refs.head_sha` (commit SHAs required to fetch raw files).

4) Fetch MR changes:
   a) GET `/projects/{project_path}/merge_requests/{iid}/changes`
   b) For each change item, note:
      - `old_path`, `new_path`
      - `new_file`, `deleted_file`
      - `diff` (maybe truncated/omitted)
      - `overflow` or `too_large` flags (indicate incomplete API-provided diff)

5) For each changed file:
   a) If `diff` exists and `overflow`/`too_large` is false:
      - Use the API-provided diff as-is.
      - Mark: `generated=False`, `binary=False`.
   b) Otherwise, generate the diff locally:
      i) Determine which side(s) to fetch:
         - If `new_file=True`: no base version → `base_text=None`.
         - If `deleted_file=True`: no head version → `head_text=None`.
         - Else fetch both.
      ii) Fetch raw file content with:
          GET `/projects/{project}/repository/files/{file_path}/raw?ref={sha}`
          - 404 → treat as `None` (unavailable).
          - If response contains NUL (`\x00`) → likely binary → return `None`.
          - Decode as UTF-8; on failure, try Latin-1; on failure → `None`.
      iii) If both `base_text` and `head_text` are `None`:
           - Mark file as binary/unavailable: `binary=True`, `diff=""`.
      iv) Else compute a unified diff with `difflib.unified_diff()`:
          - `fromfile = old_path or "/dev/null"`
          - `tofile   = new_path or "/dev/null"`
          - Join lines with `\n`; `lineterm=""`.
          - Mark: `generated=True`, `binary=False`.
      v) Append the result record:
         `{ "old_path", "new_path", "diff", "generated", "binary" }`.

6) Return the list of per-file results.

Notes & Considerations
----------------------
- Decoding strategy prioritizes UTF-8, then Latin-1 to salvage extended-ASCII;
  if both fail, we treat the file as unavailable text (possibly binary).
- Binary heuristic (NUL byte) is simple and fast; it may misclassify some
  edge-case encodings but works well in practice for source/text files.
- Network calls use timeouts; `raise_for_status()` is used to surface HTTP errors.
- Project and file paths are URL-encoded via `urllib.parse.quote`.
- This module does not log tokens or response bodies to avoid leaking secrets.
- For very large MRs, consider pagination/rate limits and backoff handling upstream.
"""
import difflib
import requests
from urllib.parse import urlparse, quote


class GitlabWorkflow:
    def __init__(self, gitlab_token: str, gitlab_api: str = ""):
        """
        GitLab API wrapper for working with merge requests.
        :param gitlab_token: personal access token
        :param gitlab_api: optional base URL of GitLab API (auto-detected from MR URL if not set)
        """
        self.gitlab_token = gitlab_token
        self.gitlab_api = gitlab_api

    def _headers(self) -> dict:
        """Standard request headers with auth."""
        return {"PRIVATE-TOKEN": self.gitlab_token}

    def parse_mr_url(self, mr_url: str) -> tuple[str, int]:
        """
        Extract project path and MR IID from GitLab MR URL.
        Example: https://gitlab.com/group/project/-/merge_requests/42
        Returns: ("group/project", 42)
        """
        parsed = urlparse(mr_url)
        path_parts = parsed.path.strip("/").split("/")

        try:
            mr_index = path_parts.index("merge_requests")
        except ValueError:
            raise ValueError("Invalid Merge Request URL")

        # project_path может заканчиваться на /-/
        project_path = "/".join(path_parts[:mr_index - 1]) if path_parts[mr_index - 1] == "-" \
            else "/".join(path_parts[:mr_index])

        mr_iid = int(path_parts[mr_index + 1])
        self.gitlab_api = f"{parsed.scheme}://{parsed.netloc}/api/v4"
        return project_path, mr_iid

    def _fetch_raw_text(self, project_path: str, ref: str, file_path: str) -> str | None:
        """
        Fetch raw file content at a given ref.
        Returns text or None for binary/unavailable files.
        """
        url = (f"{self.gitlab_api}/projects/{quote(project_path, safe='')}"
               f"/repository/files/{quote(file_path, safe='')}/raw")
        resp = requests.get(url, headers=self._headers(), params={"ref": ref}, timeout=60)

        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        data = resp.content
        if b"\x00" in data:
            return None  # binary file

        for encoding in ("utf-8", "latin-1"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return None

    def _get_mr_meta(self, project_path: str, mr_iid: int) -> dict:
        """Fetch MR metadata including diff_refs."""
        url = f"{self.gitlab_api}/projects/{quote(project_path, safe='')}/merge_requests/{mr_iid}"
        resp = requests.get(url, headers=self._headers(), timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _get_mr_changes(self, project_path: str, mr_iid: int) -> list[dict]:
        """Fetch MR changes (file list with diffs, possibly truncated)."""
        url = f"{self.gitlab_api}/projects/{quote(project_path, safe='')}/merge_requests/{mr_iid}/changes"
        resp = requests.get(url, headers=self._headers(), timeout=120)
        resp.raise_for_status()
        return resp.json()["changes"]

    @staticmethod
    def _generate_diff(old_path: str, new_path: str, base_text: str | None, head_text: str | None) -> str:
        """Generate unified diff from base and head file versions."""
        base_lines = [] if base_text is None else base_text.splitlines()
        head_lines = [] if head_text is None else head_text.splitlines()
        return "\n".join(difflib.unified_diff(
            base_lines, head_lines,
            fromfile=old_path or "/dev/null",
            tofile=new_path or "/dev/null",
            lineterm=""
        ))

    def get_mr_changes(self, project_path: str, mr_iid: int) -> list[dict]:
        """
        Returns full diffs for MR, generating them locally for large files.
        Each item: {"old_path","new_path","diff","generated":bool,"binary":bool}
        """
        mr = self._get_mr_meta(project_path, mr_iid)
        base_sha, head_sha = mr["diff_refs"]["base_sha"], mr["diff_refs"]["head_sha"]
        changes = self._get_mr_changes(project_path, mr_iid)

        results = []
        for ch in changes:
            old_path, new_path = ch.get("old_path"), ch.get("new_path")
            is_new, is_del = ch.get("new_file", False), ch.get("deleted_file", False)

            api_diff = ch.get("diff")
            overflow = ch.get("overflow") or ch.get("too_large")

            if api_diff and not overflow:
                results.append({
                    "old_path": old_path,
                    "new_path": new_path,
                    "diff": api_diff,
                    "generated": False,
                    "binary": False,
                })
                continue

            base_text = None if is_new else self._fetch_raw_text(project_path, base_sha, old_path)
            head_text = None if is_del else self._fetch_raw_text(project_path, head_sha, new_path)

            if base_text is None and head_text is None:
                results.append({
                    "old_path": old_path,
                    "new_path": new_path,
                    "diff": "",
                    "generated": False,
                    "binary": True,
                })
                continue

            results.append({
                "old_path": old_path,
                "new_path": new_path,
                "diff": self._generate_diff(old_path, new_path, base_text, head_text),
                "generated": True,
                "binary": False,
            })

        return results
