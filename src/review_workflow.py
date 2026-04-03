import time
from pathlib import Path
from jinja2 import Environment, FileSystemLoader



class ReviewWorkflow:
    def __init__(self, llm, gitlab, root_dir=".", team_styles="", max_chars_per_batch=30000):
        """
        :param llm: LLM client with method generate_response(prompt, ...)
        :param gitlab: GitLab client with methods parse_mr_url and get_mr_changes
        :param root_dir: directory for saving review results
        :param max_chars_per_batch: maximum number of chars in diff chunk
        :param team_styles: team code style description
        """
        self.llm = llm
        self.gitlab = gitlab
        self.root_dir = root_dir
        self.max_chars_per_batch = max_chars_per_batch
        self.styles = team_styles

        env = Environment(loader=FileSystemLoader("templates"))
        self.template = env.get_template("review_prompt.j2")

    def _make_prompt(self, diff):
        """Prepare review prompt for LLM."""
        return self.template.render(
            style_guide=self.styles,
            diff_content=diff,
        )

    def _review_all_changes(self, changes):
        """
        Groups changes into batches, pre-calculates their count, and reviews each.
        """
        batches = []
        current_batch_diffs = ""

        for part in changes:
            file_path = part.get('new_path', 'unknown')
            diff = part['diff']
            file_block = f"--- FILE: {file_path} ---\n{diff}\n\n"

            if len(current_batch_diffs) + len(file_block) > self.max_chars_per_batch and current_batch_diffs:
                batches.append(current_batch_diffs)
                current_batch_diffs = file_block
            else:
                current_batch_diffs += file_block

        if current_batch_diffs:
            batches.append(current_batch_diffs)

        total_batches = len(batches)
        print(f"[INFO] Всего сформировано пакетов для ревью: {total_batches}")

        all_reviews = []
        start_total_gen = time.perf_counter()

        for i, batch_content in enumerate(batches, 1):
            print(f"[INFO] Обработка пакета {i}/{total_batches}...")

            review = self._process_batch(batch_content, i, total_batches)
            all_reviews.append(review)

        elapsed_gen = time.perf_counter() - start_total_gen
        print(f"[INFO] Суммарная генерация {total_batches}/{total_batches}: {elapsed_gen:.2f} сек")

        return "\n\n---\n\n".join(all_reviews)

    def _process_batch(self, diff_content, index, total_batches):
        """Helper to render template and get response from LLM."""
        prompt = self._make_prompt(diff_content)
        review = self.llm.generate_response(prompt)

        part_str = ""
        if total_batches != 1:
            part_str = f"part {index}"
        return f"## Review {part_str}\n\n{review}"

    def review_mr(self, mr_url):
        """Run optimized review workflow for a merge request."""
        project_path, mr_iid = self.gitlab.parse_mr_url(mr_url.strip())
        changes = self.gitlab.get_mr_changes(project_path, mr_iid)

        result_text = self._review_all_changes(changes)

        print("\n\n==================== КОД-РЕВЬЮ ====================\n\n")
        print(result_text)

        output_dir = Path(self.root_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{mr_iid}.md", "w", encoding="utf-8") as f:
            f.write(result_text)
