import time
from pathlib import Path



class ReviewWorkflow:
    def __init__(self, llm, gitlab, root_dir=".", max_lines=200, team_styles=""):
        """
        :param llm: LLM client with method generate_response(prompt, ...)
        :param gitlab: GitLab client with methods parse_mr_url and get_mr_changes
        :param root_dir: directory for saving review results
        :param max_lines: maximum number of lines in diff chunk
        :param team_styles: team code style description
        """
        self.llm = llm
        self.gitlab = gitlab
        self.root_dir = root_dir
        self.max_lines = max_lines
        self.styles = team_styles

    def _split_large_diffs(self, changes):
        """Split large diffs into smaller chunks."""
        new_changes = []
        for change in changes:
            diff_lines = change["diff"].splitlines()
            if len(diff_lines) > self.max_lines:
                for i in range(0, len(diff_lines), self.max_lines):
                    chunk_lines = diff_lines[i:i + self.max_lines]
                    new_changes.append({
                        "new_path": change["new_path"],
                        "diff": "\n".join(chunk_lines),
                        "chunk_index": i // self.max_lines + 1
                    })
            else:
                new_changes.append(change)
        return new_changes

    def _make_prompt(self, file_path, diff):
        """Prepare review prompt for LLM."""
        return f"""
        Ты опытный ревьюер кода.
        Твоя задача проверить мердж реквест, найти проблемные места и написать предложения по их устранению.
        Анализируйте ТОЛЬКО изменённые и добавленные строки.

        Модуль: {file_path}

        Код стайл команды (для контекста):
        {self.styles}

        Важно:
        - Дай рекомендации ТОЛЬКО по изменённым и добавленным строкам.
        - Не давай рекомендации по поводу документации и описании изменений.
        - Если замечаний НЕТ — выведи только строку "Нет замечаний".
        - Если замечания ЕСТЬ — перечисли их, указывая место, описание проблемы и пример исправления.
        - В конце НИКОГДА не выводи "Нет замечаний", если уже были перечислены проблемы.


        Формат ответа:
        1. <указание на место с проблемой> описание проблемы, варианты решения, пример решения
        2. ...

        Код для проверки:
        {diff}
        """

    def _review_one_file_or_chunk(self, part):
        """Review one file (or chunk for large files)."""
        file_path = part['new_path']
        diff = part['diff']
        print(f"[INFO] Ревью файла {file_path}...")

        prompt = self._make_prompt(file_path, diff)
        review = self.llm.generate_response(prompt, max_tokens=2048, temperature=0.4)

        print(review)
        return file_path, review

    def review_mr(self, mr_url):
        """Run review workflow for a merge request."""
        project_path, mr_iid = self.gitlab.parse_mr_url(mr_url.strip())
        changes = self.gitlab.get_mr_changes(project_path, mr_iid)

        changes = self._split_large_diffs(changes)

        all_reviews = []
        start_total = time.perf_counter()

        for file_or_chunk in changes:
            file_path, review = self._review_one_file_or_chunk(file_or_chunk)
            all_reviews.append(f"### {file_path}\n{review}\n")

        elapsed_total = time.perf_counter() - start_total
        print(f"[INFO] Полная генерация заняла {elapsed_total:.2f} сек")

        result_text = "\n\n".join(all_reviews)

        print("\n\n==================== КОД-РЕВЬЮ ====================\n\n")
        print(result_text)

        output_dir = Path(self.root_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{mr_iid}.md", "w", encoding="utf-8") as f:
            f.write(result_text)
