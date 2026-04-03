import os
from dotenv import load_dotenv
from src.gitlab_workflow import GitlabWorkflow
from src.llm_workflow import LlmWorkflow
from src.review_workflow import ReviewWorkflow
from code_styles.code_styles_list import styles
from config import ROOT_DIR


load_dotenv()
GITLAB_TOKEN = os.environ.get("GITLAB_TOKEN", "")
MR_LINK = os.environ.get("MR_LINK", "").strip()

print("Connecting to LLM...")
llm = LlmWorkflow()
print("Connecting to GitLab...")
gitlab = GitlabWorkflow(gitlab_token=GITLAB_TOKEN)
reviewer = ReviewWorkflow(llm, gitlab, ROOT_DIR, styles)

reviewer.review_mr(MR_LINK)
