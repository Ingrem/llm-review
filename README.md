# llm-review
A simple tool to automate GitLab Merge Request reviews using Large Language Models.

## Setup Instructions

### 1. Configuration
Create a file named `.env` in the root directory of the project.

### 2. Environment Variables
Add your GitLab Personal Access Token and the link to the Merge Request you wish to review.

**`.env` sample:**
```env
GITLAB_TOKEN=your-token
MR_LINK="
https://git.something.com/some-repo/some-project/-/merge_requests/123
"
```

## Usage
Run the script using the following command in your terminal
`python run.py`
