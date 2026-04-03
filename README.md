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

### 3. Model downloading (new)
Version 0.1.0 used Qwen2.5 14b and have usefull, but lots of false positive results. 
Model was downloaded with transformers automatically. Model require ~12GB of VRAM.

Version 1.0.0 use Qwen3.5-35B-A3B and have production ready results.
Model require manual downloading, put it into ./llm-review/models/Qwen3.5-35B-A3B-Q5_K_M.gguf. 
This model can be partially swapped to RAM and work with ~20 tokens/second.

## Usage
Run the script using the following command in your terminal
`python run.py`
