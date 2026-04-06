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

Version 1.1.0 add API support for local models, tested with llama.cpp server and gemini 4 model.

### 4. Use api
Although this tool is designed to work directly with a local LLM, you may find it more convenient to run the model on a separate server via an API.

To enable API mode, add the following lines to your .env file:
```env
USE_API=true
API_PORT=8000
```
## Usage
Run the script using the following command in your terminal
`python run.py`
