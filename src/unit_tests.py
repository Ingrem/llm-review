from src.llm_workflow import LlmWorkflow


llm = LlmWorkflow()

prompt = "Напиши функцию квадрата чисел"
review = llm.generate_response(prompt, max_tokens=8192, temperature=0.3)
print(review)
