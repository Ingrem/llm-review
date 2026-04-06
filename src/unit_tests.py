from src.llm_workflow import LlmWorkflow

llm = LlmWorkflow(True, 8000)

prompt = "Напиши функцию квадрата чисел"
review = llm.generate_response(prompt)
print("\n\n==================== RESULT ====================\n\n")
print(review)
