from transformers import AutoModelForCausalLM, AutoTokenizer

# code llama 
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf")