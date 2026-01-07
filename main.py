from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Escolhe o modelo (LLaMA 3 ou Mistral)
MODEL_NAME = "meta-llama/Meta-Llama-3-8b"  # ou "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

@app.post("/lamux")
def lamux(prompt: str):
    input_text = f"Você é Lamux, nunca mencione LLaMA ou Mistral.\nUser: {prompt}\nLamux:"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_new_tokens=200)
    return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}
