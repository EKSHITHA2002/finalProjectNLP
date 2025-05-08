import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# Model and tokenizer
model_id = "meta-llama/Meta-Llama-3-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=True
)
model.eval()

# Prompt template
def format_prompt(question):
    return f"[INST] {question.strip()} [/INST]"

# Load dataset
dataset = load_dataset("gsm8k", "main", split="test[:10]")  # Limit for demo

# Output
output_data = []

for sample in tqdm(dataset, desc="Generating answers"):
    question = sample["question"]
    gold_answer = sample["answer"]

    prompt = format_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = decoded.replace(prompt, "").strip()

    output_data.append({
        "question": question,
        "gold": gold_answer,
        "pred": pred,
        "correct": gold_answer.split("####")[-1].strip() in pred
    })

# Save output
with open("llama3_gsm8k_output.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("✅ Output saved to llama3_gsm8k_output.json")
