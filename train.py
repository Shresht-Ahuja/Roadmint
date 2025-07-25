import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import os

# === Model Setup ===
model_id = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Add padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# LoRA configuration - simplified target modules
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # Just attention layers
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === Dataset Setup ===
dataset = load_dataset("json", data_files="roadmap_dataset_large.jsonl", split="train")

def format_prompt(example):
    return {
        "text": f"### Instruction:\nCreate a learning roadmap for: {example['prompt']}\n\n### Response:\n{example['completion']}{tokenizer.eos_token}"
    }

dataset = dataset.map(format_prompt)

# Split dataset
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# === Tokenization ===
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Apply tokenization
train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing train dataset",
)

eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing eval dataset",
)

# === Training Arguments - More compatible version ===
training_args = TrainingArguments(
    output_dir="./gemma-roadmap-lora",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    save_total_limit=2,
    gradient_checkpointing=True,
    report_to=[],  # Empty list instead of None
)

# === Data Collator ===
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# === Train ===
print("===== Starting Training =====")
print(f"Training samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")

try:
    trainer.train()
    print("✅ Training completed successfully!")
except Exception as e:
    print(f"❌ Training failed: {e}")
    # Save partial model if training fails
    trainer.save_model("./gemma-roadmap-lora-partial")

# === Save Model ===
save_path = "./gemma-roadmap-lora-final"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ Model saved to {save_path}")

# === Test Generation ===
def generate_roadmap(prompt, max_length=300):
    model.eval()
    input_text = f"### Instruction:\nCreate a learning roadmap for: {prompt}\n\n### Response:\n"
    
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:\n" in response:
        return response.split("### Response:\n")[1].strip()
    return response

# Test the model
print("\n🧪 Testing the model:")
test_prompt = "Python programming"
result = generate_roadmap(test_prompt)
print(f"Input: {test_prompt}")
print(f"Output: {result}")