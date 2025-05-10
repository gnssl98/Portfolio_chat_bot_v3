import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from dotenv import load_dotenv
from transformers import Seq2SeqTrainingArguments 

# 환경 설정
load_dotenv()
BASE_MODEL = "google/flan-t5-large"
DATA_PATH = "./flan_t5_qa_dataset.jsonl"  # <- JSONL 형식으로 변경
OUTPUT_DIR = "./flan_t5_lora_finetuned"

# 데이터 로딩 (JSONL 방식)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
dataset = Dataset.from_list(data)

# 토크나이저 및 모델 로딩
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16).to("cuda")

# LoRA 구성
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)
model = get_peft_model(model, lora_config)

# 전처리
def preprocess(examples):
    model_inputs = tokenizer(
        examples["input"], max_length=512, padding="max_length", truncation=True
    )
    labels = tokenizer(
        examples["output"], max_length=128, padding="max_length", truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

# 학습 설정
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=5,
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    predict_with_generate=True,
)

# Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 학습
trainer.train()

# 저장
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n LoRA 학습 완료! 모델 저장 위치: {OUTPUT_DIR}")
