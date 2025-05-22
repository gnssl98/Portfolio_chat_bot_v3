import os
import json
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType
from dotenv import load_dotenv
import torch

# ===== 설정 =====
load_dotenv()
BASE_MODEL = r"D:\dataset\fine_tuned_model\flan-t5-large"
DATA_PATH = "qa_data_filtered_soft_korean.jsonl"
OUTPUT_DIR = r"D:\dataset\fine_tuned_model\flan-t5-large-lora"

# ===== 데이터 로딩 =====
def load_lora_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line.strip())
            question = ex["question"]
            context = ex["context"]
            answer = ex["answer"]
            prompt = f"질문: {question}\n문맥: {context}"
            data.append({"input": prompt, "output": answer})
    return Dataset.from_list(data)

# ===== 토크나이저 및 모델 =====
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to("cuda")

# ===== LoRA 구성 =====
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    task_type=TaskType.SEQ_2_SEQ_LM,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, peft_config)

# ===== 데이터 전처리 함수 =====
def preprocess(example):
    model_inputs = tokenizer(
        example["input"], max_length=512, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        example["output"], max_length=128, truncation=True, padding="max_length"
    )

    # ❗ pad 토큰은 -100으로 마스킹 (loss 계산에서 무시)
    labels_ids = [
        token if token != tokenizer.pad_token_id else -100
        for token in labels["input_ids"]
    ]

    model_inputs["labels"] = labels_ids
    return model_inputs

# ===== 데이터셋 준비 =====
dataset = load_lora_dataset(DATA_PATH)
dataset = dataset.map(preprocess, batched=False)

# ===== 트레이너 설정 =====
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=False,  # 먼저 안정성 확보 위해 꺼둠
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

# ===== 학습 시작 =====
if __name__ == "__main__":
    model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
