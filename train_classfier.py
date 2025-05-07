import os
import json
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from sklearn.metrics import accuracy_score, f1_score

# 파일 경로 및 설정
DATA_FILE = "classification_data.jsonl"
MODEL_NAME = "klue/bert-base"
OUTPUT_DIR = "./classifier_model"

# 1. 데이터 로드
dataset = load_dataset("json", data_files={"train": DATA_FILE})["train"]

# 2. 토크나이저 및 모델 초기화
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 3. 전처리 함수
def preprocess(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 4. 평가 지표
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

# 5. 트레이닝 파라미터 설정
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch",  # 저장만 주기적으로 수행
    logging_steps=10,
    fp16=torch.cuda.is_available()
)

# 6. Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    compute_metrics=compute_metrics,
)

# 7. 학습 실행
trainer.train()

# 8. 모델 저장
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"학습 완료. 분류 모델이 '{OUTPUT_DIR}'에 저장되었습니다.")
