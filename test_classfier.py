from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import classification_report
from datasets import load_dataset
import torch

# 경로
MODEL_PATH = "./classifier_model"
DATA_FILE = "classification_data.jsonl"

# 모델 및 토크나이저 로드
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.eval()

# 데이터셋 로드
dataset = load_dataset("json", data_files={"test": DATA_FILE})["test"]

# 전처리
def preprocess(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

encoded = dataset.map(preprocess)
encoded.set_format(type="torch", columns=["input_ids", "attention_mask"])
labels = dataset["label"]

# 추론
preds = []
with torch.no_grad():
    for batch in torch.utils.data.DataLoader(encoded, batch_size=8):
        outputs = model(**{k: v for k, v in batch.items()})
        pred = outputs.logits.argmax(dim=-1).tolist()
        preds.extend(pred)

# 성능 평가
print(classification_report(labels, preds, target_names=["일반질문", "포트폴리오질문"]))
