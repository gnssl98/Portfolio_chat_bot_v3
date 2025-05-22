import os
import json
from tqdm import tqdm
from evaluate import load as load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ===== 경로 =====
BASE_MODEL = r"D:\dataset\fine_tuned_model\flan-t5-large"
LORA_DIR = r"D:\dataset\fine_tuned_model\flan-t5-large-lora"
#DATA_PATH = "evaluation_data.jsonl"
DATA_PATH = "qa_data_filtered_soft_korean.jsonl"
SAVE_OUTPUTS = "prediction_results.jsonl"

# ===== BLEU/ROUGE 메트릭 로딩 =====
bleu = load_metric("bleu")
rouge = load_metric("rouge")

# ===== 모델 불러오기 =====
def load_model(use_lora=False):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to("cuda")
    if use_lora:
        model = PeftModel.from_pretrained(model, LORA_DIR).to("cuda")
    return tokenizer, model

# ===== 평가 함수 =====
def evaluate_model(tokenizer, model, dataset, max_length=128, save_path=None):
    predictions = []
    references = []
    records = []

    for ex in tqdm(dataset, desc="Evaluating"):
        input_text = f"질문: {ex['question']}\n문맥: {ex['context']}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to("cuda")
        output = model.generate(**inputs, max_new_tokens=max_length)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        predictions.append(decoded)
        references.append([ex["answer"]])

        records.append({
            "question": ex["question"],
            "context": ex["context"],
            "prediction": decoded,
            "answer": ex["answer"]
        })

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n💾 예측 결과 저장됨: {save_path}")

    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=[r[0] for r in references], use_stemmer=True)

    return bleu_score["bleu"], rouge_score["rougeL"]

# ===== 데이터 로딩 =====
def load_eval_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data

# ===== 실행 =====
if __name__ == "__main__":
    eval_data = load_eval_data(DATA_PATH)

    print("Evaluating Base Model...")
    tokenizer, base_model = load_model(use_lora=False)
    bleu_base, rouge_base = evaluate_model(tokenizer, base_model, eval_data, save_path="base_model_predictions.jsonl")

    print("\n Evaluating LoRA-Finetuned Model...")
    tokenizer, lora_model = load_model(use_lora=True)
    bleu_lora, rouge_lora = evaluate_model(tokenizer, lora_model, eval_data, save_path="lora_model_predictions.jsonl")

    print("\n 평가 결과 요약:")
    print(f"Base Model → BLEU: {bleu_base:.4f}, ROUGE-L: {rouge_base:.4f}")
    print(f"LoRA  Model → BLEU: {bleu_lora:.4f}, ROUGE-L: {rouge_lora:.4f}")
