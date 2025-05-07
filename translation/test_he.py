import os
from typing import List
import torch
import evaluate
from transformers import (
    MarianTokenizer, MarianMTModel,
    MBart50TokenizerFast, MBartForConditionalGeneration
)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 경로
EN_KO_MODEL_PATH = r"D:\dataset\fine_tuned_model\translation_model\en_ko"  
KO_EN_MODEL_PATH = r"D:\dataset\fine_tuned_model\translation_model\ko_en"  


# 테스트 문장쌍
test_pairs = [
    {"en": "I am a student at Gachon University.", "ko": "저는 가천대학교 학생입니다."},
    {"en": "My major is computer engineering.", "ko": "제 전공은 컴퓨터공학입니다."},
    {"en": "I am interested in artificial intelligence.", "ko": "저는 인공지능에 관심이 많습니다."},
    {"en": "I like reading research papers.", "ko": "저는 논문 읽는 것을 좋아합니다."},
    {"en": "My strength is solving problems.", "ko": "저의 강점은 문제 해결 능력입니다."}
]

# BLEU metric 로드
bleu = evaluate.load("sacrebleu")

# Marian 모델 로드 (en → ko)
def load_marian_model(model_path):
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model

# mBART 모델 로드 (ko → en)
def load_mbart_model(model_path):
    tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
    model = MBartForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model

# Marian 번역 함수
def translate_marian(text: str, tokenizer, model) -> str:
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# mBART 번역 함수
def translate_mbart(text: str, tokenizer, model, src_lang: str, tgt_lang: str) -> str:
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=128,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 모델 로딩
en_ko_tokenizer, en_ko_model = load_mbart_model(EN_KO_MODEL_PATH)  # 영어 → 한국어
ko_en_tokenizer, ko_en_model = load_marian_model(KO_EN_MODEL_PATH)  # 한국어 → 영어


# 영어 → 한국어
print("=== [EN → KO 번역 및 평가] ===\n")
en_ko_preds: List[str] = []
en_ko_refs: List[List[str]] = []

for pair in test_pairs:
    src = pair["en"]
    tgt = pair["ko"]
    pred = translate_mbart(src, en_ko_tokenizer, en_ko_model, src_lang="en_XX", tgt_lang="ko_KR")
    en_ko_preds.append(pred)
    en_ko_refs.append([tgt])
    print(f"[EN] {src}")
    print(f"[PREDICTED KO] {pred}")
    print(f"[TARGET KO]    {tgt}\n")

en_ko_bleu = bleu.compute(predictions=en_ko_preds, references=en_ko_refs)
print(f"BLEU score (en → ko): {en_ko_bleu['score']:.2f}\n")

# 한국어 → 영어
print("=== [KO → EN 번역 및 평가] ===\n")
ko_en_preds: List[str] = []
ko_en_refs: List[List[str]] = []

for pair in test_pairs:
    src = pair["ko"]
    tgt = pair["en"]
    pred = translate_marian(src, ko_en_tokenizer, ko_en_model)
    ko_en_preds.append(pred)
    ko_en_refs.append([tgt])
    print(f"[KO] {src}")
    print(f"[PREDICTED EN] {pred}")
    print(f"[TARGET EN]    {tgt}\n")

ko_en_bleu = bleu.compute(predictions=ko_en_preds, references=ko_en_refs)
print(f"BLEU score (ko → en): {ko_en_bleu['score']:.2f}")
