from transformers import (
    MarianMTModel, MarianTokenizer,
    MBartForConditionalGeneration, MBart50TokenizerFast
)
import torch
import sacrebleu
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MBART 관련 함수
def load_mbart_model(model_dir):
    tokenizer = MBart50TokenizerFast.from_pretrained(model_dir)
    model = MBartForConditionalGeneration.from_pretrained(model_dir).to(device)
    return tokenizer, model

def translate_mbart(text, tokenizer, model, src_lang="en_XX", tgt_lang="ko_KR"):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=128,
            num_beams=5,
            no_repeat_ngram_size=2
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Marian 관련 함수
def load_marian_model(model_dir):
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir).to(device)
    return tokenizer, model

def translate_marian(text, tokenizer, model):
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            no_repeat_ngram_size=2
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# BLEU 평가 함수 (공통)
def evaluate_bleu(pairs, tokenizer, model, translate_func, source_lang, target_lang):
    predictions = []
    references = []

    for pair in pairs:
        src_text = pair[source_lang]
        tgt_text = pair[target_lang]
        pred = translate_func(src_text, tokenizer, model)
        predictions.append(pred)
        references.append([tgt_text])
        print(f"\n[SOURCE] {src_text}")
        print(f"[PREDICTED] {pred}")
        print(f"[REFERENCE] {tgt_text}")

    bleu = sacrebleu.corpus_bleu(predictions, references)
    print(f"\nBLEU score ({source_lang} → {target_lang}): {bleu.score:.2f}")
    return bleu.score


# 테스트 문장쌍
en_ko_pairs = [
    {"en": "I am a student at Gachon University.", "ko": "저는 가천대학교 학생입니다."},
    {"en": "My major is computer engineering.", "ko": "제 전공은 컴퓨터공학입니다."},
    {"en": "I am interested in artificial intelligence.", "ko": "저는 인공지능에 관심이 많습니다."},
]

ko_en_pairs = [
    {"ko": "저는 인공지능 개발자입니다.", "en": "I am an AI developer."},
    {"ko": "저는 다양한 프로젝트를 진행한 경험이 있습니다.", "en": "I have experience working on various projects."},
    {"ko": "저의 강점은 문제 해결 능력입니다.", "en": "My strength is problem-solving skills."},
]


# 경로 설정
EN_KO_MODEL_PATH = r"D:\dataset\fine_tuned_model\translation_model\en_ko_finetuned"  # MBART
KO_EN_MODEL_PATH = r"D:\dataset\fine_tuned_model\translation_model\ko_en_finetuned"  # Marian

# 모델 로드
en_ko_tokenizer, en_ko_model = load_mbart_model(EN_KO_MODEL_PATH)
ko_en_tokenizer, ko_en_model = load_marian_model(KO_EN_MODEL_PATH)

# 평가 실행
evaluate_bleu(
    en_ko_pairs,
    en_ko_tokenizer,
    en_ko_model,
    lambda text, tokenizer, model: translate_mbart(text, tokenizer, model, src_lang="en_XX", tgt_lang="ko_KR"),
    source_lang="en",
    target_lang="ko"
)

evaluate_bleu(
    ko_en_pairs,
    ko_en_tokenizer,
    ko_en_model,
    translate_marian,
    source_lang="ko",
    target_lang="en"
)
