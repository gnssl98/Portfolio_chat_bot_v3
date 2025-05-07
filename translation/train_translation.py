# Re-import and execute the script after state reset

import os
from datasets import load_dataset, Dataset
from transformers import (
    MBartForConditionalGeneration, MBart50TokenizerFast,
    MarianMTModel, MarianTokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType
import torch

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 경로 설정
EN_KO_MODEL_PATH = r"D:\dataset\fine_tuned_model\translation_model\en_ko"  # MBART
KO_EN_MODEL_PATH = r"D:\dataset\fine_tuned_model\translation_model\ko_en"  # Marian
OUTPUT_DIR = r"D:\dataset\fine_tuned_model\translation_model"
EN_KO_OUTPUT = os.path.join(OUTPUT_DIR, "en_ko_finetuned")
KO_EN_OUTPUT = os.path.join(OUTPUT_DIR, "ko_en_finetuned")

# JSONL 파일 로드
mbart_dataset = load_dataset("json", data_files="./mbart_translation_pairs.jsonl", split="train")
marian_dataset = load_dataset("json", data_files="./translation_gpt_pairs.jsonl", split="train")

# MBART 학습 함수 (en → ko)
def train_mbart_en_ko(dataset):
    tokenizer = MBart50TokenizerFast.from_pretrained(EN_KO_MODEL_PATH, local_files_only=True)
    model = MBartForConditionalGeneration.from_pretrained(EN_KO_MODEL_PATH, local_files_only=True).to(device)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1, bias="none", task_type=TaskType.SEQ_2_SEQ_LM
    ))

    def preprocess(example):
        tokenizer.src_lang = example["src_lang"]
        tokenizer.tgt_lang = example["tgt_lang"]

        inputs = tokenizer(example["src"], max_length=128, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(example["tgt"], max_length=128, truncation=True, padding="max_length")
        inputs["labels"] = labels["input_ids"]
        inputs["forced_bos_token_id"] = tokenizer.lang_code_to_id[example["tgt_lang"]]
        return inputs

    tokenized = dataset.map(preprocess)
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=EN_KO_OUTPUT,
            per_device_train_batch_size=8,
            num_train_epochs=5,
            logging_dir=os.path.join(EN_KO_OUTPUT, "logs"),
            save_total_limit=1, save_strategy="epoch", fp16=torch.cuda.is_available(), report_to="none"
        ),
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )
    trainer.train()
    model.save_pretrained(EN_KO_OUTPUT)
    tokenizer.save_pretrained(EN_KO_OUTPUT)

# Marian 학습 함수 (ko → en)
def train_marian_ko_en(dataset):
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en").to(device)

    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1, bias="none", task_type=TaskType.SEQ_2_SEQ_LM
    ))

    def preprocess(example):
        inputs = tokenizer(example["translation"]["ko"], max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(example["translation"]["en"], max_length=128, truncation=True, padding="max_length").input_ids
        inputs["labels"] = labels
        return inputs

    tokenized = dataset.map(preprocess)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=KO_EN_OUTPUT,
            per_device_train_batch_size=8,
            num_train_epochs=5,
            logging_dir=os.path.join(KO_EN_OUTPUT, "logs"),
            save_total_limit=1,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            report_to="none"
        ),
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()
    model.save_pretrained(KO_EN_OUTPUT)
    tokenizer.save_pretrained(KO_EN_OUTPUT)

# 학습 실행
train_mbart_en_ko(mbart_dataset)
train_marian_ko_en(marian_dataset)
