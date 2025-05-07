import os
from transformers import (
    MarianTokenizer, MarianMTModel,
    MBart50TokenizerFast, MBartForConditionalGeneration
)

def download_translation_model(model_name, save_dir):
    """Helsinki-NLP MarianMT 모델을 로컬에 다운로드합니다."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"모델 다운로드 경로: {save_dir}")
        print(f"{model_name} 모델 다운로드 중...")

        # 토크나이저 다운로드
        print("토크나이저 다운로드 중...")
        tokenizer = MarianTokenizer.from_pretrained(
            model_name,
            cache_dir=save_dir
        )

        # 모델 다운로드
        print("모델 다운로드 중...")
        model = MarianMTModel.from_pretrained(
            model_name,
            cache_dir=save_dir
        )

        # 모델 및 토크나이저 저장
        print("모델 및 토크나이저 저장 중...")
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)

        print(f"모델 및 토크나이저가 성공적으로 다운로드되었습니다: {save_dir}")
        print("\n모델 정보:")
        print(f"모델 이름: {model_name}")
        print(f"모델 크기: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M 파라미터")
        print(f"저장 경로: {save_dir}")
        return True
    except Exception as e:
        print(f"모델 다운로드 중 오류 발생: {str(e)}")
        return False
    
def download_mbart_model(model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    print(f"[MBart] 모델 다운로드 중: {model_name}")
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name, cache_dir=save_dir)
    model = MBartForConditionalGeneration.from_pretrained(model_name, cache_dir=save_dir)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    print(f"저장 완료: {save_dir} | 파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")


if __name__ == "__main__":
    # 영어->한국어
    download_translation_model(
        "Helsinki-NLP/opus-mt-tc-big-en-ko",
        r"D:\dataset\fine_tuned_model\translation_model\ko_en"
    )
    # 한국어->영어
    download_mbart_model(
        "facebook/mbart-large-50-many-to-many-mmt",
        r"D:\dataset\fine_tuned_model\translation_model\en_ko"
    )
