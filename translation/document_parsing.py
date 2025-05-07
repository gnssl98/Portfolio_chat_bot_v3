import os
import glob
import pdfplumber

DATA_DIR = './portfolio_data'
OUTPUT_FILE = 'document.txt'

def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text

def extract_texts_from_portfolio_data(data_dir=DATA_DIR, output_file=OUTPUT_FILE):
    # 하위 디렉토리까지 모두 탐색
    all_files = glob.glob(os.path.join(data_dir, '**', '*.txt'), recursive=True) + \
                glob.glob(os.path.join(data_dir, '**', '*.pdf'), recursive=True)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in all_files:
            out_f.write(f'\n===== 파일: {os.path.relpath(file_path, data_dir)} =====\n')
            if file_path.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                out_f.write(content + '\n')
            elif file_path.lower().endswith('.pdf'):
                content = extract_text_from_pdf(file_path)
                out_f.write(content + '\n')
            else:
                out_f.write('지원하지 않는 파일 형식입니다.\n')
    print(f'모든 문서가 {output_file}에 저장되었습니다.')

if __name__ == '__main__':
    extract_texts_from_portfolio_data() 