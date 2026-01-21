import os
import json
from pathlib import Path
from paddleocr import PaddleOCRVL
import paddleocr

def process_single_pdf():
    # 경로 설정
    pdf_folder = Path("pdf")
    output_folder = Path("output/paddle")
    
    # 출력 폴더가 없으면 생성
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 특정 PDF 파일 지정 (paddle_ocr_test.py 방식과 동일)
    pdf_file = pdf_folder / "middle.pdf"
    
    if not pdf_file.exists():
        print(f"'{pdf_file}' 파일을 찾을 수 없습니다.")
        return
    
    # PaddleOCR-VL 초기화 (PDF 문서 파싱에 최적화된 최신 모델)
    # device="gpu:0" 또는 device="gpu" 로 GPU 사용
    pipeline = PaddleOCRVL(device="gpu:0")
    
    print(f"처리 중: {pdf_file.name}\n")
    
    try:
        # PDF 파일명 (확장자 제외)
        pdf_name = pdf_file.stem
        
        # 해당 PDF를 위한 출력 폴더 생성
        pdf_output_folder = output_folder / pdf_name
        pdf_output_folder.mkdir(exist_ok=True)
        
        # OCR 실행
        result = pipeline.predict(str(pdf_file))
        
        # PDF 전체를 하나의 Markdown 파일로 저장
        markdown_list = []
        markdown_images = []
        json_results = []
        
        for res in result:
            # Markdown 정보 수집
            md_info = res.markdown
            markdown_list.append(md_info)
            markdown_images.append(md_info.get("markdown_images", {}))
            
            # JSON 정보 수집 (전체 결과 저장용)
            json_results.append(res.json)
        
        # Markdown 페이지들을 하나로 합치기
        markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)
        
        # 통합 Markdown 파일 저장
        mkd_file_path = pdf_output_folder / f"{pdf_name}.md"
        mkd_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(mkd_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_texts)
        
        print(f"  → Markdown 저장: {mkd_file_path}")
        
        # Markdown에서 참조하는 이미지들 저장
        for item in markdown_images:
            if item:
                for path, image in item.items():
                    file_path = pdf_output_folder / path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(file_path)
        
        # 통합 JSON 파일 저장
        json_file_path = pdf_output_folder / f"{pdf_name}.json"
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"  → JSON 저장: {json_file_path}")
        
        print(f"\n✓ 완료: {pdf_name} → {pdf_output_folder}")
        
    except Exception as e:
        print(f"✗ 오류 발생 ({pdf_file.name}): {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print(f"PaddleOCR version: {paddleocr.__version__}")
    print("PaddleOCR-VL을 사용하여 중급_KB골든라이프.pdf 처리를 시작합니다...\n")
    
    process_single_pdf()
