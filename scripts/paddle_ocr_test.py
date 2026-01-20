import os
import json
from pathlib import Path
from paddleocr import PaddleOCRVL
import paddleocr

def process_pdfs_in_folder():
    # 경로 설정
    pdf_folder = Path("pdf")
    output_folder = Path("output/paddle")
    
    # 출력 폴더가 없으면 생성
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # PaddleOCR-VL 초기화 (PDF 문서 파싱에 최적화된 최신 모델)
    # device="gpu:0" 또는 device="gpu" 로 GPU 사용
    pipeline = PaddleOCRVL(device="gpu:0")
    
    # .pdf 폴더 내의 모든 PDF 파일 찾기
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"'{pdf_folder}' 폴더에 PDF 파일이 없습니다.")
        return
    
    print(f"총 {len(pdf_files)}개의 PDF 파일을 처리합니다.\n")
    
    # 각 PDF 파일 처리
    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"[{idx}/{len(pdf_files)}] 처리 중: {pdf_file.name}")
        
        try:
            # PDF 파일명 (확장자 제외)
            pdf_name = pdf_file.stem
            
            # 해당 PDF를 위한 출력 폴더 생성
            pdf_output_folder = output_folder / pdf_name
            pdf_output_folder.mkdir(exist_ok=True)
            
            # OCR 실행
            result = pipeline.predict(str(pdf_file))
            
            # ===== 페이지별 저장 (기존 코드 - 주석처리) =====
            # for res in result:
            #     # JSON 형식으로 저장
            #     res.save_to_json(save_path=str(pdf_output_folder))
            #     
            #     # Markdown 형식으로 저장
            #     res.save_to_markdown(save_path=str(pdf_output_folder))
            #     
            #     # 결과 출력 (옵션)
            #     res.print()
            
            # ===== PDF 전체를 하나의 Markdown 파일로 저장 (신규) =====
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
            
            print(f"✓ 완료: {pdf_name} → {pdf_output_folder}\n")
            
        except Exception as e:
            print(f"✗ 오류 발생 ({pdf_file.name}): {str(e)}\n")
            continue
    
    print(f"\n모든 작업 완료! 결과는 '{output_folder}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    print(f"PaddleOCR version: {paddleocr.__version__}")
    print("PaddleOCR-VL을 사용하여 PDF 처리를 시작합니다...\n")
    
    process_pdfs_in_folder()