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
    pdf_file = pdf_folder / "goldenlife.pdf"
    
    if not pdf_file.exists():
        print(f"'{pdf_file}' 파일을 찾을 수 없습니다.")
        return
    
    # PaddleOCR-VL 초기화 (PDF 문서 파싱에 최적화된 최신 모델)
    # device="gpu:0" 또는 device="gpu" 로 GPU 사용
    # === 기존 코드 (파라미터 최소화) ===
    # pipeline = PaddleOCRVL(device="gpu:0")
    
    # === 수정된 코드 (첫 페이지 파싱 개선을 위한 파라미터 추가) ===
    # === 기존 코드 (최소화) ===
    # pipeline = PaddleOCRVL(device="gpu:0")
    
    # === 수정된 코드 (문서 전처리 활성화 버전) ===
    pipeline = PaddleOCRVL(
        device="gpu:0",
        use_doc_orientation_classify=True,  # ✅ 문서 방향 분류 활성화
        use_doc_unwarping=True,             # ✅ 문서 왜곡 보정 활성화 (스캔 문서 개선)
        use_layout_detection=True,          # ✅ 레이아웃 감지 활성화
        use_chart_recognition=False,        # 차트 인식 (필요시 True로 변경)
        format_block_content=False,         # 블록 콘텐츠 Markdown 포맷팅
    )
    
    print(f"처리 중: {pdf_file.name}\n")
    
    try:
        # PDF 파일명 (확장자 제외)
        pdf_name = pdf_file.stem
        
        # 해당 PDF를 위한 출력 폴더 생성
        pdf_output_folder = output_folder / pdf_name
        pdf_output_folder.mkdir(exist_ok=True)
        
        # OCR 실행
        result = pipeline.predict(str(pdf_file))
        
        # === 디버깅: 첫 페이지 파싱 결과 구조 확인 ===
        print(f"전체 페이지 수: {len(result)}")
        if result:
            first_page = result[0]
            print("\n[첫 페이지 결과 객체 속성]")
            print(f"- 객체 타입: {type(first_page)}")
            print(f"- 사용 가능한 속성: {dir(first_page)}")
            
            # JSON 결과 구조 확인
            print("\n[JSON 결과 구조]")
            json_result = first_page.json
            print(f"- JSON 키: {json_result.keys() if isinstance(json_result, dict) else 'Not a dict'}")
            
            # parsing_res_list 찾기 (json 내부에 있을 수 있음)
            if isinstance(json_result, dict) and 'parsing_res_list' in json_result:
                print(f"- 감지된 블록 수: {len(json_result['parsing_res_list'])}")
                print("- 첫 5개 블록 미리보기:")
                for i, block in enumerate(json_result['parsing_res_list'][:5]):
                    block_label = block.get('block_label', 'Unknown')
                    block_content = block.get('block_content', '')[:100]
                    print(f"  [{i}] Label: {block_label}, Content: {block_content}...")
            else:
                print("- parsing_res_list를 JSON에서 찾을 수 없습니다. 전체 JSON 구조:")
                import json as json_lib
                print(json_lib.dumps(json_result, ensure_ascii=False, indent=2, default=str)[:1000])
        print()
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
