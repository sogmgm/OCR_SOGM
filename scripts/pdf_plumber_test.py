#!/usr/bin/env python3
"""
PDF 이미지에서 bbox 정보를 기반으로 table과 figure를 crop하는 스크립트
doclayout-yolo의 JSON 결과를 입력으로 받아 crop된 이미지를 저장합니다.
"""

import json
import os
from pathlib import Path
from PIL import Image

# 경로 설정
DOCLAYOUT_OUTPUT_DIR = Path("/workspace/output/doclayout-yolo")
PDF_IMG_DIR = Path("/workspace/pdf_img")
CROP_OUTPUT_DIR = Path("/workspace/output/doclayout-yolo/pdf_crop")

# 관심 있는 클래스 (table과 figure만 crop)
TARGET_CLASSES = ["table", "figure"]

def crop_image_from_bbox(image_path: str, bbox: list, output_path: str) -> bool:
    """
    이미지에서 bbox 영역을 crop하여 저장합니다.
    
    Args:
        image_path: 원본 이미지 경로
        bbox: [x0, top, x1, bottom] 형태의 bbox 좌표
        output_path: 저장할 crop 이미지 경로
    
    Returns:
        성공 여부
    """
    try:
        # 이미지 열기
        img = Image.open(image_path)
        
        # bbox는 [x0, top, x1, bottom] 형태
        x0, top, x1, bottom = bbox
        
        # crop 수행 (PIL에서는 (left, top, right, bottom))
        cropped = img.crop((int(x0), int(top), int(x1), int(bottom)))
        
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 저장
        cropped.save(output_path)
        return True
    except Exception as e:
        print(f"Error cropping image {image_path}: {e}")
        return False


def process_json_file(json_path: Path) -> int:
    """
    하나의 JSON 파일을 처리하고 해당하는 이미지들을 crop합니다.
    
    Args:
        json_path: 처리할 JSON 파일 경로
    
    Returns:
        crop된 이미지 수
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 이미지 경로 가져오기 (상대 경로)
        image_relative_path = data.get("image", "")
        image_path = Path("/workspace") / image_relative_path
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            return 0
        
        # detections에서 table과 figure만 필터링
        detections = data.get("detections", [])
        target_detections = [
            d for d in detections 
            if d.get("class_name") in TARGET_CLASSES
        ]
        
        if not target_detections:
            return 0
        
        # JSON 파일 이름을 기반으로 output 디렉토리 결정
        json_relative = json_path.relative_to(DOCLAYOUT_OUTPUT_DIR)
        category_name = json_relative.parts[0]  # 예: "고급_가계대출"
        page_name = json_relative.stem  # 예: "page_0005_result"
        
        # crop 저장 경로
        crop_dir = CROP_OUTPUT_DIR / category_name / page_name
        os.makedirs(crop_dir, exist_ok=True)
        
        # 각 detection을 crop하여 저장
        cropped_count = 0
        for idx, detection in enumerate(target_detections):
            class_name = detection.get("class_name", "unknown")
            confidence = detection.get("confidence", 0)
            bbox = detection.get("bbox", [])
            
            if not bbox:
                continue
            
            # output 파일명: {page_name}_{class_name}_{idx}_{confidence}.png
            confidence_str = f"{confidence:.2f}"
            output_filename = f"{page_name}_{class_name}_{idx}_{confidence_str}.png"
            output_path = crop_dir / output_filename
            
            # crop 수행
            if crop_image_from_bbox(str(image_path), bbox, str(output_path)):
                cropped_count += 1
                print(f"✓ Cropped: {output_path.relative_to(Path('/workspace'))}")
        
        return cropped_count
    
    except Exception as e:
        print(f"Error processing JSON file {json_path}: {e}")
        return 0


def main():
    """메인 함수: 모든 JSON 파일을 처리합니다."""
    
    print("=" * 70)
    print("PDF Image Cropper - doclayout-yolo 결과 기반 crop")
    print("=" * 70)
    print(f"Input JSON directory: {DOCLAYOUT_OUTPUT_DIR}")
    print(f"Input image directory: {PDF_IMG_DIR}")
    print(f"Output crop directory: {CROP_OUTPUT_DIR}")
    print(f"Target classes: {', '.join(TARGET_CLASSES)}")
    print("=" * 70)
    
    # 모든 JSON 파일 찾기
    json_files = list(DOCLAYOUT_OUTPUT_DIR.glob("**/page_*_result.json"))
    print(f"\nFound {len(json_files)} JSON files\n")
    
    if not json_files:
        print("No JSON files found!")
        return
    
    # 각 JSON 파일 처리
    total_cropped = 0
    processed_count = 0
    
    for json_file in sorted(json_files):
        category = json_file.parent.name
        json_name = json_file.name
        
        print(f"Processing: {category}/{json_name}")
        cropped_count = process_json_file(json_file)
        total_cropped += cropped_count
        processed_count += 1
        
        if cropped_count > 0:
            print(f"  → {cropped_count} images cropped\n")
        else:
            print("  → No target detections\n")
    
    # 결과 요약
    print("=" * 70)
    print("Processing complete!")
    print(f"Total JSON files processed: {processed_count}")
    print(f"Total images cropped: {total_cropped}")
    print(f"Output saved to: {CROP_OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
