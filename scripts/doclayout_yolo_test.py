#!/usr/bin/env python3
"""
Process images in pdf_img folder using DocLayout-YOLO
and save results to output/doclayout-yolo
"""

import json
from pathlib import Path
from doclayout_yolo import YOLOv10
import cv2
from huggingface_hub import hf_hub_download


def setup_output_dir(output_base: str = "output/doclayout-yolo") -> Path:
    """Create output directory structure"""
    output_dir = Path(output_base)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_image_files(pdf_img_dir: str = "pdf_img") -> dict:
    """
    Get all images from pdf_img subdirectories
    Returns: {category_name: [image_paths]}
    """
    image_dict = {}
    pdf_img_path = Path(pdf_img_dir)
    
    if not pdf_img_path.exists():
        print(f"❌ {pdf_img_dir} 폴더가 없습니다.")
        return image_dict
    
    # 각 카테고리 폴더 순회
    for category_dir in sorted(pdf_img_path.iterdir()):
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        image_paths = []
        
        # 이미지 파일 수집 (jpg, png, jpeg)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            image_paths.extend(category_dir.glob(ext))
        
        if image_paths:
            image_dict[category_name] = sorted(image_paths)
            print(f"📁 {category_name}: {len(image_paths)}개 이미지")
    
    return image_dict


def process_images(
    image_dict: dict,
    model_path: str = "doclayout_yolo_docstructbench_imgsz1024.pt",
    output_base: str = "output/doclayout-yolo",
    imgsz: int = 1024,
    conf: float = 0.2,
    device: str = "cuda:0"
):
    """
    Process all images with DocLayout-YOLO
    """
    output_dir = Path(output_base)
    
    # 모델 로드
    print("\n🔄 모델 로드 중...")
    try:
        # 먼저 로컬 경로 확인
        model_file = Path(model_path)
        if not model_file.exists():
            print("   로컬 모델 없음, Hugging Face에서 다운로드 중...")
            # Hugging Face에서 모델 다운로드
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                filename="doclayout_yolo_docstructbench_imgsz1024.pt",
                cache_dir="./models"
            )
            print(f"   ✅ 다운로드 완료: {model_path}")
        
        model = YOLOv10(model_path)
        print("   ✅ 모델 로드 성공!")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    total_images = sum(len(paths) for paths in image_dict.values())
    processed = 0
    
    # 각 카테고리별로 처리
    for category, image_paths in image_dict.items():
        category_output = output_dir / category
        category_output.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📂 처리 중: {category} ({len(image_paths)}개)")
        
        for idx, image_path in enumerate(image_paths, 1):
            try:
                image_path = Path(image_path)
                print(f"  [{idx}/{len(image_paths)}] {image_path.name}...", end=" ", flush=True)
                
                # 예측 수행
                det_res = model.predict(
                    str(image_path),
                    imgsz=imgsz,
                    conf=conf,
                    device=device
                )
                
                # 결과 저장
                # 1. 시각화 이미지 저장
                if len(det_res) > 0:
                    annotated_frame = det_res[0].plot(pil=True, line_width=2, font_size=12)
                    result_image_path = category_output / f"{image_path.stem}_result.jpg"
                    cv2.imwrite(str(result_image_path), annotated_frame)
                    
                    # 2. 감지 결과 JSON 저장
                    result_json_path = category_output / f"{image_path.stem}_result.json"
                    result_data = {
                        "image": str(image_path),
                        "detections": []
                    }
                    
                    # 각 감지된 요소
                    for obj in det_res[0].boxes:
                        detection = {
                            "class": int(obj.cls[0]),
                            "class_name": model.names.get(int(obj.cls[0]), "unknown"),
                            "confidence": float(obj.conf[0]),
                            "bbox": [float(x) for x in obj.xyxy[0].tolist()]
                        }
                        result_data["detections"].append(detection)
                    
                    with open(result_json_path, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"✅ ({len(result_data['detections'])} 요소 감지)")
                else:
                    print("⚠️  감지 실패")
                
                processed += 1
                
            except Exception as e:
                print(f"❌ 에러: {e}")
    
    # 결과 요약
    print("\n" + "="*60)
    print("✅ 처리 완료!")
    print(f"   총 이미지: {total_images}개")
    print(f"   성공: {processed}개")
    print(f"   저장 위치: {output_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DocLayout-YOLO로 pdf_img 이미지 처리")
    parser.add_argument("--pdf-img", default="pdf_img", help="pdf_img 폴더 경로")
    parser.add_argument("--output", default="output/doclayout-yolo", help="출력 폴더 경로")
    parser.add_argument("--model", default="doclayout_yolo_docstructbench_imgsz1024.pt", help="모델 경로 또는 이름")
    parser.add_argument("--imgsz", type=int, default=1024, help="예측 이미지 크기")
    parser.add_argument("--conf", type=float, default=0.2, help="신뢰도 임계값")
    parser.add_argument("--device", default="cuda:0", help="사용 디바이스 (cuda:0 또는 cpu)")
    
    args = parser.parse_args()
    
    # 이미지 수집
    image_dict = get_image_files(args.pdf_img)
    
    if not image_dict:
        print("❌ 처리할 이미지가 없습니다.")
        exit(1)
    
    # 이미지 처리
    process_images(
        image_dict,
        model_path=args.model,
        output_base=args.output,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device
    )
