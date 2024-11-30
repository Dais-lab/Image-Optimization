# 이미지 최적화를 위한 전처리 파이프라인

이 프로젝트는 **IQI 이미지의 품질 최적화**를 목표로 하며, 객체 탐지와 유전 알고리즘(GA)을 통해 전처리 파라미터(대비와 밝기)를 최적화합니다.  
탐지된 강관 영역을 대상으로 최적화된 전처리 결과를 생성하며, 해당 파라미터는 원본 이미지에도 동일하게 적용됩니다.

---

## 주요 기능

1. **탐지 및 객체 크롭**:
   - YOLOv5를 사용하여 이미지에서 객체를 탐지하고, 탐지된 영역을 크롭합니다.

2. **유전 알고리즘(GA)을 통한 전처리 최적화**:
   - 크롭된 이미지에 대해 대비(Contrast)와 밝기(Brightness)를 조정하는 최적의 파라미터를 찾습니다.

3. **원본 이미지 전처리**:
   - 최적화된 파라미터를 원본 이미지에 적용하여 전처리 품질을 향상시킵니다.

4. **결과 저장**:
   - 크롭된 이미지, 최적화된 크롭 이미지, 전처리된 원본 이미지를 저장합니다.
   - 최적화된 파라미터는 CSV 파일로 저장됩니다.

---

## 설치 방법

### 필수 요건

- Python >= 3.8
- CUDA (GPU 가속 선택 사항)
- 필요 Python 라이브러리:
  - `numpy`
  - `torch`
  - `opencv-python`
  - `deap`
  - `pandas`
  - `tqdm`
  - `argparse`

아래 명령어로 필요한 라이브러리를 설치하세요:

```
pip install -r requirements.txt
```
사용법
최적화 파이프라인 실행
아래 명령어로 전처리 최적화 파이프라인을 실행합니다:
```
python Image_Optimization.py --source <source_folder> 
```

명령어 옵션
```
--source: 원본 이미지가 저장된 폴더 경로 (필수).
--use_gpu: GPU를 사용하여 처리 (기본값: False).
--ngen: 유전 알고리즘의 세대 수 (기본값: 50).
--pop_size: 유전 알고리즘의 인구 크기 (기본값: 20).
--brightness_range: 밝기 조정 범위 (기본값: -500 500).
--contrast_range: 대비 조정 범위 (기본값: 0.5 1.5).
--patience: 조기 종료 기준 (기본값: 10).
--delta: 개선 기준 임계값 (기본값: 1e-4).
```

실행 예제
```
python Image_Optimization.py --source ./data/images --use_gpu --ngen 100 --pop_size 30
```
출력 구조

결과는 ./result_images/ 폴더에 저장되며, 폴더 구조는 다음과 같습니다:
```
result_images/
├── yolo_results/        # YOLO 탐지 결과
│   ├── exp/
│   │   ├── labels/      # 경계 상자 좌표 파일
│   │   ├── *.jpg        # 탐지 결과가 포함된 원본 이미지
├── crops/               # 크롭된 이미지
│   ├── <image_name>_0.png
│   ├── <image_name>_1.png
├── optimized/           # 최적화된 크롭 이미지
│   ├── <image_name>_0.png
│   ├── <image_name>_1.png
├── full_processed/      # 원본 IQI 이미지 전처리 결과
│   ├── <image_name>_processed.png
├── optimization_results.csv  # 최적화 파라미터가 저장된 CSV 파일
```
CSV 파일 상세 내용

optimization_results.csv 파일에는 다음과 같은 정보가 포함됩니다:

Image Name: 원본 이미지 이름.
Alpha (Contrast): 최적화된 대비 파라미터.
Beta (Brightness): 최적화된 밝기 파라미터.


### 프로세스 개요
**강관 영역 탐지**: YOLOv5를 사용하여 이미지에서 객체를 탐지하고, 경계 상자 좌표를 생성합니다.
**크롭**: YOLO 탐지 결과를 기반으로 객체 영역을 크롭합니다.
**유전 알고리즘 최적화**: 크롭된 이미지에 대해 대비(alpha)와 밝기(beta)를 유전 알고리즘(GA)을 통해 최적화합니다.
**원본 이미지 처리**: 최적화된 파라미터를 원본 이미지에 적용하여 전처리 품질을 향상시킵니다.
