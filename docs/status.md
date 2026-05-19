# Project Status

기준 시점: 2026.05.19 11:35 KST

## 완료된 작업

### 1. 데이터셋 준비 완료

SERAPH에서 Kaggle/RSNA 데이터셋 다운로드 및 압축 해제 완료.

원본 zip 위치:
- /data/daniel3290/datasets/tarfiles

실제 학습/검증용 데이터 위치:
- /local_datasets/daniel3290/chest_xray_kaggle/chest_xray
- /local_datasets/daniel3290/rsna

확인된 데이터 개수:
- Kaggle train NORMAL: 1341
- Kaggle train PNEUMONIA: 3875
- Kaggle test NORMAL: 234
- Kaggle test PNEUMONIA: 390
- RSNA train images: 26684

### 2. Kaggle train/val split 완료

src/prepare_kaggle_split.py 작성 및 SERAPH 실제 데이터로 검증 완료.

생성된 CSV:
- outputs/splits/kaggle_split_seed42.csv

split 결과:
- train NORMAL: 1073
- train PNEUMONIA: 3092
- val NORMAL: 268
- val PNEUMONIA: 783
- total: 5216

### 3. Baseline CNN 학습 완료

추가된 코드:
- src/dataset.py
- src/models/baseline_cnn.py
- src/train_baseline.py

SERAPH moana-y2에서 Baseline CNN 5 epoch 학습 완료.

결과:
- Best val_auc: 0.9613
- val_accuracy: 0.9115
- val_precision: 0.9612
- val_recall: 0.9183
- val_f1: 0.9393

저장된 산출물:
- outputs/baseline/best_baseline_seed42.pt
- outputs/baseline/metrics_seed42.json

주의:
- outputs/, .pt, .pth 파일은 GitHub에 올리지 않음.

## 현재 상태 요약

Kaggle 데이터 기준으로 train/validation split, Dataset/DataLoader, Baseline CNN 학습 파이프라인까지 정상 작동 확인 완료.

현재 성능은 Kaggle 내부 validation 기준이며, 최종 목표는 RSNA 외부 검증을 통해 Domain Shift를 분석하는 것임.

## 다음 작업

1. ResNet50 전이학습 코드 작성
2. 내부 validation에서 threshold 계산
3. 동일 threshold로 RSNA 외부 검증
4. Bootstrap 95% CI 계산
5. Grad-CAM 시각화
