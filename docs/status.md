# Project Status

Last updated: 2026.05.19 12:31 KST

---

## Latest Update

### 2026.05.19 12:31 KST

#### Baseline CNN RSNA external validation 완료

Baseline CNN checkpoint를 사용해 RSNA external validation을 실행했다.

평가 방식:

- Kaggle validation에서 Youden's J statistic으로 threshold 계산
- 계산된 threshold를 RSNA에 그대로 적용
- RSNA에서는 threshold 재튜닝하지 않음
- RSNA sample: NORMAL 1000장 + PNEUMONIA 1000장 = 총 2000장

적용 threshold:

- threshold: 0.5721

RSNA external validation 결과:

- Accuracy: 0.5095
- Precision: 0.5049
- Recall: 0.9700
- F1-score: 0.6642
- AUC: 0.6402
- n_samples: 2000

해석:

- Kaggle internal validation AUC: 0.9613
- RSNA external validation AUC: 0.6402
- Baseline CNN 기준으로 Domain Shift 확인
- RSNA에서는 Recall은 높지만 Precision이 낮아, 정상 이미지도 폐렴으로 많이 예측하는 경향이 있음

추가된 코드:

- `src/rsna_dataset.py`
- `src/evaluate_baseline_external.py`

서버 산출물:

- `outputs/baseline_external/internal_threshold_seed42.json`
- `outputs/baseline_external/internal_metrics_seed42.json`
- `outputs/baseline_external/rsna_external_metrics_seed42.json`
- `outputs/baseline_external/rsna_predictions_seed42.csv`

---

## Previous Updates

### 2026.05.19 11:35 KST

#### 데이터셋 준비 및 Baseline CNN 학습 완료

완료한 작업:

- SERAPH에서 Kaggle/RSNA 데이터셋 다운로드 완료
- moana-y2 컴퓨트 노드에서 `/local_datasets/daniel3290`에 압축 해제 완료
- Kaggle train 데이터 기준 train/validation split 생성 완료
- Baseline CNN 학습 코드 작성 및 실행 완료

데이터 확인:

- Kaggle train NORMAL: 1341
- Kaggle train PNEUMONIA: 3875
- Kaggle test NORMAL: 234
- Kaggle test PNEUMONIA: 390
- RSNA train images: 26684

Kaggle split 결과:

- 생성 파일: `outputs/splits/kaggle_split_seed42.csv`
- train NORMAL: 1073
- train PNEUMONIA: 3092
- val NORMAL: 268
- val PNEUMONIA: 783
- total: 5216

Baseline CNN internal validation 결과:

- Best val AUC: 0.9613
- Accuracy: 0.9115
- Precision: 0.9612
- Recall: 0.9183
- F1-score: 0.9393

추가된 코드:

- `src/prepare_kaggle_split.py`
- `src/dataset.py`
- `src/models/baseline_cnn.py`
- `src/train_baseline.py`

서버 산출물:

- `outputs/baseline/best_baseline_seed42.pt`
- `outputs/baseline/metrics_seed42.json`

---

## Current Code Files

- `src/prepare_kaggle_split.py`
- `src/dataset.py`
- `src/models/baseline_cnn.py`
- `src/train_baseline.py`
- `src/rsna_dataset.py`
- `src/evaluate_baseline_external.py`

---

## Notes

- `outputs/`, `.pt`, `.pth` 파일은 GitHub에 올리지 않음
- RSNA는 학습에 사용하지 않고 external validation에만 사용
- RSNA threshold는 새로 튜닝하지 않고 Kaggle validation에서 정한 threshold를 그대로 적용

---

## Next Tasks

- ResNet50 전이학습 코드 작성
- ResNet50 internal validation 성능 확인
- ResNet50 기준 RSNA external validation 수행
- Bootstrap 95% CI 계산
- Grad-CAM 시각화
