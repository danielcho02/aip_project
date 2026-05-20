# Project Status

Last updated: 2026.05.20 15:00 KST

---

## Latest Update

### 2026.05.20 15:00 KST

#### TorchXRayVision 학습 및 RSNA external validation 완료

Proposal에 포함된 후보 모델 B인 TorchXRayVision 사전학습 모델 실험을 완료했다.

사용한 weight:

- `densenet121-res224-chex`

주의:

- RSNA weight는 사용하지 않음
- `all` weight도 사용하지 않음
- RSNA는 학습에 사용하지 않고 external validation에만 사용

추가된 코드:

- `src/models/torchxrayvision_model.py`
- `src/train_torchxrayvision.py`
- `src/evaluate_torchxrayvision_external.py`

TorchXRayVision 학습 결과:

- epochs: 10
- batch size: 32
- lr: 1e-4
- seed: 42
- Best val AUC: 0.9968

Kaggle internal validation 결과:

- Accuracy: 0.9762
- Precision: 0.9871
- Recall: 0.9808
- F1-score: 0.9840
- AUC: 0.9968
- threshold: 0.5366
- n_samples: 1051

RSNA external validation 방식:

- Kaggle validation에서 Youden's J statistic으로 threshold 계산
- 계산된 threshold를 RSNA에 그대로 적용
- RSNA에서는 threshold 재튜닝하지 않음
- RSNA sample: NORMAL 1000장 + PNEUMONIA 1000장 = 총 2000장

RSNA external validation 결과:

- Accuracy: 0.6820
- Precision: 0.6204
- Recall: 0.9380
- F1-score: 0.7468
- AUC: 0.7833
- n_samples: 2000

해석:

- TorchXRayVision이 현재까지 RSNA external validation에서 가장 높은 AUC, Accuracy, Precision, F1-score를 기록함
- 의료영상 사전학습 모델이 ResNet50보다 외부 데이터에서 조금 더 안정적인 성능을 보임
- 다만 Kaggle internal AUC 0.9968 대비 RSNA external AUC 0.7833으로 하락하여 Domain Shift는 여전히 확인됨

서버 산출물:

- `outputs/torchxrayvision/best_torchxrayvision_seed42.pt`
- `outputs/torchxrayvision/metrics_seed42.json`
- `outputs/torchxrayvision_external/internal_threshold_seed42.json`
- `outputs/torchxrayvision_external/internal_metrics_seed42.json`
- `outputs/torchxrayvision_external/rsna_external_metrics_seed42.json`
- `outputs/torchxrayvision_external/rsna_predictions_seed42.csv`

---

## Previous Updates

### 2026.05.20 KST

#### ResNet50 학습 및 RSNA external validation 완료

ResNet50 전이학습 코드를 작성하고, SERAPH `moana-y2`에서 Kaggle train/validation split 기준으로 학습을 완료했다.

추가된 코드:

- `src/models/resnet50.py`
- `src/train_resnet50.py`
- `src/evaluate_resnet50_external.py`

ResNet50 학습 결과:

- epochs: 10
- batch size: 32
- lr: 1e-4
- seed: 42
- Best val AUC: 0.9983

Kaggle internal validation 결과:

- Accuracy: 0.9810
- Precision: 0.9923
- Recall: 0.9821
- F1-score: 0.9872
- AUC: 0.9983

RSNA external validation 방식:

- Kaggle validation에서 Youden's J statistic으로 threshold 계산
- 계산된 threshold를 RSNA에 그대로 적용
- RSNA에서는 threshold 재튜닝하지 않음
- RSNA sample: NORMAL 1000장 + PNEUMONIA 1000장 = 총 2000장

적용 threshold:

- threshold: 0.2040

RSNA external validation 결과:

- Accuracy: 0.6165
- Precision: 0.5683
- Recall: 0.9690
- F1-score: 0.7165
- AUC: 0.7710
- n_samples: 2000

해석:

- Baseline CNN external AUC: 0.6402
- ResNet50 external AUC: 0.7710
- ResNet50이 Baseline CNN보다 RSNA 외부 성능은 개선됨
- 그러나 Kaggle internal AUC 0.9983 대비 RSNA external AUC 0.7710으로 성능이 크게 하락하여 Domain Shift가 여전히 확인됨
- RSNA에서는 Recall은 높지만 Precision이 낮아, 정상 이미지도 폐렴으로 예측하는 경향이 있음

서버 산출물:

- `outputs/resnet50/best_resnet50_seed42.pt`
- `outputs/resnet50/metrics_seed42.json`
- `outputs/resnet50_external/internal_threshold_seed42.json`
- `outputs/resnet50_external/internal_metrics_seed42.json`
- `outputs/resnet50_external/rsna_external_metrics_seed42.json`
- `outputs/resnet50_external/rsna_predictions_seed42.csv`

---

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
- `src/models/resnet50.py`
- `src/train_resnet50.py`
- `src/evaluate_resnet50_external.py`
- `src/models/torchxrayvision_model.py`
- `src/train_torchxrayvision.py`
- `src/evaluate_torchxrayvision_external.py`

---

## Current Result Summary

| Model | Internal AUC | External RSNA AUC | External F1 | External Recall | External Precision |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 0.9613 | 0.6402 | 0.6642 | 0.9700 | 0.5049 |
| ResNet50 | 0.9983 | 0.7710 | 0.7165 | 0.9690 | 0.5683 |
| TorchXRayVision | 0.9968 | 0.7833 | 0.7468 | 0.9380 | 0.6204 |

---

## Notes

- `outputs/`, `.pt`, `.pth` 파일은 GitHub에 올리지 않음
- RSNA는 학습에 사용하지 않고 external validation에만 사용
- RSNA threshold는 새로 튜닝하지 않고 Kaggle validation에서 정한 threshold를 그대로 적용
- TorchXRayVision은 `densenet121-res224-chex` weight를 사용했으며, RSNA/all weight는 사용하지 않음
- 현재 결과 기준, TorchXRayVision이 RSNA external validation에서 가장 높은 성능을 보였지만 Domain Shift는 여전히 확인됨

---

## Next Tasks

- Bootstrap 95% CI 계산
- Grad-CAM 시각화
- 모델별 결과표 정리
- Internal vs External 성능 차이 분석
