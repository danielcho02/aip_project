# Project Status

Last updated: 2026.05.21 17:08 KST

---

## Latest Update

### 2026.05.21 17:08 KST

#### Multi-threshold 평가 + Bootstrap 95% CI 완료

`src/evaluate_all_models.py` (브랜치 `woohyuk/eval-multi-threshold`) 를 사용해서 baseline CNN, ResNet50, TorchXRayVision 세 모델을 동일 파이프라인에서 평가했다. Kaggle internal validation 에서 5개 threshold policy 를 도출한 뒤, RSNA external validation 에 동일 threshold 를 그대로 적용했다.

평가 설정:

- seed: 42
- bootstrap resamples: 1000
- Main policy: **`youden_j`** (Kaggle validation 에서 Youden's J statistic 최대화)
- 사용한 5개 policy: `default_0.5`, `youden_j`, `f1_max`, `recall_target_0.9`, `recall_target_0.95`
- Internal: Kaggle validation split (n=1051)
- External: RSNA stage_2_train_images 에서 클래스별 1,000 샘플 (n=2000)

#### Main 결과 — youden_j policy, Bootstrap 95% CI

Threshold 는 Kaggle validation 에서 결정한 값을 RSNA 에 그대로 적용했다.

| Model | Threshold | Internal AUC [95% CI] | External AUC [95% CI] | AUC Drop |
|---|---:|---:|---:|---:|
| Baseline CNN | 0.5721 | 0.9613 [0.9479, 0.9719] | 0.6402 [0.6159, 0.6635] | -0.3211 |
| ResNet50 | 0.2016 | 0.9983 [0.9971, 0.9992] | 0.7709 [0.7501, 0.7900] | -0.2274 |
| TorchXRayVision | 0.5366 | 0.9968 [0.9947, 0.9984] | 0.7833 [0.7634, 0.8034] | -0.2135 |

F1 / Recall / Precision 비교 (외부 검증, 95% CI 포함):

| Model | External F1 [95% CI] | External Recall [95% CI] | External Precision [95% CI] |
|---|---:|---:|---:|
| Baseline CNN | 0.6642 [0.6443, 0.6834] | 0.9700 [0.9589, 0.9802] | 0.5049 [0.4828, 0.5279] |
| ResNet50 | 0.7154 [0.6949, 0.7330] | 0.9690 [0.9589, 0.9790] | 0.5670 [0.5424, 0.5897] |
| TorchXRayVision | 0.7468 [0.7286, 0.7651] | 0.9380 [0.9219, 0.9521] | 0.6204 [0.5972, 0.6449] |

#### 5개 Threshold Policy 비교

각 셀: External AUC / External F1 / External Recall (모든 policy 의 External AUC 는 threshold 와 무관하므로 동일).

| Model | Policy | Ext F1 | Ext Recall | Ext Precision | Ext Pos Rate |
|---|---|---:|---:|---:|---:|
| Baseline CNN | default_0.5 | 0.6642 | 0.9750 | 0.5036 | 0.968 |
| Baseline CNN | **youden_j** | 0.6642 | 0.9700 | 0.5049 | 0.961 |
| Baseline CNN | f1_max | 0.6660 | 0.9870 | 0.5025 | 0.982 |
| Baseline CNN | recall_target_0.9 | 0.6642 | 0.9700 | 0.5049 | 0.961 |
| Baseline CNN | recall_target_0.95 | 0.6660 | 0.9870 | 0.5025 | 0.982 |
| ResNet50 | default_0.5 | 0.7411 | 0.9420 | 0.6109 | 0.771 |
| ResNet50 | **youden_j** | 0.7154 | 0.9690 | 0.5670 | 0.855 |
| ResNet50 | f1_max | 0.6981 | 0.9780 | 0.5427 | 0.901 |
| ResNet50 | recall_target_0.9 | 0.7373 | 0.8450 | 0.6540 | 0.646 |
| ResNet50 | recall_target_0.95 | 0.7440 | 0.9240 | 0.6226 | 0.742 |
| TorchXRayVision | default_0.5 | 0.7453 | 0.9410 | 0.6170 | 0.763 |
| TorchXRayVision | **youden_j** | 0.7468 | 0.9380 | 0.6204 | 0.756 |
| TorchXRayVision | f1_max | 0.7347 | 0.9610 | 0.5947 | 0.808 |
| TorchXRayVision | recall_target_0.9 | 0.7443 | 0.8210 | 0.6808 | 0.603 |
| TorchXRayVision | recall_target_0.95 | 0.7558 | 0.8900 | 0.6568 | 0.678 |

#### 관찰

- 모든 모델에서 internal → external AUC drop 이 0.21–0.32 로 일관되게 큼. 도메인 시프트 확인.
- Internal 에서는 ResNet50 (0.998) > TorchXRayVision (0.997) > Baseline (0.961) 순서이지만, external 에서는 **TorchXRayVision (0.783) > ResNet50 (0.771) > Baseline (0.640)** 으로 사전학습 가중치 (chex) 의 일반화 효과가 가장 크게 드러남.
- Threshold policy 별 외부 성능 차이는 모델마다 다른 양상:
  - Baseline 은 youden_j 와 recall_target_0.9 가 동일 threshold 로 수렴 → policy 다양성이 약함.
  - ResNet50 은 youden_j threshold (0.2016) 가 매우 낮게 잡혀서 external recall 은 0.969 로 보존되지만 precision 이 0.567 까지 떨어짐 (overpredict).
  - TorchXRayVision 은 policy 간 F1 편차가 가장 작고 (0.735–0.756) precision/recall trade-off 가 가장 안정적.

산출물:

- `outputs/baseline_cnn_multi/{domain_shift_summary,internal_report,rsna_external_report,threshold_policies}_seed42.json`
- `outputs/resnet50_multi/...`
- `outputs/torchxrayvision_multi/...`
- 각 디렉토리에 prediction CSV (`kaggle_val_predictions_seed42.csv`, `rsna_predictions_seed42.csv`) 포함

운영 메모:

- 평가 잡은 `scripts/run_eval.sh <model> <ckpt>` → `scripts/eval_job.sh` (sbatch) 패턴으로 통일.
- partition: `batch_ce_ugrad` (학부 동시 GPU 한도 1 → 3 모델 순차 실행)
- dataloader 경로는 NAS 직접 사용 금지 규정에 따라 잡 시작 시 `/data/woohyuk/aip_data` → `/local_datasets/woohyuk/aip_data` 로 rsync 후 진행 (같은 노드 재사용 시 캐시 hit 으로 1초 미만).
- `setup_env.sh` 에 `pydicom` 누락이 발견되어 추가 패치.

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

`youden_j` threshold policy 기준, Bootstrap 95% CI (n=1000). Internal = Kaggle val (n=1051), External = RSNA (n=2000).

| Model | Internal AUC [95% CI] | External AUC [95% CI] | External F1 | External Recall | External Precision |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 0.9613 [0.948, 0.972] | 0.6402 [0.616, 0.663] | 0.6642 | 0.9700 | 0.5049 |
| ResNet50 | 0.9983 [0.997, 0.999] | 0.7709 [0.750, 0.790] | 0.7154 | 0.9690 | 0.5670 |
| TorchXRayVision | 0.9968 [0.995, 0.998] | 0.7833 [0.763, 0.803] | 0.7468 | 0.9380 | 0.6204 |

---

## Notes

- `outputs/`, `.pt`, `.pth` 파일은 GitHub에 올리지 않음
- RSNA는 학습에 사용하지 않고 external validation에만 사용
- RSNA threshold는 새로 튜닝하지 않고 Kaggle validation에서 정한 threshold를 그대로 적용
- TorchXRayVision은 `densenet121-res224-chex` weight를 사용했으며, RSNA/all weight는 사용하지 않음
- 현재 결과 기준, TorchXRayVision이 RSNA external validation에서 가장 높은 성능을 보였지만 Domain Shift는 여전히 확인됨

---

## Next Tasks

- Grad-CAM 시각화
- Internal vs External 성능 차이 정성적 분석 (실패 케이스 샘플링)
- 5개 threshold policy 비교에 대한 통계적 유의성 검정 (paired bootstrap)
