# Project Status

Last updated: 2026.05.29 KST

---

## Latest Update

### 2026.05.29 KST

#### 성능 시각화 및 TorchXRayVision Grad-CAM 완료

기존 학습 및 외부 검증 결과를 바탕으로 성능 비교 그래프와 Grad-CAM 시각화를 생성했다. 새 학습은 수행하지 않았고, 실제 checkpoint와 prediction CSV가 보존되어 있는 TorchXRayVision 기본 모델을 기준으로 Grad-CAM을 수행했다.

이번 작업의 목적은 단순 metric 확인이 아니라, 최종 보고서와 발표 자료에서 사용할 수 있는 시각 자료를 만들고, Kaggle internal validation과 RSNA external validation에서 모델이 어떤 영역을 보고 판단하는지 정성적으로 확인하는 것이다.

추가된 코드:

- `src/preprocessing.py`
- `src/visualize_results.py`
- `src/visualize_gradcam.py`
- `src/export_kaggle_val_predictions.py`
- `scripts/run_visualize_results.sh`
- `scripts/run_gradcam.sh`
- `scripts/gradcam_job.sh`
- `scripts/run_export_kaggle_val_predictions.sh`
- `.gitattributes`

#### 성능 시각화 결과

`outputs/`에 저장된 기존 metric json/csv를 사용해 성능 비교 그래프를 생성했다.

생성된 성능 그래프:

- `outputs/figures/internal_vs_external_auc.png`
- `outputs/figures/auc_drop_by_model.png`
- `outputs/figures/external_f1_recall_precision_by_model.png`
- `outputs/figures/preprocess_crop_focal_ablation.png`
- `outputs/figures/visualized_result_dirs.csv`

시각화에 사용된 실제 결과 디렉터리:

- `outputs/baseline_external`
- `outputs/resnet50_external`
- `outputs/torchxrayvision_external`
- `outputs/torchxrayvision_external_focal_a025_g2`

확인된 주요 결과:

| Model / Setting | Internal AUC | External AUC | External F1 | External Recall | External Precision | AUC Drop |
|---|---:|---:|---:|---:|---:|---:|
| Baseline CNN | 0.9613 | 0.6402 | 0.6642 | 0.9700 | 0.5049 | 0.3211 |
| ResNet50 | 0.9983 | 0.7710 | 0.7165 | 0.9690 | 0.5683 | 0.2273 |
| TorchXRayVision | 0.9968 | 0.7833 | 0.7468 | 0.9380 | 0.6204 | 0.2135 |
| TorchXRayVision + Focal Loss | 0.9979 | 0.8195 | 0.7824 | 0.9220 | 0.6794 | 0.1783 |

관찰:

- 모든 모델에서 internal AUC 대비 external AUC가 하락하여 Domain Shift가 다시 확인되었다.
- Baseline CNN은 external AUC가 0.6402로 가장 낮고 AUC drop도 가장 컸다.
- TorchXRayVision은 ResNet50보다 external AUC와 F1이 높아 의료 영상 사전학습 모델의 외부 일반화 효과가 확인되었다.
- Focal Loss 실험은 external AUC 0.8195로 높게 나타났지만, 기존 status 기준으로는 Preprocessing + Body Crop 단독 결과의 AUC 0.8222보다 낮아 최종 개선 기법이 아니라 ablation 결과로 유지한다.

#### Grad-CAM 입력 파일 생성

Grad-CAM 실행을 위해 기존 TorchXRayVision checkpoint를 사용하여 Kaggle validation prediction CSV를 추가 생성했다. 이는 새 학습이 아니라, 저장된 checkpoint를 이용한 validation inference이다.

사용한 파일:

- checkpoint: `outputs/torchxrayvision/best_torchxrayvision_seed42.pt`
- split csv: `outputs/splits/kaggle_split_seed42.csv`
- RSNA prediction: `outputs/torchxrayvision_external/rsna_predictions_seed42.csv`
- threshold: `outputs/torchxrayvision_external/internal_threshold_seed42.json`

생성된 Grad-CAM 입력 파일:

- `outputs/torchxrayvision_external/kaggle_val_predictions_seed42.csv`
- `outputs/torchxrayvision_external/threshold_policies_seed42.json`

적용 threshold:

- `youden_j`: 0.536637

해당 threshold는 Kaggle validation에서 계산된 값을 그대로 사용했으며, RSNA에서 threshold를 재튜닝하지 않았다.

#### TorchXRayVision Grad-CAM 생성 완료

실제 checkpoint와 prediction CSV가 남아 있는 TorchXRayVision 기본 모델을 기준으로 Grad-CAM을 생성했다.

Grad-CAM 기준 모델:

- Model: TorchXRayVision
- Weight: `densenet121-res224-chex`
- Checkpoint: `outputs/torchxrayvision/best_torchxrayvision_seed42.pt`
- Result dir: `outputs/torchxrayvision_external`

생성 위치:

- `outputs/figures/gradcam_torchxrayvision`

생성 결과:

- 총 파일 수: 66개
- 전체 크기: 약 17MB
- Kaggle internal validation과 RSNA external validation 각각에 대해 TP, FP, FN, TN 케이스를 샘플링
- 각 케이스별로 `original.png`, `preprocessed.png`, `gradcam_overlay.png`, `panel.png` 생성
- 각 domain별 `manifest.csv` 생성

생성 구조:

```text
outputs/figures/gradcam_torchxrayvision/
├── kaggle_internal/
│   ├── TP / FP / FN / TN cases
│   └── manifest.csv
└── rsna_external/
    ├── TP / FP / FN / TN cases
    └── manifest.csv
```

대표 Grad-CAM 패널은 GitHub 공유를 위해 `docs/figures/gradcam/`에 복사했다.

공유용 Grad-CAM 대표 이미지:

- `docs/figures/gradcam/kaggle_tp_panel.png`
- `docs/figures/gradcam/kaggle_fp_panel.png`
- `docs/figures/gradcam/kaggle_fn_panel.png`
- `docs/figures/gradcam/kaggle_tn_panel.png`
- `docs/figures/gradcam/rsna_tp_panel.png`
- `docs/figures/gradcam/rsna_fp_panel.png`
- `docs/figures/gradcam/rsna_fn_panel.png`
- `docs/figures/gradcam/rsna_tn_panel.png`

성능 그래프 대표 이미지도 GitHub 공유를 위해 `docs/figures/performance/`에 복사했다.

공유용 성능 시각화 이미지:

- `docs/figures/performance/internal_vs_external_auc.png`
- `docs/figures/performance/auc_drop_by_model.png`
- `docs/figures/performance/external_f1_recall_precision_by_model.png`
- `docs/figures/performance/preprocess_crop_focal_ablation.png`

GitHub 반영:

- commit: `docs: add performance and gradcam figures`
- 대표 성능 그래프 및 Grad-CAM 패널 12개를 `docs/figures/`에 추가
- `outputs/`, `.pt`, `.pth`, prediction CSV, metric JSON은 GitHub에 커밋하지 않음

#### 운영 메모

- Grad-CAM은 SERAPH `sbatch`를 통해 GPU node에서 실행했다.
- 실행 노드: `moana-u4`
- CUDA 사용 확인: `torch 2.5.1+cu121 cuda True`
- node-local dataset 경로:
  - Kaggle: `/local_datasets/daniel3290/chest_xray_kaggle/chest_xray`
  - RSNA: `/local_datasets/daniel3290/rsna`
- `/data`를 DataLoader가 직접 읽지 않도록 하고, `/local_datasets` 기준으로 실행했다.
- `outputs/figures/` 전체 결과는 로컬 Windows `aip_project/outputs/figures/`에도 다운로드 완료했다.
- `docs/figures/`는 팀원 공유 및 최종 보고서/발표 자료용 대표 이미지 보관 위치로 사용한다.

#### 해석 방향

이번 Grad-CAM 결과는 최종 정량 성능이 가장 높았던 Preprocessing + Body Crop 모델이 아니라, 실제 checkpoint와 prediction CSV가 보존되어 있는 TorchXRayVision 기본 모델 기준으로 생성했다.

따라서 최종 보고서에서는 다음과 같이 정리한다.

- 정량 성능 비교에서는 Preprocessing + Body Crop 결과를 포함한다.
- 정성적 Grad-CAM 분석은 실제 산출물이 보존된 TorchXRayVision 기본 모델을 기준으로 수행한다.
- Grad-CAM은 성능 개선 자체보다 Kaggle internal과 RSNA external에서 모델 주목 영역이 어떻게 달라지는지 확인하는 Domain Shift 해석 자료로 사용한다.

#### 현재 완료된 작업

- 성능 비교 그래프 생성 완료
- TorchXRayVision Grad-CAM 입력 파일 생성 완료
- Kaggle internal / RSNA external TP, FP, FN, TN Grad-CAM 생성 완료
- 대표 성능 그래프 및 Grad-CAM 패널 GitHub 업로드 완료
- 전체 시각화 결과 로컬 다운로드 완료


---

## Previous Updates

### 2026.05.23 KST

#### 전처리 통일 + Body Crop 적용 및 Focal Loss Ablation 완료

TorchXRayVision 모델의 RSNA external validation 성능 개선을 위해 입력 전처리 단계를 추가로 수정했다. 기존 모델 구조와 평가 방식은 유지하되, Kaggle JPEG 이미지와 RSNA DICOM 이미지 간 입력 분포 차이를 줄이기 위해 동일한 preprocessing pipeline을 적용했다.

추가 적용한 기법:

- percentile 기반 intensity normalization
- non-black/background 영역 제거를 위한 body crop
- Kaggle/RSNA 입력 처리 방식 통일
- TorchXRayVision 학습 코드에 loss function 선택 옵션 추가
- focal loss 실험을 ablation study로 추가

평가 설정:

- seed: 42
- Main policy: **`youden_j`**
- Threshold: Kaggle validation에서 계산한 threshold를 RSNA external validation에 그대로 적용
- External: RSNA stage_2_train_images에서 클래스별 1,000 샘플 (n=2000)
- RSNA에서는 threshold 재튜닝하지 않음

#### Preprocessing + Body Crop 결과

| Setting | Accuracy | Precision | Recall | F1 | AUC | Threshold | Predicted Positive Rate | Mean Probability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TorchXRayVision + Preprocessing + Body Crop | 0.7500 | 0.6905 | 0.9060 | 0.7837 | 0.8222 | 0.6867 | 0.6560 | 0.6727 |

이전 TorchXRayVision external AUC 0.7833 대비, preprocessing + body crop 적용 후 external AUC가 0.8222로 상승했다. 또한 F1-score도 0.7468에서 0.7837로 상승했다. 따라서 현재 실험 기준으로는 모델 구조를 추가로 변경하는 것보다, Kaggle과 RSNA 사이의 입력 분포 차이를 줄이는 전처리 통일과 배경 제거가 외부 검증 성능 개선에 더 직접적으로 기여한 것으로 정리할 수 있다.

#### Focal Loss Ablation

전처리 + body crop을 적용한 상태에서 loss function만 focal loss로 변경하여 ablation study를 수행했다. Focal loss는 hard example에 더 집중하도록 하는 목적이 있으므로, RSNA external validation에서 혼동되는 샘플에 대한 성능 개선 가능성을 확인하기 위해 추가했다.

| Setting | Accuracy | Precision | Recall | F1 | AUC | Threshold | Predicted Positive Rate | Mean Probability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Preprocessing + Body Crop | 0.7500 | 0.6905 | 0.9060 | 0.7837 | 0.8222 | 0.6867 | 0.6560 | 0.6727 |
| Preprocessing + Body Crop + Focal Loss | 0.7435 | 0.6794 | 0.9220 | 0.7824 | 0.8195 | 0.5867 | 0.6785 | 0.6333 |

Focal loss 적용 결과 recall은 0.9060에서 0.9220으로 증가했지만, precision은 0.6905에서 0.6794로 감소했다. 그 결과 F1-score는 0.7837에서 0.7824로 소폭 감소했고, AUC도 0.8222에서 0.8195로 소폭 감소했다. 따라서 focal loss는 폐렴 예측을 더 많이 하도록 만드는 방향의 변화는 있었지만, false positive 증가로 인해 최종 성능 개선에는 기여하지 못했다. 본 결과는 최종 성능 개선 기법이 아니라 loss function ablation study로 포함한다.

#### 관찰

- 전처리 통일 + body crop 적용 후 TorchXRayVision의 RSNA external AUC가 0.7833에서 0.8222로 상승했다.
- F1-score도 0.7468에서 0.7837로 상승하여, 외부 검증 기준 현재까지 가장 좋은 결과를 기록했다.
- Focal loss는 recall을 높였지만 precision과 AUC가 감소하여 최종 모델 개선으로 채택하지 않는다.
- 현재 결과 기준으로는 threshold policy 변경이나 loss 변경보다 preprocessing/domain harmonization이 더 효과적인 개선 방향으로 보인다.

산출물:

- `outputs/torchxrayvision/outputs` 또는 `outputs/torchxrayvision_preprocess_crop` 계열 checkpoint
- `external_outputs/rsna_external_metrics_seed42.json`
- `external_outputs/rsna_predictions_seed42.csv`
- focal loss 실험 결과는 ablation 결과로 별도 보관

운영 메모:

- `torchxrayvision` package를 프로젝트 내부 `python_packages/`에 설치하여 sbatch 실행 환경에서 import 가능하도록 처리.
- `pydicom`도 RSNA DICOM 로딩을 위해 `python_packages/`에 추가 설치.
- TorchXRayVision DenseNet의 18-output `op_norm` 충돌을 피하기 위해 backbone feature extractor를 직접 사용하고 binary classifier head를 적용.

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
- `src/xray_preprocess.py`
- `src/models/baseline_cnn.py`
- `src/train_baseline.py`

서버 산출물:

- `outputs/baseline/best_baseline_seed42.pt`
- `outputs/baseline/metrics_seed42.json`

---

---

## Current Code Files

- `src/prepare_kaggle_split.py`
- `src/dataset.py`
- `src/xray_preprocess.py`
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
- `src/preprocessing.py`
- `src/visualize_results.py`
- `src/visualize_gradcam.py`
- `src/export_kaggle_val_predictions.py`
- `scripts/run_visualize_results.sh`
- `scripts/run_gradcam.sh`
- `scripts/gradcam_job.sh`
- `scripts/run_export_kaggle_val_predictions.sh`


---

## Current Result Summary

`youden_j` threshold policy 기준. Internal = Kaggle val (n=1051), External = RSNA (n=2000).  
Baseline/ResNet50/TorchXRayVision 기본 실험은 Bootstrap 95% CI (n=1000)를 포함한다. Preprocessing + Body Crop 및 Focal Loss ablation 결과는 현재 단일 seed 평가 결과 기준이다.

| Model / Setting | Internal AUC [95% CI] | External AUC | External F1 | External Recall | External Precision |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 0.9613 [0.948, 0.972] | 0.6402 [0.616, 0.663] | 0.6642 | 0.9700 | 0.5049 |
| ResNet50 | 0.9983 [0.997, 0.999] | 0.7709 [0.750, 0.790] | 0.7154 | 0.9690 | 0.5670 |
| TorchXRayVision | 0.9968 [0.995, 0.998] | 0.7833 [0.763, 0.803] | 0.7468 | 0.9380 | 0.6204 |
| TorchXRayVision + Preprocessing + Body Crop | - | 0.8222 | 0.7837 | 0.9060 | 0.6905 |
| TorchXRayVision + Preprocessing + Body Crop + Focal Loss (Ablation) | - | 0.8195 | 0.7824 | 0.9220 | 0.6794 |

---

---

## Notes

- `outputs/`, `.pt`, `.pth` 파일은 GitHub에 올리지 않음
- RSNA는 학습에 사용하지 않고 external validation에만 사용
- RSNA threshold는 새로 튜닝하지 않고 Kaggle validation에서 정한 threshold를 그대로 적용
- TorchXRayVision은 `densenet121-res224-chex` weight를 사용했으며, RSNA/all weight는 사용하지 않음
- 현재 결과 기준, TorchXRayVision + Preprocessing + Body Crop이 RSNA external validation에서 가장 높은 성능을 보였지만 Domain Shift는 여전히 확인됨
- Focal loss는 recall을 높였지만 AUC/F1 개선으로 이어지지 않아 ablation study 결과로만 포함

---
- `docs/figures/`에는 발표 및 보고서 공유용 대표 성능 그래프와 Grad-CAM 패널만 저장한다.
- 전체 Grad-CAM 결과와 prediction CSV는 `outputs/`에만 보관하고 GitHub에는 커밋하지 않는다.
- Grad-CAM 정성 분석은 실제 checkpoint와 prediction CSV가 보존된 TorchXRayVision 기본 모델 기준으로 수행했다.
- Preprocessing + Body Crop 모델은 정량 성능 비교에는 포함하지만, 해당 산출물의 checkpoint/prediction 파일이 현재 보존되어 있지 않아 Grad-CAM 기준 모델로는 사용하지 않았다.

---

## Next Tasks

- Grad-CAM 대표 이미지 육안 검토 및 정성 분석
- Kaggle internal과 RSNA external의 주목 영역 차이 해석
- FP/FN 실패 케이스 중심 Domain Shift 분석
- 최종 보고서 및 발표 자료에 성능 그래프와 Grad-CAM 패널 반영
- 선택 사항: Preprocessing + Body Crop 결과에 대한 Bootstrap 95% CI 계산
- 선택 사항: preprocessing/crop 개선 효과에 대한 paired bootstrap 검정

