# Pneumonia Classification AI Model

> 흉부 X-ray 영상 기반 폐렴 분류 AI 모델 개발 및 외부 데이터셋 검증을 통한 Domain Shift 분석 프로젝트

---

## 1. 프로젝트 요약

본 프로젝트는 흉부 X-ray 영상으로부터 폐렴 여부를 자동 분류하는 딥러닝 모델을 개발하고, 외부 데이터셋 검증을 통해 모델의 일반화 성능과 Domain Shift 문제를 분석하는 것을 목표로 한다.

기존 의료 AI 연구는 단일 데이터셋 내부 성능만 보고하는 경우가 많다. 그러나 실제 임상 환경에서는 병원, 장비, 환자군, 촬영 조건이 달라지면서 모델 성능이 크게 저하될 수 있다. 본 프로젝트는 Kaggle Chest X-Ray Images 데이터셋을 학습용으로 사용하고, RSNA Pneumonia Detection Challenge 데이터셋을 외부 검증용으로 활용하여 내부 성능과 외부 성능의 차이를 정량적으로 비교한다.

### 핵심 키워드

`PyTorch` `Transfer Learning` `ResNet50` `timm` `External Validation` `Domain Shift` `Bootstrap CI` `Grad-CAM`

### 팀 정보

| 항목 | 내용 |
|---|---|
| 팀명 | 우리팀 |
| 팀원 | 조민규, 강우혁, 정희승 |
| 프로젝트 주제 | 흉부 X-ray 기반 폐렴 분류 AI 모델 개발 |
| 주요 목표 | 폐렴 분류 모델 개발, 외부 검증, Domain Shift 분석 |

---

## 2. 목적 및 필요성

폐렴은 주요 호흡기 질환 중 하나이며, 흉부 X-ray 판독은 진단 과정에서 중요한 역할을 한다. 하지만 실제 의료 현장에서는 다음과 같은 문제가 발생한다.

첫째, 전문 판독 인력 부족 문제가 있다. 중소형 의료기관이나 농어촌 지역에서는 방사선 전문의를 충분히 확보하기 어렵다.

둘째, 판독 지연 문제가 있다. 응급 상황에서는 빠른 판독이 필요하지만, 의료 인력 부족이나 업무량 증가로 인해 치료 적기를 놓칠 수 있다.

셋째, 판독자 간 편차가 존재한다. 동일한 X-ray 영상이라도 판독자의 경험과 판단 기준에 따라 결과가 달라질 수 있다.

넷째, 의료 접근성 불평등 문제가 있다. 의료 인프라가 취약한 지역에서는 전문 판독 서비스를 안정적으로 제공받기 어렵다.

본 프로젝트는 이러한 문제를 해결하기 위해 흉부 X-ray 기반 폐렴 자동 분류 모델을 개발한다. 특히 단순히 내부 데이터셋에서 높은 성능을 얻는 것에 그치지 않고, 외부 데이터셋에서 모델 성능이 어떻게 변화하는지 통계적으로 분석한다.

---

## 3. 문제 인식

기존 폐렴 분류 모델 연구의 주요 한계는 단일 데이터셋 내부 성능 보고에 집중되어 있다는 점이다. 내부 validation 성능이 높더라도 외부 병원 데이터나 다른 촬영 조건의 데이터에서는 성능이 낮아질 수 있다.

이러한 현상을 Domain Shift라고 한다. 의료 영상 AI에서는 Domain Shift가 특히 중요하다. 병원별 촬영 장비, 환자 연령대, 질병 분포, 라벨링 기준, 이미지 전처리 방식 등이 모두 모델 성능에 영향을 줄 수 있기 때문이다.

따라서 본 프로젝트는 다음 문제를 중심으로 진행한다.

> Kaggle 데이터셋으로 학습한 폐렴 분류 모델이 RSNA 외부 데이터셋에서도 안정적으로 작동하는가?

이를 확인하기 위해 내부 validation 성능과 외부 validation 성능을 비교하고, 성능 차이를 Bootstrap 95% Confidence Interval을 통해 통계적으로 보고한다.

---

## 4. 프로젝트 개요

본 프로젝트는 Kaggle의 `Chest X-Ray Images (Pneumonia)` 데이터셋을 학습용으로 사용하고, `RSNA Pneumonia Detection Challenge` 데이터셋을 외부 검증용으로 사용한다.

모델은 ResNet50 기반 전이학습을 중심으로 구현하며, 내부 validation 결과를 바탕으로 임계값을 결정한 뒤, 동일한 임계값을 외부 데이터셋에 적용한다. 이를 통해 외부 데이터셋에서의 성능 저하 여부를 공정하게 평가한다.

### 전체 흐름

```text
Kaggle Chest X-Ray Dataset
        ↓
Data Preprocessing
        ↓
Train / Validation Split
        ↓
ResNet50 Transfer Learning
        ↓
Internal Validation
        ↓
Threshold Selection using Youden's J
        ↓
RSNA External Validation
        ↓
Bootstrap Confidence Interval
        ↓
Domain Shift Analysis
        ↓
Grad-CAM Visualization
```

---

## 5. 데이터셋

## 5.1 학습용 데이터셋: Kaggle Chest X-Ray Images

학습용 데이터셋은 Kaggle에 공개된 `Chest X-Ray Images (Pneumonia)` 데이터셋이다.

| 항목 | 내용 |
|---|---|
| 출처 | Kaggle Chest X-Ray Images (Pneumonia) |
| 원 출처 | 광저우 여성·아동 의료센터 |
| 대상 | 1~5세 소아 환자 |
| 형식 | JPEG Grayscale Chest X-ray |
| 총 이미지 수 | 5,863장 |
| 정상 이미지 | 1,583장 |
| 폐렴 이미지 | 4,280장 |

원본 validation set은 이미지 수가 16장으로 매우 적어 통계적 신뢰성이 낮다. 따라서 본 프로젝트에서는 training set을 자체적으로 train/validation으로 재분할하여 사용한다.

데이터 분할 시 환자 ID 기반 GroupShuffleSplit을 적용하여 동일 환자의 이미지가 train과 validation에 동시에 포함되지 않도록 한다. 다만 NORMAL 클래스의 경우 환자 ID 추출에 한계가 있을 수 있으므로, 이 부분은 프로젝트 한계로 명시한다.

### 클래스 불균형 처리

Kaggle 데이터셋은 폐렴 이미지 비율이 약 73%로 높다. 따라서 다음 방법을 사용해 클래스 불균형 문제를 완화한다.

| 방법 | 설명 |
|---|---|
| Data Augmentation | 학습 데이터 다양성 증가 |
| Class Weight | 손실 함수에서 소수 클래스 가중치 반영 |
| 핵심 지표 조정 | Accuracy보다 F1, Recall, AUC 중심 평가 |

---

## 5.2 외부 검증용 데이터셋: RSNA Pneumonia Detection Challenge

외부 검증용 데이터셋은 RSNA Pneumonia Detection Challenge 데이터셋이다.

| 항목 | 내용 |
|---|---|
| 출처 | RSNA Pneumonia Detection Challenge |
| 형식 | DICOM Chest X-ray |
| 규모 | 약 26,000장 |
| 라벨링 | 방사선과 전문의 재라벨링 |
| 사용 방식 | 학습에는 사용하지 않고 외부 검증에만 사용 |

본 프로젝트에서는 계산 자원과 Bootstrap 반복 횟수를 고려하여 각 클래스에서 1,000장씩 무작위 추출하여 총 2,000장을 외부 검증에 사용한다.

| 클래스 | 샘플 수 |
|---|---:|
| Normal | 1,000 |
| Pneumonia | 1,000 |
| Total | 2,000 |

RSNA 데이터셋은 학습 과정에서 일절 사용하지 않고, 추론 및 외부 검증 단계에서만 로드한다. 이를 통해 데이터 누수를 방지한다.

---

## 6. 적용 기술

### 핵심 기술 스택

| 구분 | 기술 |
|---|---|
| Language | Python 3.10+ |
| Deep Learning Framework | PyTorch |
| Pretrained Model Library | timm |
| Data Augmentation | albumentations |
| Evaluation | torchmetrics, scikit-learn |
| Explainable AI | Grad-CAM |
| DICOM Processing | pydicom |
| Experiment Environment | KHU SERAPH GPU Cluster |
| Job Scheduler | Slurm, sbatch |
| Version Control | Git, GitHub |

---

## 7. 모델 아키텍처

본 프로젝트에서는 단순 CNN 모델을 베이스라인으로 사용하고, ResNet50 전이학습 모델과 TorchXRayVision 사전학습 모델을 후보 모델로 비교한다.

| 모델 | 설명 | 목적 |
|---|---|---|
| Baseline CNN | 직접 설계한 단순 CNN | 최소 성능 기준선 설정 |
| ResNet50 | ImageNet 사전학습 기반 전이학습 모델 | 일반 이미지 도메인 사전학습 활용 |
| TorchXRayVision Model | 의료 영상 사전학습 모델 | 흉부 X-ray 도메인 사전학습 활용 |

최종 모델은 내부 validation 성능을 기준으로 선정한다. 핵심 평가 지표는 AUC, F1, Recall이다.

---

## 8. 학습 및 평가 설정

### 전처리

내부 데이터셋과 외부 데이터셋 모두 동일한 전처리 과정을 적용한다.

| 처리 | 내용 |
|---|---|
| Resize | 224 × 224 |
| Channel | Grayscale → 3-channel |
| Normalization | ImageNet mean/std |
| Format | Tensor 변환 |

### 데이터 증강

학습 단계에서만 데이터 증강을 적용한다.

| Augmentation | 설명 |
|---|---|
| RandomHorizontalFlip | 좌우 반전 |
| RandomRotation | ±10도 회전 |
| RandomResizedCrop | 무작위 크롭 후 리사이즈 |
| Brightness/Contrast | 제한적 밝기·대비 조정 |

### 학습 설정

| 항목 | 설정 |
|---|---|
| Loss Function | BCEWithLogitsLoss with class weight |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR 또는 ReduceLROnPlateau |
| Early Stopping | Validation 성능 기준 |
| Random Seeds | 42, 2025, 31337 |

### 평가 지표

| 지표 | 사용 이유 |
|---|---|
| F1-score | 클래스 불균형 상황에서 Precision과 Recall 균형 평가 |
| Recall | 폐렴 환자를 놓치지 않는 능력 평가 |
| AUC-ROC | 임계값 변화에 따른 전반적 분류 성능 평가 |
| Accuracy | 보조 지표로만 사용 |

---

## 9. 핵심 검증 방법

## 9.1 3 Random Seed 반복 학습

단일 실험 결과만 보고하지 않고, 3개의 random seed로 반복 학습하여 평균과 표준편차를 함께 보고한다.

사용 seed는 다음과 같다.

```text
42
2025
31337
```

이를 통해 모델 성능이 특정 초기값이나 데이터 분할에 과하게 의존하지 않는지 확인한다.

---

## 9.2 Bootstrap 95% Confidence Interval

내부 validation set과 RSNA 외부 validation set에서 각각 Bootstrap resampling을 수행한다.

| 항목 | 설정 |
|---|---|
| 반복 횟수 | 1,000회 |
| 대상 지표 | AUC, F1 |
| 산출 결과 | 95% Confidence Interval |

Bootstrap CI를 통해 내부 성능과 외부 성능 차이가 단순한 우연인지, 통계적으로 의미 있는 차이인지 확인한다.

---

## 9.3 Youden's J Statistic 기반 임계값 고정

내부 validation ROC curve에서 Youden's J statistic을 사용해 최적 임계값을 결정한다.

```text
Youden's J = Sensitivity + Specificity - 1
```

이 임계값을 외부 데이터셋에도 동일하게 적용한다. 외부 데이터셋에 맞춰 임계값을 다시 조정하지 않기 때문에, 실제 배포 상황에 더 가까운 평가가 가능하다.

---

## 9.4 Grad-CAM 시각화

Grad-CAM을 활용하여 모델이 X-ray 이미지의 어느 영역을 보고 폐렴 여부를 판단했는지 시각화한다.

이를 통해 내부 데이터셋과 외부 데이터셋에서 모델의 주목 영역이 달라지는지 정성적으로 비교한다.

주요 확인 사항은 다음과 같다.

| 항목 | 설명 |
|---|---|
| 폐 영역 집중 여부 | 모델이 실제 폐 영역을 보고 판단하는지 확인 |
| 배경 정보 의존 여부 | 병원 마커, 이미지 외곽, 텍스트 등에 의존하는지 확인 |
| 내부/외부 차이 | 데이터셋 변화에 따라 주목 영역이 달라지는지 비교 |

---

## 10. 주요 기능

본 프로젝트의 주요 기능은 다음과 같다.

| 기능 | 설명 |
|---|---|
| X-ray 이진 분류 | 정상/폐렴 확률값 출력 |
| 반복 학습 | 3 random seed 기반 반복 실험 |
| 내부 검증 | Kaggle validation set 성능 측정 |
| 외부 검증 | RSNA 데이터셋 성능 측정 |
| Bootstrap CI | 성능 지표의 95% 신뢰구간 산출 |
| Threshold 고정 | 내부 validation 기준 임계값을 외부 검증에 동일 적용 |
| Grad-CAM | 모델 주목 영역 시각화 |
| 성능 비교표 | 내부/외부 성능 차이 자동 정리 |

---

## 11. 결과물

### 필수 산출물

| 산출물 | 설명 |
|---|---|
| 모델 가중치 | 3 seed별 학습된 `.pt` 파일 |
| 학습 코드 | Baseline CNN 및 ResNet50 학습 파이프라인 |
| 내부 검증 보고서 | F1, AUC, Bootstrap CI 포함 |
| 외부 검증 보고서 | RSNA 성능 및 내부/외부 비교 |
| Grad-CAM 결과 | 대표 이미지 시각화 |
| sbatch 스크립트 | SERAPH 클러스터 실행용 스크립트 |
| GitHub Repository | 전체 코드 및 문서 공개 |

### 선택 산출물

| 산출물 | 설명 |
|---|---|
| Grad-CAM 무게중심 분석 | 모델 주목 영역 정량화 |
| False Negative Gallery | 폐렴을 놓친 케이스 정리 |
| Gradio Demo | 간단한 웹 기반 추론 데모 |
| Additional Backbone Comparison | EfficientNet 등 추가 모델 비교 |
| Calibration Analysis | Reliability Diagram, ECE 분석 |

---

## 12. 역할 분담

| 담당자 | 역할 |
|---|---|
| 조민규 | Kaggle/RSNA 데이터 수집 및 전처리, RSNA DICOM 메타데이터 EDA, 환자 ID 기반 GroupShuffleSplit 구현, albumentations 증강 파이프라인 구성, Baseline CNN 학습 |
| 강우혁 | ResNet50 전이학습 구현, KHU SERAPH 환경 설정, sbatch 스크립트 작성, 3 seed 반복 학습 실행, 내부 validation 성능 측정, Youden's J 임계값 결정, Grad-CAM 시각화 |
| 정희승 | RSNA 외부 검증 수행, 고정 임계값 적용, Bootstrap 95% CI 산출, 내부 vs 외부 비교표 작성, 최종 보고서 작성, 선택적으로 Grad-CAM 무게중심 분석 및 FN 케이스 갤러리 작성 |

---

## 13. 기대 효과

## 13.1 사용자 측면

흉부 X-ray 판독 보조 도구로 활용하여 1차 스크리닝 속도를 향상시킬 수 있다. 특히 의료 인력이 부족한 지역이나 기관에서 폐렴 의심 사례를 빠르게 선별하는 데 도움을 줄 수 있다.

## 13.2 비즈니스 측면

외부 검증을 통해 모델의 실제 임상 적용 가능성을 사전에 평가할 수 있다. 단순히 높은 내부 성능을 제시하는 것이 아니라, 외부 데이터셋에서의 성능 변화를 함께 보고하기 때문에 의료 AI 모델의 신뢰성과 실용성을 더 현실적으로 판단할 수 있다.

## 13.3 개발자 측면

3 seed 반복 학습, Bootstrap 95% CI, Youden's J 임계값 고정, Grad-CAM 시각화를 포함한 검증 파이프라인은 의료 AI 연구에서 재현 가능한 베이스라인으로 활용될 수 있다.

---

## 14. 활용 방안

## 14.1 단기 활용 방안

프로젝트 종료 후 GitHub에 오픈소스로 공개하여 학술 및 교육 자료로 활용한다. Kaggle Notebook을 함께 공유하여 유사 연구자들이 실험 과정을 쉽게 참고할 수 있도록 한다.

## 14.2 중기 활용 방안

세균성/바이러스성 폐렴 세분류 모델로 확장한다. 또한 다기관 X-ray 데이터셋을 추가 수집하여 Domain Shift 요인을 더 세분화해 분석한다. 경량화 모델을 적용하여 모바일 및 edge device 배포 가능성도 탐색한다.

## 14.3 장기 활용 방안

DICOM 형식 지원과 PACS 연동 모듈을 개발하여 실제 임상 환경 적용 가능성을 검토한다. 또한 의료 AI 소프트웨어 인허가 절차를 참고하여 의료기관 파일럿 적용 가능성을 탐색한다. Grad-CAM 외에도 SHAP, LIME 등 다양한 XAI 기법을 적용하여 의료진의 신뢰도를 높이는 방향으로 고도화할 수 있다.

---

## 15. 프로젝트 의의

본 프로젝트의 핵심 의의는 단순한 폐렴 분류 모델 구현이 아니라, 외부 검증을 통해 모델의 일반화 가능성을 확인한다는 점에 있다.

의료 AI 모델은 내부 데이터셋에서 높은 성능을 보이는 것만으로는 충분하지 않다. 실제 환경에서는 데이터 분포가 달라지고, 그에 따라 모델 성능이 예상보다 크게 낮아질 수 있다. 따라서 본 프로젝트는 Kaggle 데이터셋 학습과 RSNA 외부 검증을 결합하여 Domain Shift 문제를 직접 확인하고, 이를 통계적으로 보고하는 재현 가능한 실험 파이프라인을 제시한다.

---

## 16. Repository 구조 예시

```text
aip_project/
├── README.md
├── docs/
│   └── proposal.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   ├── datasets/
│   ├── models/
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
├── scripts/
│   └── train.sbatch
├── outputs/
│   ├── checkpoints/
│   ├── reports/
│   └── figures/
└── requirements.txt
```

---

## 17. 실행 계획 요약

| 단계 | 내용 |
|---|---|
| 1단계 | Kaggle/RSNA 데이터 수집 및 구조 정리 |
| 2단계 | 데이터 전처리 및 train/validation split 구현 |
| 3단계 | Baseline CNN 학습 |
| 4단계 | ResNet50 전이학습 구현 |
| 5단계 | 3 seed 반복 학습 |
| 6단계 | 내부 validation 성능 측정 |
| 7단계 | Youden's J 기반 threshold 결정 |
| 8단계 | RSNA 외부 검증 |
| 9단계 | Bootstrap CI 계산 |
| 10단계 | Grad-CAM 시각화 및 최종 보고서 작성 |

---

## 18. 요약

본 프로젝트는 흉부 X-ray 기반 폐렴 분류 모델을 개발하고, 외부 데이터셋 검증을 통해 Domain Shift 문제를 분석하는 의료 AI 프로젝트이다. Kaggle 데이터셋으로 모델을 학습하고 RSNA 데이터셋으로 외부 검증을 수행함으로써, 단일 데이터셋 중심의 성능 보고 한계를 보완한다.

최종적으로 본 프로젝트는 폐렴 분류 모델, 외부 검증 파이프라인, Bootstrap 신뢰구간 분석, Grad-CAM 시각화 결과를 포함하는 재현 가능한 의료 AI 연구 베이스라인을 제공하는 것을 목표로 한다.
