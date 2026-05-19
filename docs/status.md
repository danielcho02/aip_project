# Project Status

기준 시점: 2026.05.19 11:35 KST

## 완료된 작업

- SERAPH에서 Kaggle/RSNA 데이터셋 다운로드 완료
- moana-y2 컴퓨트 노드에서 `/local_datasets/daniel3290`에 압축 해제 완료
- Kaggle train 데이터 기준 train/validation split 생성 완료
- Baseline CNN 학습 코드 작성 및 실행 완료

## 데이터 확인

- Kaggle train NORMAL: 1341
- Kaggle train PNEUMONIA: 3875
- Kaggle test NORMAL: 234
- Kaggle test PNEUMONIA: 390
- RSNA train images: 26684

## Kaggle split 결과

생성 파일:

- `outputs/splits/kaggle_split_seed42.csv`

split 결과:

- train NORMAL: 1073
- train PNEUMONIA: 3092
- val NORMAL: 268
- val PNEUMONIA: 783
- total: 5216

## Baseline CNN 결과

실행 환경:

- SERAPH `moana-y2`
- seed: 42
- epochs: 5
- batch size: 32
- lr: 1e-3

Validation 결과:

- Best val AUC: 0.9613
- Accuracy: 0.9115
- Precision: 0.9612
- Recall: 0.9183
- F1-score: 0.9393

저장된 결과:

- `outputs/baseline/best_baseline_seed42.pt`
- `outputs/baseline/metrics_seed42.json`

위 파일들은 서버 산출물이고 GitHub에는 올리지 않음.

## 현재 추가된 코드

- `src/prepare_kaggle_split.py`
- `src/dataset.py`
- `src/models/baseline_cnn.py`
- `src/train_baseline.py`

## 다음 작업 후보

- ResNet50 전이학습 코드 작성
- 내부 validation threshold 계산
- RSNA 외부 검증 코드 작성
