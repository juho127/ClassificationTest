# CIFAR-10 MLP 분류 실습

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juho127/ClassificationTest/blob/main/cifar10_mlp_tutorial.ipynb)

강의 실습용 CIFAR-10 데이터셋 MLP(다층 퍼셉트론) 분류 코드입니다.

## 🚀 빠른 시작 (Google Colab 추천!)

**가장 쉬운 방법**: 위의 "Open in Colab" 배지를 클릭하세요!
- ✅ 설치 불필요 (웹 브라우저만 있으면 OK)
- ✅ 무료 GPU 사용 가능
- ✅ 구글 계정만 있으면 바로 시작

## 💻 로컬 설치 방법

```bash
pip install -r requirements.txt
```

## 실행 방법

### 방법 1: Google Colab (추천)
1. 위의 "Open in Colab" 배지 클릭
2. 구글 계정으로 로그인
3. 런타임 > 런타임 유형 변경 > GPU 선택 (선택사항)
4. 셀을 순서대로 실행 (Shift + Enter)

### 방법 2: Python 스크립트
```bash
# MLP 기본 실습
python cifar10_mlp.py

# 모델 비교 (MLP vs CNN vs ViT)
python model_comparison.py
```

### 방법 3: Jupyter 노트북 (로컬)
```bash
# MLP 기본 실습
jupyter notebook cifar10_mlp_tutorial.ipynb

# 모델 비교
jupyter notebook model_comparison.ipynb
```

## 📚 실습 자료

### 1. MLP 기본 실습
- `cifar10_mlp.py` - Python 스크립트
- `cifar10_mlp_tutorial.ipynb` - Jupyter 노트북
- MLP 기초 이해 및 PyTorch 사용법 학습

### 2. 모델 비교 실습 ⭐ NEW!
- `model_comparison.py` - Python 스크립트
- `model_comparison.ipynb` - Jupyter 노트북 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juho127/ClassificationTest/blob/main/model_comparison.ipynb)
- **MLP vs CNN vs ViT** 세 가지 아키텍처 비교
- 성능, 학습 시간, 파라미터 수 비교
- 클래스별 정확도 분석

## 프로그램 구조

### 1. 데이터셋
- **CIFAR-10**: 32x32 크기의 컬러 이미지 60,000개
- **클래스**: 10개 (비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭)
- **훈련 데이터**: 50,000개
- **테스트 데이터**: 10,000개

### 2. 모델 아키텍처 (MLP)
```
입력층: 3072 (32x32x3)
    ↓
은닉층 1: 512 뉴런 + ReLU + Dropout(0.3)
    ↓
은닉층 2: 256 뉴런 + ReLU + Dropout(0.3)
    ↓
출력층: 10 (클래스 개수)
```

### 3. 하이퍼파라미터
- **배치 크기**: 128
- **학습률**: 0.001
- **에포크 수**: 20
- **옵티마이저**: Adam
- **손실 함수**: CrossEntropyLoss

## 출력 파일

1. **cifar10_samples.png**: 데이터셋 샘플 이미지
2. **training_history.png**: 학습 과정 그래프 (손실/정확도)
3. **best_model.pth**: 최고 성능 모델 가중치

## 예상 성능

### MLP 단독 실습
- **테스트 정확도**: 약 50-55%
- **참고**: MLP는 이미지의 공간적 구조를 활용하지 못하므로 CNN보다 성능이 낮습니다.

### 모델 비교 실습 (model_comparison)
- **MLP**: ~50-55% (빠른 학습, 단순)
- **CNN**: ~70-75% (최고 성능, 공간 구조 활용)
- **ViT**: ~65-70% (글로벌 어텐션, 더 많은 데이터 필요)

## 학습 내용

이 실습을 통해 다음을 배울 수 있습니다:

1. PyTorch로 데이터셋 로드 및 전처리
2. MLP 모델 구현 및 학습
3. 모델 평가 및 성능 측정
4. 학습 과정 시각화
5. 클래스별 정확도 분석

## 주요 코드 설명

### MLP 모델
```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3072, 512)  # 첫 번째 은닉층
        self.fc2 = nn.Linear(512, 256)   # 두 번째 은닉층
        self.fc3 = nn.Linear(256, 10)    # 출력층
```

### 학습 루프
1. 순전파(Forward Pass)
2. 손실 계산
3. 역전파(Backward Pass)
4. 가중치 업데이트

## 개선 아이디어

실습 후 다음을 시도해보세요:

1. **하이퍼파라미터 조정**
   - 학습률 변경
   - 은닉층 크기 변경
   - 에포크 수 증가

2. **모델 구조 변경**
   - 은닉층 추가
   - Dropout 비율 조정
   - 다른 활성화 함수 사용 (LeakyReLU, ELU 등)

3. **데이터 증강**
   - RandomHorizontalFlip
   - RandomCrop
   - ColorJitter

4. **CNN으로 확장**
   - Convolutional Layer 추가
   - Pooling Layer 추가
   - 성능 비교

## 문제 해결

- **GPU 사용**: CUDA가 설치되어 있으면 자동으로 GPU를 사용합니다.
  - Colab: 런타임 > 런타임 유형 변경 > GPU
- **메모리 부족**: 배치 크기를 줄여보세요 (예: 64, 32)
- **느린 학습**: NUM_EPOCHS를 줄이거나 더 작은 모델을 사용하세요.
- **Colab 세션 종료**: 90분 미사용 시 종료됩니다. 중요한 파일은 Drive에 저장하세요.

## 📚 추가 문서

- [Google Colab 설정 가이드](COLAB_SETUP.md) - 강의 자료 배포 시 참고하세요

