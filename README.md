#  The Langevin Trader: LSTM & SDE 기반 주가 예측 시뮬레이터

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![MCMC](https://img.shields.io/badge/Method-MCMC%20%26%20SDE-orange)

## 1. 개요 (Abstract)
> "주가는 하나의 점이 아니라, 확률의 구름(Distribution)으로 예측해야 한다."

기존의 단순 딥러닝 주가 예측 모델들은 미래의 불확실성(Uncertainty)을 고려하지 않고 단일 가격만을 예측하여 리스크 관리에 취약하다는 단점이 있었습니다.

본 프로젝트 [The Langevin Trader]는 딥러닝(LSTM)으로 주가의 장기적인 '추세(Drift)'를 학습하고, 수업 시간에 배운 랑주뱅 역학(Langevin Dynamics)을 적용하여 시장의 무작위성(Noise)을 반영한 "확률적 시뮬레이터"입니다. 사용자가 원하는 기업(한국/미국)의 코드를 입력하면, AI가 해당 기업의 미래 주가 변동 범위를 예측하여 시각화합니다.

---

## 2. 수업 내용 적용 및 기술적 배경 (Theoretical Background)
본 프로젝트는 [2025-2 기계학습과 응용] 수업의 커리큘럼을 단계별로 충실히 구현하여 설계되었습니다.

### 2.1 시계열 추세 학습 (Deep Learning)
* 적용 이론: RNN & LSTM (Long Short-Term Memory)
* 수업 연계: [LSTM]
* 구현 내용:
    * 주가 데이터는 전형적인 시계열(Time-series) 데이터입니다.
    * 단순 RNN의 장기 의존성(Long-term dependency) 문제를 해결하기 위해 LSTM 모델을 직접 설계했습니다 (`StockLSTM` 클래스).
    * 과거 60일간의 데이터를 `Input Gate`, `Forget Gate`를 통과시켜 주가의 상승/하락 모멘텀(Drift)을 추출합니다.

### 2.2 최적화 및 학습 (Optimization)
* 적용 이론: Gradient Descent & Backpropagation
* 수업 연계: [Gradient Descent]
* 구현 내용:
    * 외부 API에 의존하지 않고 PyTorch의 학습 루프를 직접 구현했습니다.
    * `optimizer.zero_grad()`로 그레디언트 누적을 방지하고, `loss.backward()`로 오차를 역전파하여 파라미터를 업데이트합니다.

### 2.3 확률적 시뮬레이션 (Stochastic Process)
* 적용 이론: MCMC (Markov Chain Monte Carlo) & Langevin Dynamics
* 수업 연계:  [MCMC], [Diffusion Model]
* 구현 내용 (핵심 차별점):
    * 미래 주가를 결정론적(Deterministic)으로 계산하지 않고, 확률 미분 방정식(SDE)을 이산화(Discretization)하여 시뮬레이션합니다.
    * 수식 적용: $$S_{t+1} = S_t \times (1 + \mu \Delta t + \sigma \epsilon \sqrt{\Delta t})$$
        * $\mu$ (Drift): LSTM 모델이 예측한 주가의 기울기 사용.
        * $\sigma$ (Diffusion): 최근 주가의 변동성(Volatility)을 계산하여 적용.
        * $\epsilon$ (Noise): MCMC 강의에서 배운 정규분포($N(0,1)$) 노이즈 샘플링.

---

## 3. 주요 기능 (Features)
### 3.1.  다국적 기업 지원: 삼성전자, 카카오 등 국내 주식(.KS, .KQ) 뿐만 아니라 애플, 테슬라 등 해외 주식(Ticker) 데이터도 실시간으로 불러옵니다.
### 3.2.  하이브리드 예측: LSTM(인공지능)의 예측력과 Monte Carlo(통계학)의 다양성을 결합했습니다.
### 3.3.  리스크 시각화: 미래 주가가 움직일 수 있는 95% 신뢰 구간(Confidence Interval)을 시각화하여, 최악/최선의 시나리오를 한눈에 보여줍니다.

---

## 4. 실행 방법 (How to Run)

### 4.1 환경 설정필요한 라이브러리를 설치합니다.

Bash

pip install torch numpy pandas matplotlib yfinance scikit-learn

### 4.2 실행터미널에서 아래 명령어를 입력합니다.

Bash

python main.py

### 4.3 입력 예시프로그램이 실행되면 분석하고 싶은 기업의 코드를 입력하세요.

삼성전자: 엔터(Enter) 키 입력 (기본값)

현대차: 005380.KS에코프로: 086520.KQ

애플: AAPL엔비디아: NVDA5. 

## 결과 해석 (Result Interpretation)

그래프 창이 뜨면 다음과 같이 해석합니다.

● 파란 선 (History): 과거 실제 주가 흐름입니다.

● 빨간 선 (LSTM Trend): AI가 예측한 가장 유력한 미래 주가 추세입니다. (이동 평균적 성격)

● 회색 영역 (Simulations): 1,000번의 랑주뱅 시뮬레이션 결과입니다. 색이 진할수록 실현 가능성이 높은 구간입니다.

● 주황색 영역 (Confidence Interval): 통계적으로 주가가 존재할 확률이 95%인 범위입니다. 이 범위가 좁을수록 예측 신뢰도가 높습니다.


## 실행 예시
<img width="788" height="749" alt="image" src="https://github.com/user-attachments/assets/d3bf28ac-cd06-4ebb-b34d-28c42c663f0d" />

