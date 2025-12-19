# <The Langevin Trader: LSTM & SDE 기반 주가 예측 시뮬레이터>

&lt;기계학습과 응용 기말고사 프로젝트> 학부 기계학습과 응용 수업에서 배운 MCMC와 Langevin Dynamics, 그리고 LSTM을 결합한 프로젝트를 설계.

<The Langevin Trader: LSTM & SDE 기반 주가 예측 시뮬레이터>

# 1. 개요 (Abstract)
기존의 주가 예측 모델들은 단순히 '내일의 가격'을 점 하나로 예측하려다 보니 정확도가 떨어지고 리스크 관리가 불가능했습니다. 
본 프로젝트는 '딥러닝(LSTM)'으로 주가의 추세를 학습하고, 수업 시간에 배운 '확률 미분 방정식'과 '몬테카를로 시뮬레이션(Monte Carlo)'을 결합하여 미래 주가의 '확률적 분포'를 예측하는 하이브리드 시스템입니다.

# 2. 수업 내용 적용 (Curriculum Mapping)
이 프로젝트는 '기계학습과 응용' 수업의 핵심 이론들을 실제 금융 데이터에 적용했습니다.

 2.1 시계열 추세 학습 (Time Series Learning)
* 적용 기술: LSTM (Long Short-Term Memory)
* 수업 연계: RNN/LSTM 강의 (11월 17일)
* 구현: 주가 데이터의 시퀀스(Sequence)를 입력받아 장기 의존성(Long-term dependency)을 학습하고, 주가의 Drift(이동 방향)를 예측합니다.

 2.2 확률적 시뮬레이션 (Stochastic Simulation)
* 적용 기술: MCMC & Langevin Dynamics (SDE)
* 수업 연계: MCMC 및 Diffusion Model 강의
* 구현: 미래의 주가를 결정론적(Deterministic)으로 계산하지 않고, 랑주뱅 역학의 원리인 'Gradient(LSTM 예측값) + Noise(시장 변동성)' 구조를 사용하여 1,000번 이상의 시뮬레이션을 수행합니다.
    * 수식: $S_{t+1} = S_t + \mu(LSTM) \Delta t + \sigma \epsilon \sqrt{\Delta t}$
    * ($\mu$: LSTM이 학습한 추세, $\sigma$: 변동성, $\epsilon$: 가우시안 노이즈)

 2.3 최적화 (Optimization)
* 적용 기술: Adam Optimizer & MSE Loss
* 수업 연계: Gradient Descent 강의 (9월 24일)
* 구현: LSTM 모델 학습 시 역전파(Backpropagation)를 통해 손실을 최소화합니다.

# 3. 실행 방법 (How to Run)
 3.1. 필요 라이브러리 설치: `pip install -r requirements.txt`
 3.2. 실행: `python main.py`
 3.3. 기업 코드(Ticker) 입력 (예: 삼성전자 `005930.KS`, 애플 `AAPL`)
 3.4. 결과 그래프 확인 (LSTM 예측선 + 몬테카를로 시뮬레이션 범위)

# 4. 결과 해석
* 파란선: 실제 과거 주가
* 빨간선: LSTM이 예측한 기준 추세선 (Drift)
* 회색 영역: 몬테카를로 시뮬레이션으로 예측된 주가 변동 범위 (불확실성 시각화)
