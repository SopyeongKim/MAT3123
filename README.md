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


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# 1. 설정 및 하이퍼파라미터 (Configuration)
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 주식 티커 설정 (예: 삼성전자 '005930.KS', 애플 'AAPL')
TICKER = '005930.KS' 
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'

SEQ_LENGTH = 60   # 지난 60일의 데이터를 보고
PREDICT_DAYS = 30 # 향후 30일을 예측
SIMULATIONS = 1000 # 몬테카를로 시뮬레이션 횟수 (MCMC)

# ---------------------------------------------------------
# 2. 데이터 전처리 (Data Preparation)
# ---------------------------------------------------------
def load_data(ticker):
    print(f"{ticker} 데이터를 다운로드 중입니다...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE)
    if df.empty:
        raise ValueError("데이터를 가져올 수 없습니다. 티커를 확인해주세요.")
    data = df['Close'].values.reshape(-1, 1)
    return data

# 정규화 (MinMax Scaling) - 수업 시간 딥러닝 학습의 필수 과정
scaler = MinMaxScaler()
raw_data = load_data(TICKER)
scaled_data = scaler.fit_transform(raw_data)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Tensor 변환 (PyTorch)
X_tensor = torch.from_numpy(X).float().to(device)
y_tensor = torch.from_numpy(y).float().to(device)

# ---------------------------------------------------------
# 3. LSTM 모델 정의 (Model Architecture)
# 수업 시간 배운 LSTM 구조 직접 구현
# ---------------------------------------------------------
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # 마지막 시퀀스의 출력만 사용
        return out

model = StockLSTM().to(device)

# ---------------------------------------------------------
# 4. 학습 루프 (Training Loop - Gradient Descent)
# ---------------------------------------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("모델 학습 시작 (Gradient Descent)...")
epochs = 100
for epoch in range(epochs):
    model.train()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    optimizer.zero_grad() # Gradient 초기화 (수업 강조 내용)
    loss.backward()       # Backpropagation
    optimizer.step()      # Parameter Update
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# ---------------------------------------------------------
# 5. 몬테카를로 시뮬레이션 (MCMC & Langevin Dynamics Logic)
# ---------------------------------------------------------
print("몬테카를로 시뮬레이션 수행 중...")

# 5.1 최근 데이터를 기반으로 '추세(Drift)' 예측
model.eval()
last_sequence = X_tensor[-1].unsqueeze(0)
predicted_trend = []

curr_seq = last_sequence
with torch.no_grad():
    for _ in range(PREDICT_DAYS):
        pred = model(curr_seq)
        predicted_trend.append(pred.item())
        # 다음 예측을 위해 시퀀스 업데이트 (Sliding Window)
        new_seq = torch.cat((curr_seq[:, 1:, :], pred.unsqueeze(1)), dim=1)
        curr_seq = new_seq

# 역정규화 (원래 가격으로 변환)
predicted_trend = scaler.inverse_transform(np.array(predicted_trend).reshape(-1, 1))
current_price = scaler.inverse_transform(y_tensor[-1].cpu().detach().numpy().reshape(-1, 1))[0][0]

# 5.2 변동성(Volatility) 계산 - 노이즈의 크기 결정
# 최근 60일간의 로그 수익률 표준편차 사용
recent_prices = scaler.inverse_transform(X[-1].reshape(-1, 1))
returns = np.diff(np.log(recent_prices), axis=0)
volatility = np.std(returns)

# 5.3 랑주뱅 역학 기반 시뮬레이션 (SDE)
# S_{t+1} = S_t + (Drift) + (Diffusion/Noise)
simulation_results = np.zeros((PREDICT_DAYS, SIMULATIONS))
simulation_results[0, :] = current_price

for t in range(1, PREDICT_DAYS):
    # LSTM이 예측한 '오늘 대비 내일의 변화율'을 Drift로 사용
    drift = (predicted_trend[t] - predicted_trend[t-1]) / predicted_trend[t-1]
    
    # 랜덤 노이즈 생성 (수업시간 MCMC/Langevin에서 배운 정규분포 노이즈)
    shock = volatility * np.random.normal(0, 1, SIMULATIONS)
    
    # 주가 업데이트 수식 적용
    simulation_results[t, :] = simulation_results[t-1, :] * (1 + drift + shock)

# ---------------------------------------------------------
# 6. 결과 시각화 (Visualization)
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))

# 과거 주가 (최근 일부만 표시)
plt.plot(range(len(raw_data)-100, len(raw_data)), raw_data[-100:], label='History', color='blue')

# 시뮬레이션 경로들 (흐리게 표시하여 분포 느낌 주기)
future_idx = range(len(raw_data), len(raw_data) + PREDICT_DAYS)
plt.plot(future_idx, simulation_results, color='gray', alpha=0.05)

# LSTM이 예측한 기준 추세선 (Mean)
plt.plot(future_idx, predicted_trend, label='LSTM Trend Prediction', color='red', linewidth=2)

# 신뢰 구간 (Confidence Interval) - MCMC의 장점
mean_sim = np.mean(simulation_results, axis=1)
std_sim = np.std(simulation_results, axis=1)
plt.fill_between(future_idx, mean_sim - 1.96*std_sim, mean_sim + 1.96*std_sim, color='orange', alpha=0.2, label='95% Confidence Interval')

plt.title(f'Monte Carlo Simulation with LSTM Trend: {TICKER}')
plt.xlabel('Time Step')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

print("완료! 그래프 창을 확인하세요.")
