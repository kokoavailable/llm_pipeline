import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# 데이터 로드
ticker = yf.Ticker("GOOGL")
raw_df = ticker.history(period="max", auto_adjust=False)

raw_df['3MA'] = raw_df['Adj Close'].rolling(window=3).mean()
raw_df['5MA'] = raw_df['Adj Close'].rolling(window=5).mean()

# 데이터 로드 점검
# 팔레트 생성
plt.figure(figsize=(7, 4))
plt.title('Google Stock Price History')
plt.ylabel('price (won)')
plt.xlabel('period (day)')
plt.grid()
plt.plot(raw_df['Adj Close'], label='Adj Close', color='b') # 그래프 그리기, 범례 추가
plt.legend(loc='best')

plt.show()

# 데이터 전처리
# 결측치 제거
raw_df['Volume'] = raw_df['Volume'].replace(0, np.nan)

for col in raw_df.columns:
    missing_rows = raw_df.loc[raw_df[col]==0].shape[0]
    print(col + ': ' + str(missing_rows))

raw_df = raw_df.dropna()

scale_cols = [ 'Open', 'High', 'Low', 'Close', 'Adj Close', '3MA', '5MA', 'Volume']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(raw_df[scale_cols])
scaled_df = pd.DataFrame(scaled_df, columns=scale_cols)

# 특성과 라벨 설정
feature_cols = ['3MA', '5MA', 'Adj Close']
label_cols = ['Adj Close']

label_df = pd.DataFrame(scaled_df, columns = label_cols)
feature_df = pd.DataFrame(scaled_df, columns = feature_cols)

label_np = label_df.to_numpy()
feature_np = feature_df.to_numpy()

## 시퀀스 데이터 생성
def make_sequence_dataset(feature, label, window_size):
    feature_list = []
    label_list = []

    for i in range(len(feature) - window_size):
        feature_list.append(feature[i:i + window_size])
        label_list.append(label[i + window_size])

    return np.array(feature_list), np.array(label_list)

window_size = 40
X, Y = make_sequence_dataset(feature_np, label_np, window_size)

# 훈련/테스트 array 분할
split = -200
x_train = X[:split]
y_train = Y[:split]
x_test = X[split:]
y_test = Y[split:]

# 텐서 변환
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)

# DataLoader 준비
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# GRU 모델 정의
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

model = GRUModel(input_size =3, hidden_size=256, output_size=1)

# 손실 함수 및 옵티마이저
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 100
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        output = model(x_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 검증
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x_val, y_val in test_loader:
            val_output = model(x_val)
            val_loss = criterion(val_output, y_val)
            val_losses.append(val_loss.item())
    val_loss = np.mean(val_losses)

    print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# 예측 및 시각화
model.eval()
with torch.no_grad():
    pred = model(x_test).numpy()

plt.figure(figsize=(12, 6))
plt.title('3MA + 5MA + Adj Close, window_size=40')
plt.ylabel('adj close')
plt.xlabel('period')
plt.plot(y_test, label='actual')
plt.plot(pred, label='prediction')
plt.grid()
plt.legend(loc='best')

plt.show()