import torch
import torch.nn as nn
import numpy as np
import yaml
import os
from torch.utils.data import DataLoader, TensorDataset
import time
import json

class ModelTrainer:
    def __init__(self, model, config_path):
        # 모델 객체를 받아 내부 변수로 저장한다.
        self.model = model
        
        # 설정 파일(config.yaml)을 읽어서 학습 설정을 로드한다.
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # 설정 파일에서 학습에 필요한 하이퍼파라미터를 추출한다.
        # 학습을 통한게 아닌 사람이 조정하는 값으로 하이퍼라는 접두사가 붙었다. 
        
        # 한번의 학습에서 모델이 얼마나 크게 매개변수를 조정할지
        # 학습속도와 관련이 있으나 갚이 너무 높으면 부정확할 수 있다. 
        self.learning_rate = self.config["training"]["learning_rate"]
       
        # 값이 높으면 더 많은 데이터를 한번에 처리하나, 일반화 성능이 떨어질수 있다. (노이즈의 영향은 덜 받음)
        self.batch_size = self.config["training"]["batch_size"]

        # 더 많이 학습한다. 더 좋은 성능을 보일 수 있으나, 과적합의 위험이 있다.
        # epoch는 전체 학습 데이터를 한번 학습하는 단위이다. 즉 학습을 몇번 반복할 것인가를 의미한다.
        # early stopping 과 함께 사용하는 것이 좋다.
        self.num_epochs = self.config["training"]["epochs"]

        # 조기 종료 기준이 되는 인내 횟수: 성능 향상이 없더라도 몇 번까지 기다릴지 설정
        self.patience = self.config["training"]["early_stopping"]["patience"]
        
        # 손실 함수 설정: 평균 제곱 오차를 사용한다.
        # 예측값과 실제값의 차이를 제곱하여 평균을 내는 방식 (회귀 문제에 적합)
        self.criterion = nn.MSELoss()

        # 옵티마이저 설정: Adam 옵티마이저를 사용한다.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def prepare_data_loaders(self, x_train, y_train, x_test, y_test):
        """
        데이터 로더를 준비.
        
        넘파이 배열 형태의 훈련 및 테스트 데이터를 텐서로 변환해, pytorch 학습에 사용한다. 
        데이터 로더 객체로 감싸 배치 단위로 공급되도록 구성한다.
        
        """
        # 넘파이 배열을 텐서로 변환한다.
        # 모델은 넘파이 배열이 아닌 토치 텐서 형태의 데이터를 요구한다.
        x_train_tensor = torch.FloatTensor(x_train)
        y_train_tensor = torch.FloatTensor(y_train)
        x_test_tensor = torch.FloatTensor(x_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # 텐서들을 x, y 쌍으로 묶어 데이터셋을 만든다.
        # 텐서 데이터셋은 파이토치가 제공하는 기본 데이터 셋으로, x, y 쌍으로 묶어주는 역할을 한다.
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        
        # 데이터를 배치단위로 잘라서 모델이 공급해주는 역할을 한다. 시계열 데이터기 때문에 suffle False
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def train(self, train_loader, test_loader, log_dir=None):
        """모델을 훈련시킨다."""

        # 현재까지 가장 낮은 검증, 손실 값을 저장 (초기값은 무한대)
        best_val_loss = float('inf')

        # 검증 손실이 좋아지지 않은 에폭 수를 세는 카운터 (Early Stopping용)
        counter = 0

        # 매 epoch마다의 손실 기록을 담는 리스트 (학습 기록)
        training_history = []
        
        # 전체 훈련 시간 기록용
        start_time = time.time()
        
        for epoch in range(self.num_epochs): # 에폭 수만큼 반복한다.
            # 모델을 학습 모드로 설정 (Dropout, BatchNorm 등 활성화)
            self.model.train()
            # 각 배치간 loss 값을 담는 리스트
            train_losses = []
            
            for x_batch, y_batch in train_loader:
                output = self.model(x_batch) # 모델 예측
                loss = self.criterion(output, y_batch) # 손실 함수 계산
                
                # PyTorch는 gradient를 기본적으로 누적(accumulate)하기 때문에,
                # 매 배치마다 gradient를 초기화해야 역전파 시에도 이전 값이 다음 계산에 영향을 주지 않는다.
                self.optimizer.zero_grad() 
                loss.backward() # 역전파: 손실에 따른 기울기 계산
                # 그라디언트를 바탕으로 파라미터 값 갱신
                self.optimizer.step() # 옵티마이저가 파라미터 갱신
                
                # oss는 토치의 텐서 객체로 연산그래프등 많은 불필요한 정보가 붙어있다. 
                # item 은 텐서에서 실제 숫자 값만 꺼내 float으로 반환한다.
                train_losses.append(loss.item()) # 손실값 저장 (그라디언트 제외)
            
            # 검증 단계, 평가 모드 시작 (모델이 Dropout, BatchNorm 등 비활성화)
            self.model.eval()
            val_losses = []
            
            # torch.no_grad()는 해당 블록 내에 파이토치가 자동미분을 하지 않도록 설정한다.
            # 그라디언트, 역전파를 비활성화화여 연산속도 및 메모리 사용량 세이브.
            with torch.no_grad():
                # 테스트 데이터셋을 배치 단위로 반복한다.
                for x_val, y_val in test_loader:
                    # 모델에 입력 데이터를 넣어 예측값 생성 (추론만 수행)
                    val_output = self.model(x_val)

                    # 예측값과 실제값(y_val) 간의 손실 계산
                    val_loss = self.criterion(val_output, y_val)

                    # 손실 값을 파이썬 숫자(float)로 변환하여 리스트에 저장
                    # .item()은 텐서에서 실제 숫자 값만 꺼내 float으로 반환한다.
                    val_losses.append(val_loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time_elapsed": time.time() - start_time
            }
            
            training_history.append(epoch_info)
            
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early Stopping 검증 손실이 개선되지 않으면 카운터 증가
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= self.patience:
                    print("Early stopping triggered.")
                    break
        
        # 훈련 기록 저장
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            # w 모드로 열면 기존 파일이 덮어씌워진다.
            with open(os.path.join(log_dir, "training_history.json"), "w") as f:
                # 두번째 인자로 파이썬 객체를 넣어주면, json으로 변환하여 저장한다.
                json.dump(training_history, f)
        
        return self.model, training_history
    
    def predict(self, x_test_tensor):
        """실제 예측 결과가 필요할떄."""
        self.model.eval()
        # 학습이 끝난 모델으로 한번에 데이터를 전달해도 된다.
        with torch.no_grad():
            # 시각화 라이브러리들은 보통 넘파이 기반이다. 데이터 프레임 변환후 csv저장시에도 필요.
            # 직접 numpy는 cpu위의 텐서에만 가능하며, gpu 위의 텐서는 cpu로 이동후 변환해야한다(.cpu())
            predictions = self.model(x_test_tensor).numpy()
        return predictions