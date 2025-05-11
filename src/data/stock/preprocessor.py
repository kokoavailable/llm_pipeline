import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import yaml
import joblib

class StockDataPreprocessor:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # 사용할 특성 칼럼
        self.feature_cols = self.config["model"]["features"]
        # 예측 대상
        self.label_cols = self.config["model"]["labels"]
        # 시퀀스 생성시 사용할 윈도우 크기(10일 -> 1일)
        self.window_size = self.config["model"]["window_size"]
        # 학습/ 테스트 데이터 나누는 위치
        self.split_point = self.config["model"]["split_point"]
        
    def preprocess_data(self, raw_df):
        """데이터 전처리를 수행합니다."""
        # 거래량이 0인 데이터는 오류일 수 있으므로 0 처리한다.
        raw_df['Volume'] = raw_df['Volume'].replace(0, np.nan)
        
        # 각 컬럼별로 0 값이 얼마나 있는지 출력한다.
        for col in raw_df.columns:
            # 특정칼럼이 0인지를 반환하는 불리안 시리즈를 조건식으로 사용하여 미싱로우 필터링
            missing_rows = raw_df.loc[raw_df[col]==0].shape[0]
            print(f"{col}: {missing_rows}")
        
        # NaN 값이 있는 행 제거
        raw_df = raw_df.dropna()
        
        # 스케일링 적용 칼럼 지정
        scale_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', '3MA', '5MA', 'Volume']
        # 값의 비율에 따라, 0~ 1 사이로 스케일링하는 객체 생성.
        scaler = MinMaxScaler(feature_range=(0, 1))
        # 스케일링 객체를 사용해 지정된 컬럼만 꺼내서 스케일리ng을 적용한다. 결과는 넘파이 배열이된다.
        scaled_df = scaler.fit_transform(raw_df[scale_cols])
        # 다시 판다스 데이터 프레임으로 바꿔주며 원래 컬럼명을 붙인다.
        scaled_df = pd.DataFrame(scaled_df, columns=scale_cols)
        
        return scaled_df, scaler
    
    def prepare_features_labels(self, scaled_df):
        """특성과 레이블을 준비합니다."""
        # scaled_df에서 예측할 대상에 해당하는 컬럼들만 따로 추출하여 데이터 프레임으로 만든다.
        label_df = pd.DataFrame(scaled_df, columns=self.label_cols)
        # scaled_df에서 예측할 대상에 해당하지 않는 컬럼들만 따로 추출하여 데이터 프레임으로 만든다.
        feature_df = pd.DataFrame(scaled_df, columns=self.feature_cols)
        
        # 레이블을 넘파이 배열로 만든다. (모델 학습에 맞는 형태)
        label_np = label_df.to_numpy()
        feature_np = feature_df.to_numpy()
        
        return feature_np, label_np
    
    def make_sequence_dataset(self, feature, label):
        """시퀀스 데이터셋을 생성합니다."""
        # 시퀀스 데이터 저장 리스트
        feature_list = []
        label_list = []
        
        # 전체 데이터에서 window_size 만큼의 시퀀스를 반복하면서 리스트로 만든다.
        for i in range(len(feature) - self.window_size):
            # 전체 데이터에서 윈도우 사이즈만큼 잘라내기 반복.
            feature_list.append(feature[i:i + self.window_size])
            label_list.append(label[i + self.window_size])
        
        # 릿스트를 넘파이 배열로 변환한다.
        # shape 0차원. 샘플 수, 1차원. 윈도우 사이즈, 2차원. 특성 수
        return np.array(feature_list), np.array(label_list)
    
    def split_train_test(self, X, Y):
        """
        훈련 및 테스트 데이터를 분할한다.
        트레인 데이터는 직접 학습을 하는데 사용되며,
        나머지 테스트 데이터로는 학습으로 얻은 모델을 평가하는데 사용된다.
        """

        # 설정된 split_point 이전 까지를 훈련용 데이터로 사용한다.
        x_train = X[:self.split_point]
        y_train = Y[:self.split_point]

        # 이후를 테스트용 데이터로 사용한다.
        x_test = X[self.split_point:]
        y_test = Y[self.split_point:]
        
        # 훈련용/ 테스트용 데이터를 반환한다.
        return x_train, y_train, x_test, y_test
    
    def save_processed_data(self, x_train, y_train, x_test, y_test, scaler, output_dir):
        """
        전처리된 데이터를 저장합니다.

        Args:
            x_train (ndarray): 훈련용 입력 데이터
            y_train (ndarray): 훈련용 정답 레이블
            x_test (ndarray): 테스트용 입력 데이터
            y_test (ndarray): 테스트용 정답 레이블
            scaler (object): 데이터 정규화를 위해 사용한 스케일러 객체 (예: StandardScaler)
            output_dir (str): 저장할 디렉토리 경로

        Returns:
            str: 저장된 디렉토리 경로
        
        """
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, "x_train.npy"), x_train)
        np.save(os.path.join(output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(output_dir, "x_test.npy"), x_test)
        np.save(os.path.join(output_dir, "y_test.npy"), y_test)
        
        # 스케일러 저장
        joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
        
        return output_dir