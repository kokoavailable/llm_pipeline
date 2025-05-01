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
            
        self.feature_cols = self.config["model"]["features"]
        self.label_cols = self.config["model"]["labels"]
        self.window_size = self.config["model"]["window_size"]
        self.split_point = self.config["model"]["split_point"]
        
    def preprocess_data(self, raw_df):
        """데이터 전처리를 수행합니다."""
        # 결측치 처리
        raw_df['Volume'] = raw_df['Volume'].replace(0, np.nan)
        
        for col in raw_df.columns:
            missing_rows = raw_df.loc[raw_df[col]==0].shape[0]
            print(f"{col}: {missing_rows}")
        
        # NaN 값이 있는 행 제거
        raw_df = raw_df.dropna()
        
        # 스케일링
        scale_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', '3MA', '5MA', 'Volume']
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_df = scaler.fit_transform(raw_df[scale_cols])
        scaled_df = pd.DataFrame(scaled_df, columns=scale_cols)
        
        return scaled_df, scaler
    
    def prepare_features_labels(self, scaled_df):
        """특성과 레이블을 준비합니다."""
        label_df = pd.DataFrame(scaled_df, columns=self.label_cols)
        feature_df = pd.DataFrame(scaled_df, columns=self.feature_cols)
        
        label_np = label_df.to_numpy()
        feature_np = feature_df.to_numpy()
        
        return feature_np, label_np
    
    def make_sequence_dataset(self, feature, label):
        """시퀀스 데이터셋을 생성합니다."""
        feature_list = []
        label_list = []
        
        for i in range(len(feature) - self.window_size):
            feature_list.append(feature[i:i + self.window_size])
            label_list.append(label[i + self.window_size])
        
        return np.array(feature_list), np.array(label_list)
    
    def split_train_test(self, X, Y):
        """훈련 및 테스트 데이터를 분할합니다."""
        x_train = X[:self.split_point]
        y_train = Y[:self.split_point]
        x_test = X[self.split_point:]
        y_test = Y[self.split_point:]
        
        return x_train, y_train, x_test, y_test
    
    def save_processed_data(self, x_train, y_train, x_test, y_test, scaler, output_dir):
        """전처리된 데이터를 저장합니다."""
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, "x_train.npy"), x_train)
        np.save(os.path.join(output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(output_dir, "x_test.npy"), x_test)
        np.save(os.path.join(output_dir, "y_test.npy"), y_test)
        
        # 스케일러 저장
        joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
        
        return output_dir