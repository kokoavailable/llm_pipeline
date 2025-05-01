import os
import argparse
import yaml
import torch
import numpy as np

from data.stock.loader import StockDataLoader
from data.stock.preprocessor import StockDataPreprocessor
from models.gru_model import GRUModel
from models.trainer import ModelTrainer
from visualization.stock.plotter import StockVisualizer

def run_stock_pipeline(config_path, output_dir):
    """전체 주식 예측 파이프라인을 실행합니다."""
    # 설정 로드
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 디렉토리 설정
    raw_data_dir = os.path.join(output_dir, "data/raw")
    processed_data_dir = os.path.join(output_dir, "data/processed")
    model_dir = os.path.join(output_dir, "models")
    logs_dir = os.path.join(output_dir, "logs")
    viz_dir = os.path.join(output_dir, "data/visualizations")
    
    for directory in [raw_data_dir, processed_data_dir, model_dir, logs_dir, viz_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # 1. 데이터 로드
    print("Loading stock data...")
    loader = StockDataLoader(config_path)
    raw_df = loader.load_data()
    loader.save_raw_data(raw_df, raw_data_dir)
    
    # 2. 데이터 시각화
    print("Visualizing raw data...")
    visualizer = StockVisualizer(viz_dir)
    visualizer.plot_stock_history(raw_df, 'Adj Close', f"{config['model']['ticker']} Stock Price History")
    
    # 3. 데이터 전처리
    print("Preprocessing data...")
    preprocessor = StockDataPreprocessor(config_path)
    scaled_df, scaler = preprocessor.preprocess_data(raw_df)
    feature_np, label_np = preprocessor.prepare_features_labels(scaled_df)
    
    # 4. 시퀀스 데이터셋 생성
    print("Creating sequence dataset...")
    X, Y = preprocessor.make_sequence_dataset(feature_np, label_np)
    
    # 5. 훈련/테스트 데이터 분할
    print("Splitting data...")
    x_train, y_train, x_test, y_test = preprocessor.split_train_test(X, Y)
    
    # 6. 처리된 데이터 저장
    preprocessor.save_processed_data(x_train, y_train, x_test, y_test, scaler, processed_data_dir)
    
    # 7. 모델 초기화
    print("Initializing model...")
    model = GRUModel(config_path)
    
    # 8. 훈련기 초기화 및 데이터 로더 준비
    trainer = ModelTrainer(model, config_path)
    train_loader, test_loader = trainer.prepare_data_loaders(x_train, y_train, x_test, y_test)
    
    # 9. 모델 훈련
    print("Training model...")
    trained_model, history = trainer.train(train_loader, test_loader, logs_dir)
    
    # 10. 모델 저장
    print("Saving model...")
    model_path = os.path.join(model_dir, "gru_model.pth")
    trained_model.save_model(model_path)
    
    # 11. 예측 및 시각화
    print("Making predictions and visualizing results...")
    x_test_tensor = torch.FloatTensor(x_test)
    predictions = trainer.predict(x_test_tensor)
    
    visualizer.plot_prediction_results(
        y_test, 
        predictions, 
        config["model"]["window_size"],
        config["model"]["features"]
    )
    
    # 12. 훈련 기록 시각화
    visualizer.plot_training_history(history)
    
    print("Pipeline completed successfully!")
    return {
        "model_path": model_path,
        "data_paths": {
            "raw": raw_data_dir,
            "processed": processed_data_dir
        },
        "visualizations": viz_dir
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Price Prediction Pipeline")
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    
    args = parser.parse_args()
    
    run_pipeline(args.config, args.output)