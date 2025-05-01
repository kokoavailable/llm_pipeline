import sys
import os

# [중요] src를 모듈 탐색 경로에 추가
sys.path.append(os.path.abspath("src"))

from pipelines.news_pipeline import run_news_pipeline
from pipelines.stock_pipeline import run_stock_pipeline

if __name__ == "__main__":
    print("📰 뉴스 파이프라인 시작")
    # run_news_pipeline(config_path="config/crawler_config.DEV.yaml")

    print("📈 주식 파이프라인 시작")
    run_stock_pipeline(config_path="config/model_config.yaml")