"""
전체 파이프라인 실행 모듈
"""
import os
import logging
import argparse
from datetime import datetime

from data.crawler import NewsCrawler
from data.preprocessor import NewsPreprocessor
from models.summarizer import GPTSummarizer
from visualization.plotter import NewsVisualizer
from utils.helpers import ensure_directory, get_timestamp, save_results, logger


def run_pipeline(config_path=None, output_dir=None):
    """
    전체 파이프라인 실행
    
    Args:
        config_path (str): 설정 파일 경로
        output_dir (str): 출력 디렉토리
        
    Returns:
        dict: 파이프라인 실행 결과
    """
    start_time = datetime.now()
    logger.info(f"파이프라인 실행 시작: {start_time}")
    
    # 타임스탬프
    timestamp = get_timestamp()
    
    # 출력 디렉토리 설정
    if not output_dir:
        output_dir = f"data/runs/{timestamp}"
    
    # 디렉토리 생성
    ensure_directory(output_dir)
    ensure_directory("logs")
    
    # 파일 경로 설정
    raw_data_path = os.path.join(output_dir, "raw_news.csv")
    processed_data_path = os.path.join(output_dir, "processed_news.csv")
    summary_data_path = os.path.join(output_dir, "news_summaries.csv")
    viz_output_dir = os.path.join(output_dir, "visualizations")
    
    results = {
        "timestamp": timestamp,
        "output_directory": output_dir,
        "files": {}
    }
    
    try:
        # 1. 데이터 수집
        logger.info("1단계: 뉴스 데이터 크롤링 중...")
        crawler = NewsCrawler(config_path)
        raw_df = crawler.run(raw_data_path)
        results["files"]["raw_data"] = raw_data_path
        results["article_count"] = len(raw_df)
        logger.info(f"  → 크롤링 완료: {len(raw_df)}개 기사")
        
        # 2. 데이터 전처리
        logger.info("2단계: 데이터 전처리 중...")
        preprocessor = NewsPreprocessor()
        processed_df = preprocessor.run(raw_data_path, processed_data_path)
        results["files"]["processed_data"] = processed_data_path
        results["processed_article_count"] = len(processed_df)
        logger.info(f"  → 전처리 완료: {len(processed_df)}개 기사")
        
        # 3. 요약 생성
        logger.info("3단계: GPT 요약 생성 중...")
        summarizer = GPTSummarizer()
        summary_df = summarizer.run(processed_data_path, summary_data_path)
        results["files"]["summary_data"] = summary_data_path
        logger.info("  → 요약 생성 완료")
        
        # 4. 시각화 생성
        logger.info("4단계: 시각화 생성 중...")
        visualizer = NewsVisualizer()
        viz_results = visualizer.run(summary_data_path, viz_output_dir)
        results["files"]["visualizations"] = viz_results
        logger.info("  → 시각화 생성 완료")
        
        # 결과 보고서 저장
        report_path = os.path.join(output_dir, "pipeline_results.json")
        save_results(results, report_path)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        results["execution_time_seconds"] = execution_time
        logger.info(f"파이프라인 실행 완료! 총 소요 시간: {execution_time:.2f}초")
        
        return results
        
    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        results["error"] = str(e)
        report_path = os.path.join(output_dir, "pipeline_error.json")
        save_results(results, report_path)
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KDX 뉴스 요약 파이프라인 실행')
    parser.add_argument('--config', type=str, default='config/crawler_config.yaml', help='크롤러 설정 파일 경로')
    parser.add_argument('--output', type=str, default=None, help='출력 디렉토리 경로')
    
    args = parser.parse_args()
    run_pipeline(args.config, args.output)