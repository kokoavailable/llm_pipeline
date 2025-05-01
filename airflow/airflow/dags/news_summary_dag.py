"""
KDX 뉴스 요약 Airflow DAG
"""
from datetime import datetime, timedelta
import os
import sys

# 프로젝트 루트 경로 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# 프로젝트 모듈 임포트
from data.news.crawler import NewsCrawler
from data.news.preprocessor import NewsPreprocessor
from llm.gpt_summarizer import GPTSummarizer
from visualization.news.plotter import NewsVisualizer

# 기본 인수 설정
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG 정의
dag = DAG(
    'kdx_news_summary_pipeline',
    default_args=default_args,
    description='KDX 뉴스 기사 수집 및 요약 파이프라인',
    schedule_interval=timedelta(days=1),  # 매일 실행
    start_date=days_ago(1),
    tags=['kdx', 'news', 'llm', 'gpt'],
)

# 실행 날짜 기반 출력 경로 생성
def get_output_paths(**kwargs):
    """실행 날짜 기반 출력 경로 생성"""
    execution_date = kwargs['execution_date']
    date_str = execution_date.strftime('%Y%m%d')
    
    # 기본 경로
    base_dir = os.path.join(PROJECT_ROOT, 'data', 'airflow_runs', date_str)
    
    # 경로 정의
    paths = {
        'raw_data': os.path.join(base_dir, 'raw_news.csv'),
        'processed_data': os.path.join(base_dir, 'processed_news.csv'),
        'summary_data': os.path.join(base_dir, 'news_summaries.csv'),
        'viz_dir': os.path.join(base_dir, 'visualizations')
    }
    
    # 디렉토리 생성
    for path in paths.values():
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
    
    return paths

# Task 1: 경로 설정
get_paths_task = PythonOperator(
    task_id='get_paths',
    python_callable=get_output_paths,
    provide_context=True,
    dag=dag,
)

# Task 2: 뉴스 데이터 크롤링
def crawl_news(**kwargs):
    """뉴스 데이터 크롤링"""
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='get_paths')
    
    crawler = NewsCrawler()
    df = crawler.run(paths['raw_data'])
    return len(df)

crawl_task = PythonOperator(
    task_id='crawl_news',
    python_callable=crawl_news,
    provide_context=True,
    dag=dag,
)

# Task 3: 데이터 전처리
def preprocess_data(**kwargs):
    """데이터 전처리"""
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='get_paths')
    
    preprocessor = NewsPreprocessor()
    df = preprocessor.run(paths['raw_data'], paths['processed_data'])
    return len(df)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag,
)

# Task 4: 요약 생성
def generate_summaries(**kwargs):
    """요약 생성"""
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='get_paths')
    
    summarizer = GPTSummarizer()
    df = summarizer.run(paths['processed_data'], paths['summary_data'])
    return len(df)

summarize_task = PythonOperator(
    task_id='generate_summaries',
    python_callable=generate_summaries,
    provide_context=True,
    dag=dag,
)

# Task 5: 시각화 생성
def generate_visualizations(**kwargs):
    """시각화 생성"""
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='get_paths')
    
    visualizer = NewsVisualizer()
    results = visualizer.run(paths['summary_data'], paths['viz_dir'])
    return list(results.keys())

visualize_task = PythonOperator(
    task_id='generate_visualizations',
    python_callable=generate_visualizations,
    provide_context=True,
    dag=dag,
)

# Task 6: 결과 알림
def notify_completion(**kwargs):
    """작업 완료 알림"""
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='get_paths')
    article_count = ti.xcom_pull(task_ids='crawl_news')
    viz_results = ti.xcom_pull(task_ids='generate_visualizations')
    
    message = f"""
    KDX 뉴스 요약 파이프라인 완료!
    
    실행 날짜: {kwargs['execution_date']}
    기사 수: {article_count}
    결과 위치: {paths['summary_data']}
    생성된 시각화: {', '.join(viz_results)}
    """
    
    print(message)
    # 여기에 이메일 또는 슬랙 알림 등을 추가할 수 있음
    
    return message

notify_task = PythonOperator(
    task_id='notify_completion',
    python_callable=notify_completion,
    provide_context=True,
    dag=dag,
)

# 작업 순서 설정
get_paths_task >> crawl_task >> preprocess_task >> summarize_task >> visualize_task >> notify_task

globals()['dag'] = dag