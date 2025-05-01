from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append('/path/to/your/project')  # 실제 프로젝트 경로로 변경하세요

from src.pipeline import run_pipeline

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 4, 27),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_prediction_pipeline',
    default_args=default_args,
    description='A DAG to run stock prediction pipeline',
    schedule_interval=timedelta(days=1),
)

def run_stock_prediction(**kwargs):
    config_path = kwargs.get('config_path', 'config/model_config.yaml')
    output_dir = kwargs.get('output_dir', '.')
    
    return run_pipeline(config_path, output_dir)

run_task = PythonOperator(
    task_id='run_stock_prediction',
    python_callable=run_stock_prediction,
    op_kwargs={
        'config_path': 'config/model_config.yaml',
        'output_dir': '.'
    },
    dag=dag,
)

run_task