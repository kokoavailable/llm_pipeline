"""
유틸리티 헬퍼 함수
"""
import os
import json
import yaml
import logging
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any, Optional




def setup_logging():
    """
    로깅 설정 함수
    파일 핸들러와 로그 핸들러를 사용합니다.
    """
    _logger = logging.getLogger('main')

    # 로그 레벨 설정
    _logger.setLevel(logging.DEBUG)  # 개별 로거 레벨 설정

    # 로그 파일 경로 설정
    log_file_path = os.path.join(
        '..', 
        'logs', 
        f"{app_env}_{datetime.now().strftime('%Y-%m-%d')}.log"
    )

    # 로그 디렉토리 생성
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)  # 파일에는 INFO 이상의 로그만 저장
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    _logger.addHandler(file_handler)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # 콘솔 로그 레벨
    console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console_handler.setFormatter(console_formatter)
    logging.getLogger('').addHandler(console_handler)

    return _logger

# 로그 설정 함수 호출 싱글톤
logger = setup_logging()

def ensure_directory(directory_path: str) -> bool:
    """
    디렉토리 존재 확인 및 생성

    Args:
        directory_path (str): 확인 및 생성할 디렉토리 경로

    Returns:
        bool: 성공 여부

    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"디렉토리 생성 실패 {directory_path}: {e}")
        return False
    
def load_config(config_path: str) -> Dict[str, Any]:
    """
    YAML 형식의 설정 파일 로드

    Args:
        config_path (str): 설정 파일 경로

    Returns:
        dict: 설정 내용
    """
    if not os.path.exists(config_path):
        logger.error(f"설정 파일을 찾을 수 없음: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            elif config_path.endswith('.json'):
                return json.load(f)
            else:
                logger.error(f"지원하지 않는 설정 파일 형식: {config_path}")
                return {}
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {e}")
        return {}
    
def save_results(results: Dict, output_path: str) -> bool:
    """
    결과 저장
    
    Args:
        results (Dict): 저장할 결과
        output_path (str): 저장 경로
        
    Returns:
        bool: 성공 여부
    """
    try:
        # 디렉토리 확인
        directory = os.path.dirname(output_path)
        if directory:
            ensure_directory(directory)
            
        # 형식에 따른 저장
        if output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        elif output_path.endswith('.yaml') or output_path.endswith('.yml'):
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
        else:
            # 기본 텍스트 파일
            with open(output_path, 'w', encoding='utf-8') as f:
                for key, value in results.items():
                    f.write(f"{key}: {value}\n")
                    
        logger.info(f"결과 저장 완료: {output_path}")
        return True
    except Exception as e:
        logger.error(f"결과 저장 실패: {e}")
        return False
    
def get_timestamp() -> str:
    """
    현재 타임스탬프 문자열 반환
    
    Returns:
        str: YYYYMMDD_HHMMSS 형식의 타임스탬프
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')