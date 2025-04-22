import re
import logging
from bs4 import BeautifulSoup
import pandas as pd
import os
class NewsPreprocessor:
    """크롤링한 raw 뉴스 데이터를 전처리하는 클래스"""
    def __init__(self):
        """뉴스 전처리 클래스 초기화"""
        pass

    def clean_text(self, text):
        """
        텍스트 정제
        
        Args:
            text (str): 원본 텍스트
            
        Returns:
            str: 정제된 텍스트
        """
        if not isinstance(text, str):
            return ""
            
        # HTML 태그 제거
        text = re.sub(r'<.*?>', '', text)
        
        # 특수문자 처리
        text = re.sub(r'[^\w\s\.]', ' ', text)
        
        # 여러 개의 공백을 하나로 통합
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text

    def process_dataframe(self, df):
        """
        데이터프레임 전처리
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 전처리된 데이터프레임
        """
        logger.info("데이터프레임 전처리 시작")
        
        # 복사본 생성
        processed_df = df.copy()
        
        # 중복 제거
        processed_df.drop_duplicates(subset=['title', 'content'], inplace=True)
        
        # 결측치 처리
        processed_df['title'] = processed_df['title'].fillna('')
        processed_df['content'] = processed_df['content'].fillna('')
        
        # 텍스트 정제
        processed_df['title_clean'] = processed_df['title'].apply(self.clean_text)
        processed_df['content_clean'] = processed_df['content'].apply(self.clean_text)
        
        # 길이가 너무 짧은 기사 필터링
        processed_df = processed_df[processed_df['content_clean'].str.len() > 50]
        
        # 날짜 형식 통일
        try:
            processed_df['date'] = pd.to_datetime(processed_df['date']).dt.strftime('%Y-%m-%d')
        except:
            logger.warning("날짜 형식 변환 오류, 원본 유지")
        
        # 고유 ID 생성
        processed_df['article_id'] = [
            f"{src}_{i}" for i, src in enumerate(processed_df['source'])
        ]
        
        logger.info(f"전처리 완료: {len(processed_df)}개 기사")
        return processed_df
    
    def process_file(self, input_path, output_path):
        """
        파일 처리
        
        Args:
            input_path (str): 입력 파일 경로
            output_path (str): 출력 파일 경로
            
        Returns:
            pd.DataFrame: 전처리된 데이터프레임
        """
        # 파일 존재 확인
        if not os.path.exists(input_path):
            logger.error(f"파일을 찾을 수 없음: {input_path}")
            return None
            
        # 파일 형식에 따라 읽기
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.json'):
            df = pd.read_json(input_path)
        else:
            logger.error(f"지원하지 않는 파일 형식: {input_path}")
            return None
            
        # 전처리
        processed_df = self.process_dataframe(df)
        
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 저장
        if output_path.endswith('.csv'):
            processed_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        elif output_path.endswith('.json'):
            processed_df.to_json(output_path, orient='records', force_ascii=False, indent=4)
        
        logger.info(f"전처리된 데이터 저장 완료: {output_path}")
        return processed_df
    
    def run(self, input_path='data/raw/news_articles.csv', output_path='data/processed/processed_news.csv'):
        """
        전처리 실행
        
        Args:
            input_path (str): 입력 파일 경로
            output_path (str): 출력 파일 경로
            
        Returns:
            pd.DataFrame: 전처리된 데이터프레임
        """
        return self.process_file(input_path, output_path)