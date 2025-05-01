"""
시각화 모듈
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import re

from utils.helpers import logger

import matplotlib.font_manager as fm

class NewsVisualizer:
    """뉴스 데이터 시각화 클래스"""
    
    def __init__(self):
        """시각화 도구 초기화"""
        # 한글 폰트 설정 시도
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.PLT_FONT_PATH = os.path.join(self.BASE_DIR, 'fonts', 'NanumGothic-Bold.ttf')
        self.CLOUD_FONT_PATH = os.path.join(self.BASE_DIR, 'fonts', 'NanumGothic-Bold.ttf')

        logger.info(f"폰트 파일 경로: {self.PLT_FONT_PATH}")

        try:
            if not os.path.exists(self.PLT_FONT_PATH):
                logger.warning(f"폰트 파일을 찾을 수 없습니다: {self.PLT_FONT_PATH}")
                raise FileNotFoundError(f"Font file not found: {self.PLT_FONT_PATH}")
            
            # Register font with matplotlib
            fm.fontManager.addfont(self.PLT_FONT_PATH)
            fm._load_fontmanager(try_read_cache=False)

            # Set as default font
            plt.rcParams['font.family'] = 'NanumGothic'
            plt.rcParams['axes.unicode_minus'] = False

            logger.info(f"폰트 설정 완료: NanumGothic")

            self.font_prop = fm.FontProperties(fname=self.PLT_FONT_PATH)
        except Exception as e:
            logger.warning(f"폰트 설정 실패: {e}")
            try:
                plt.rc('font', family='AppleGothic')  # macOS
                logger.info("AppleGothic으로 대체합니다.")
            except:
                logger.warning("기본 한글 폰트를 찾을 수 없습니다. 직접 설정이 필요할 수 있습니다.")
        
        # 그래프 스타일 설정
        # 하얀 배경에 격자.
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def create_wordcloud(self, keywords_list: List[str], output_path: str = 'wordcloud.png'):
        """
        키워드로부터 워드클라우드 생성
        
        Args:
            keywords_list (List[str]): 키워드 목록
            output_path (str): 저장 경로
        """
        # 키워드 통합
        all_keywords = []
        for keywords in keywords_list:
            if isinstance(keywords, str):
                # 쉼표로 구분된 키워드 분리
                keywords = [k.strip() for k in keywords.split(',')]
                all_keywords.extend(keywords)
                
        # 키워드가 없으면 종료
        if not all_keywords:
            logger.warning("워드클라우드를 생성할 키워드가 없습니다.")
            return None
            
        # 단어 빈도수 계산
        word_counts = Counter(all_keywords)
        
        # 워드클라우드 생성
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            font_path=self.CLOUD_FONT_PATH, # 폰트 경로
            colormap='viridis' # 색상 맵
        ).generate_from_frequencies(word_counts)
        
        # 그림 저장
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"워드클라우드 저장 완료: {output_path}")
        return output_path
        
    def plot_source_distribution(self, df: pd.DataFrame, output_path: str = 'source_dist.png'):
        """
        뉴스 소스별 분포 시각화
        
        Args:
            df (pd.DataFrame): 데이터프레임
            output_path (str): 저장 경로
        """
        if 'source' not in df.columns:
            logger.warning("데이터프레임에 'source' 컬럼이 없습니다.")
            return None

        df['date'] = pd.to_datetime(df['date'])

        # 날짜 범위 가져오기
        start_date = df['date'].min().strftime('%Y-%m-%d')
        end_date = df['date'].max().strftime('%Y-%m-%d')
            
        # 소스별 기사 수 계산
        # 현소스 전부 매일 경제.. 
        source_counts = df['source'].value_counts()
        sns.set_theme(font=self.font_prop.get_name())
        
        # 그래프 그리기
        plt.figure(figsize=(8, 6))
        sns.barplot(x=source_counts.index, y=source_counts.values) # 내부적으로 활성화된 axes 에 그림
        plt.title(f'뉴스 소스별 기사 수 ({start_date} ~ {end_date})', fontsize=15)
        plt.xlabel('뉴스 소스', fontsize=12)
        plt.ylabel('기사 수', fontsize=12)
        plt.xticks(rotation=45)
        
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.tight_layout() # 라벨 회전의 경우 써주는게 좋음
        plt.savefig(output_path, dpi=300) # 인쇄 가능한 수준의 퀄리티
        plt.close()
        
        logger.info(f"소스별 분포 그래프 저장 완료: {output_path}")
        return output_path
    
    def plot_monthly_articles(self, df: pd.DataFrame, output_path: str = 'daily_articles.png'):
        """
        월별 기사 수 추이 시각화
        
        Args:
            df (pd.DataFrame): 데이터프레임
            output_path (str): 저장 경로
        """
        if 'date' not in df.columns:
            logger.warning("데이터프레임에 'date' 컬럼이 없습니다.")
            return None
            
        # 날짜 형식 변환
        df['date'] = pd.to_datetime(df['date'])

        # 월 단위로 자르기
        df['month'] = df['date'].dt.to_period('M')

        # 전체 월 범위 생성
        full_month_range = pd.period_range(start=df['month'].min(), end=df['month'].max(), freq='M')
        
        # 월별 기사 수 계산
        monthly_counts = df.groupby('month').size()
        monthly_counts = monthly_counts.reindex(full_month_range, fill_value=0)

        # 인덱스를 datetime으로 변환 (플로팅용)
        monthly_counts.index = monthly_counts.index.strftime('%Y-%m')
        
        # 그래프 그리기
        plt.figure(figsize=(12, 6))
        ax = monthly_counts.plot(kind='bar')
        plt.title('월별 KDX 관련 기사 수', fontsize=15)
        plt.xlabel('날짜', fontsize=12)
        plt.ylabel('기사 수', fontsize=12)
        plt.grid(True, alpha=0.3) # 격자선 투명도 
        
        # x축 날짜 포맷 설정
        plt.gcf().autofmt_xdate()
        
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"일별 기사 수 그래프 저장 완료: {output_path}")
        return output_path
        
    def generate_all_visualizations(self, df: pd.DataFrame, output_dir: str = 'data/visualizations'):
        """
        모든 시각화 생성
        
        Args:
            df (pd.DataFrame): 데이터프레임
            output_dir (str): 출력 디렉토리
            
        Returns:
            dict: 생성된 시각화 파일 경로 목록
        """
        # 디렉토리가 없으면 생성
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # 워드클라우드 생성
        if 'keywords' in df.columns:
            wordcloud_path = os.path.join(output_dir, 'keywords_wordcloud.png')
            results['wordcloud'] = self.create_wordcloud(df['keywords'], wordcloud_path)
            
        # 소스별 분포
        if 'source' in df.columns:
            source_dist_path = os.path.join(output_dir, 'source_distribution.png')
            results['source_dist'] = self.plot_source_distribution(df, source_dist_path)
            
        # 일별 기사 수
        if 'date' in df.columns:
            daily_path = os.path.join(output_dir, 'monthly_articles.png')
            results['monthly_articles'] = self.plot_monthly_articles(df, daily_path)
            
        logger.info(f"모든 시각화 생성 완료: {len(results)}개")
        return results
        
    def run(self, input_path='data/summaries/news_summaries.csv', output_dir='data/visualizations'):
        """
        시각화 프로세스 실행
        
        Args:
            input_path (str): 입력 파일 경로
            output_dir (str): 출력 디렉토리
            
        Returns:
            dict: 생성된 시각화 파일 경로 목록
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
            
        # 모든 시각화 생성
        return self.generate_all_visualizations(df, output_dir)
        
