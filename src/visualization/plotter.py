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

# # 한글 폰트 설정 (필요시)
# import matplotlib.font_manager as fm
# # 예시: font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
# # plt.rc('font', family='NanumGothic')


class NewsVisualizer:
    """뉴스 데이터 시각화 클래스"""
    
    def __init__(self):
        """시각화 도구 초기화"""
        # 한글 폰트 설정 시도
        try:
            plt.rc('font', family='Malgun Gothic')  # Windows
        except:
            try:
                plt.rc('font', family='AppleGothic')  # macOS
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
            font_path=fm.findfont(fm.FontProperties(family='Malgun Gothic')),
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
            
        # 소스별 기사 수 계산
        # 현소스 전부 매일 경제.. 
        source_counts = df['source'].value_counts()
        
        # 그래프 그리기
        plt.figure(figsize=(8, 6))
        sns.barplot(x=source_counts.index, y=source_counts.values) # 내부적으로 활성화된 axes 에 그림
        plt.title('뉴스 소스별 기사 수', fontsize=15)
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
    
    def plot_daily_articles(self, df: pd.DataFrame, output_path: str = 'daily_articles.png'):
        """
        일별 기사 수 추이 시각화
        
        Args:
            df (pd.DataFrame): 데이터프레임
            output_path (str): 저장 경로
        """
        if 'date' not in df.columns:
            logger.warning("데이터프레임에 'date' 컬럼이 없습니다.")
            return None
            
        # 날짜 형식 변환
        df['date'] = pd.to_datetime(df['date'])
        
        # 일별 기사 수 계산
        daily_counts = df.groupby(df['date'].dt.date).size()
        
        # 그래프 그리기
        plt.figure(figsize=(12, 6))
        ax = daily_counts.plot(kind='line', marker='o')
        plt.title('일별 KDX 관련 기사 수', fontsize=15)
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
            daily_path = os.path.join(output_dir, 'daily_articles.png')
            results['daily_articles'] = self.plot_daily_articles(df, daily_path)
            
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
        
