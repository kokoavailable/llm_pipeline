"""
GPT 기반 텍스트 요약 모듈
"""
import os
import time
import json
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
import openai
import textwrap
from dotenv import load_dotenv

from ..utils.helpers import logger, api_key
class GPTSummarizer:
    """GPT를 사용한 텍스트 요약 클래스"""

    def __init__(self, model="gpt-4o-mini"):
        """
        요약기 초기화

        Args:
            model (str): 사용할 OpenAI 모델
        """
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)

        if not self.client:
            logger.warning("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")



    def summarize_text(self, text: str, max_retries: int = 3) -> str:
        """
        텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            max_retries (int): 최대 재시도 횟수

        Returns:
            str: 요약된 텍스트    
        """

        if not text or len(text.strip()) < 10:
            return ""
        
        # 요약 요청을 위한 시스템 프롬프트
        system_prompt = textwrap.dedent(f"""
        당신은 뉴스 기사를 읽고, 핵심 내용을 부드럽고 자연스러운 단락으로 요약하는 AI입니다.
        요약은 객관적이며 불렛포인트 없이, 사람이 쓴 것처럼 자연스럽게 이어지는 3~4문장으로 작성되어야 합니다.
        """).strip()
        
        # 요약 요청을 위한 사용자 프롬프트
        user_prompt = textwrap.dedent(f"""
        다음은 뉴스 기사입니다. 중요한 내용을 중심으로, 자연스럽고 흐름이 끊기지 않는 단락으로 3~4문장 정도로 요약해주세요. 요약은 마치 사람이 쓴 것처럼 자연스럽게 이어지고, 불렛포인트나 나열식 표현 없이 작성되어야 합니다.

        기사 원문:
        {text}

        요약:
        """).strip()

        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7,
                    n=1,
                    stop=None
                )

                summary = response.choices[0].message.content.strip()
                return summary
            
            except Exception as e:
                logger.error(f"요약 생성 중 오류 발생 (시도 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 지수 백오프
                    logger.info(f"{wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    return f"요약 생성 실패: {str(e)}"
                
    def extract_keywords(self, text: str, max_retries: int = 3) -> List[str]:
        """
        텍스트에서 키워드 추출
        Args:
            text (str): 키워드를 추출할 텍스트
            max_retries (int): 최대 재시도 횟수
        Returns:
            List[str]: 추출된 키워드 목록
        """

        if not text or len(text.strip()) < 10:
            return []
        
        system_prompt = textwrap.dedent(f"""당신은 뉴스 기사를 읽고, 핵심 키워드를 추출하는 AI입니다.""").strip()
        
        user_prompt = textwrap.dedent(f"""다음은 뉴스 기사입니다. 이 기사에서 가장 중요한 핵심 키워드 5개를 추출해주세요.  
        - 각 키워드는 단어 또는 짧은 구문 형태여야 합니다.  
        - 키워드 간에는 쉼표(,)로 구분해주세요.  
        - 불렛포인트나 숫자 없이, 키워드만 한 줄로 출력해주세요.  
        - 문장은 작성하지 마세요.  

        기사 내용:  
        {text}

        결과:
        """).strip()

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=100
                )

                keywords_text = response.choices[0].message.content.strip()

                keywords = [k.strip() for k in keywords_text.split(',')]
                return keywords
            
            except Exception as e:
                logger.error(f"키워드 추출 중 오류 발생 (시도 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"{wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    return []
                
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame의 각 기사에 대해 요약 및 키워드 추출 수행

        Args:
            df (pd.DataFrame): 기사 데이터가 포함된 DataFrame

        Returns:
            pd.DataFrame: 요약 및 키워드가 추가된 DataFrame
        """
        df['summary'] = ""
        df['keywords'] = ""

        for idx, row in df.iterrows():
            logger.info(f"진행 중: {idx}/{len(df)} 기사 처리 완료")

            content = row.get('content_clean', row.get('content', ''))

            # 요약 생성
            summary = self.summarize_text(content)
            df.at[idx, 'summary'] = summary

            # 키워드 추출
            keywords = self.extract_keywords(content)
            df.at[idx, 'keywords'] = ', '.join(keywords)

            # API 호출 간격 설정
            time.sleep(1)

        logger.info("요약 및 키워드 추출 완료")
        return df
    
    def process_file(self, input_path: str, output_path: str) -> pd.DataFrame:
        """
        파일 기반 처리

        Args:
            input_path (str): 입력 파일 경로
            output_path (str): 출력 파일 경로

        Returns:
            pd.DataFrame: 처리된 DataFrame
        """

        if not os.path.exists(input_path):
            logger.error(f"파일을 찾을 수 없음: {input_path}")
            return None
        
        df = pd.read_csv(input_path)
        
        # 요약 및 키워드 추출
        result_df = self.process_dataframe(df)

        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)


        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        json_path = output_path.replace('.csv', '.json')

        result_df.to_json(json_path, orient='records', force_ascii=False, indent=4)

        logger.info(f"요약 결과 저장 완료: {output_path}")
        return result_df
        
    def run(self, input_path='data/processed/processed_news.csv', output_path='data/summaries/news_summaries.csv'):
        """
        요약 및 키워드 추출 실행

        Args:
            input_path (str): 입력 파일 경로
            output_path (str): 출력 파일 경로

        Returns:
            pd.DataFrame: 요약 및 키워드가 추가된 DataFrame
        """
        return self.process_file(input_path, output_path)