"""
LangChain 기반 텍스트 요약 모듈
"""
import os
import time
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from src.utils.helpers import logger, api_key
from src.llm.base_prompt import BasePromptTemplates

class GPTSummarizer:
    """LangChain을 사용한 텍스트 요약 클래스"""

    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=300):
        """
        요약기 초기화

        Args:
            model_name (str): 사용할 OpenAI 모델
            temperature (float): 생성 다양성 조절 (0~1)
            max_tokens (int): 최대 토큰 수
        """
        # 사용자 입력 또는 기본값으로부터 모델 파라미터 설정한다.
        self.model_name = model_name # 사용할 llm 모델 이름
        self.temperature = temperature # 생성의 무작위 정도를 조절한다. 0.0은 결정론적이고, 1.0은 무작위적이다.
        self.max_tokens = max_tokens # 생성응답으로 허용되는 최대 토큰 수이다.
        
        # LangChain 모델 초기화
        try:
            # 랭체인의 챗 오픈 AI 모델을 사용하여 대화형 모델을 초기화한다.
            self.chat_model = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=api_key
            )
            
            # 요약에 사용할 프롬프트 템플릿을 불러온다.
            # 입력 텍스트를 받아 요약하는 형태의 프롬프트.
            self.summarize_prompt = BasePromptTemplates.get_summarization_template()
            
            # 키워트 추출에 사용할 프롬프트 템플릿을 불러온다.
            # 입력 텍스트를 받아 키워드를 추출하는 형태의 프롬프트.
            self.keyword_prompt = BasePromptTemplates.get_keyword_extraction_template()
            
            # 요약 작업을 위한 llm 체인을 구성한다. 사용할 llm 객체와 프롬프트를 인자로 넣는다.
            self.summarize_chain = LLMChain(llm=self.chat_model, prompt=self.summarize_prompt)
            self.keyword_chain = LLMChain(llm=self.chat_model, prompt=self.keyword_prompt)
            
        except Exception as e:
            logger.error(f"LangChain 모델 초기화 중 오류: {e}")
            raise Exception(f"LangChain 모델 초기화 실패: {e}")

    def summarize_text(self, text: str, max_retries: int = 3) -> str:
        """
        텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            max_retries (int): 최대 재시도 횟수

        Returns:
            str: 요약된 텍스트    
        """
        # 텍스트가 비어 있거나 너무 짧은 경우는 추출의 의미가 없으므로 거른다.
        if not text or len(text.strip()) < 10:
            return ""
        
        for attempt in range(max_retries):
            try:
                # LangChain 체인 실행. 프롬프트에 바로 텍스트를 넣어 LLM으로 실행한다.
                result = self.summarize_chain.run(text=text)

                # 결과 앞뒤 공백 제거후 반환
                return result.strip()
            
            except Exception as e:
                # 예외 발생 시 로깅하고, 재시도
                logger.error(f"요약 생성 중 오류 발생 (시도 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # 실패할때마다 기다리는 시간을 2배 늘려가며 시도하는 전략이다.
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
        # 텍스트가 비어 있거나 너무 짧은 경우는 추출의 의미가 없으므로 거른다.
        if not text or len(text.strip()) < 10:
            return []
        
        for attempt in range(max_retries):
            try:
                # LangChain 체인 키워드 프롬프트를 실행.
                result = self.keyword_chain.run(text=text)
                
                # 결과 처리
                keywords_text = result.strip()
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
        df['summary'] = "" # 요약 결과 저장용 컬럼 추가
        df['keywords'] = "" # 키워드 결과 저장용 컬럼 추가

        # 행단위 반복
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
        
        # 입력 파일로드
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