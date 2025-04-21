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
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

class GPTSummarizer:
    """GPT를 사용한 텍스트 요약 클래스"""

    def __init__(self, model="gpt-3.5-turbo"):
        """
        요약기 초기화

        Args:
            model (str): 사용할 OpenAI 모델
        """
        self.model = model
        self._check_api_key()

    def _check_api_key(self):
        """API 키 확인"""
        if not openai.api_key:
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
        system_prompt = f"""
        당신은 뉴스 기사를 읽고, 핵심 내용을 부드럽고 자연스러운 단락으로 요약하는 AI입니다.
        요약은 객관적이며 불렛포인트 없이, 사람이 쓴 것처럼 자연스럽게 이어지는 3~4문장으로 작성되어야 합니다.
        """
        
        # 요약 요청을 위한 사용자 프롬프트
        user_prompt = f"""
        다음은 뉴스 기사입니다. 중요한 내용을 중심으로, 자연스럽고 흐름이 끊기지 않는 단락으로 3~4문장 정도로 요약해주세요. 요약은 마치 사람이 쓴 것처럼 자연스럽게 이어지고, 불렛포인트나 나열식 표현 없이 작성되어야 합니다.

        기사 원문:
        {text}

        요약:
        """

        
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
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

                summary = response.choices[0].message['content'].strip()
                return summary
            
            except Exception as e:
                logger.error(f"요약 생성 중 오류 발생 (시도 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 지수 백오프
                    logger.info(f"{wait_time}초 후 재시도...")
                    time.sleep(wait_time)