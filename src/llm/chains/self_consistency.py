"""
Self-Consistency 프롬프팅 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter

class SelfConsistencySummarizer:
    """Self-Consistency 접근법을 사용한 요약 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.8, max_tokens=300, n_generations=3):
        """
        Self-Consistency 요약기 초기화
        
        Args:
            model_name (str): 사용할 OpenAI 모델
            temperature (float): 생성 다양성 조절 (0~1)
            max_tokens (int): 최대 토큰 수
            n_generations (int): 생성할 요약 수
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_generations = n_generations
        
        # 모델 초기화
        self.chat_model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 기본 프롬프트 템플릿 정의
        self.template = """
        다음 뉴스 기사를 읽고 핵심 내용을 3-4문장으로 요약해주세요.
        요약은 불렛포인트 없이 자연스러운 문단 형태로 작성해주세요.
        
        뉴스 기사:
        {text}
        
        요약:
        """
        
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=self.template
        )
        
        # 체인 설정
        self.chain = LLMChain(llm=self.chat_model, prompt=self.prompt)
    
    def summarize(self, text: str) -> str:
        """
        Self-Consistency 방식으로 텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            
        Returns:
            str: 요약된 텍스트
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        # 여러 요약 생성
        summaries = []
        for _ in range(self.n_generations):
            result = self.chain.run(text=text)
            summaries.append(result.strip())
        
        # 가장 일관된 요약 선택 (간단한 구현)
        # 여기서는 가장 자주 등장하는 요약을 선택
        if self.n_generations > 1:
            summary_counter = Counter(summaries)
            most_common_summary = summary_counter.most_common(1)[0][0]
            return most_common_summary
        else:
            return summaries[0]
    
    def analyze_consistency(self, summaries: List[str]) -> Dict[str, Any]:
        """
        생성된 요약들의 일관성 분석
        
        Args:
            summaries (List[str]): 생성된 요약 목록
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        # 간단한 구현으로, 중복된 요약 수와 고유 요약 수 반환
        unique_summaries = set(summaries)
        
        return {
            "total_summaries": len(summaries),
            "unique_summaries": len(unique_summaries),
            "consistency_ratio": 1 - (len(unique_summaries) - 1) / len(summaries) if len(summaries) > 1 else 1
        }