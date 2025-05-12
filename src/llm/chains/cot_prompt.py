"""
Chain-of-Thought 프롬프팅 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional

class ChainOfThoughtSummarizer:
    """Chain-of-Thought 접근법을 사용한 요약 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=500):
        """
        CoT 요약기 초기화
        
        Args:
            model_name (str): 사용할 OpenAI 모델
            temperature (float): 생성 다양성 조절 (0~1)
            max_tokens (int): 최대 토큰 수
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 모델 초기화
        self.chat_model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Chain-of-Thought 프롬프트 템플릿 정의
        # 단계별 reasoning
        self.template = """
        다음 뉴스 기사를 읽고, 단계적으로 생각하며 요약해 주세요.
        
        1. 먼저, 기사의 주요 주제나 핵심 정보가 무엇인지 파악하세요.
        2. 다음으로, 기사에서 제시된 중요한 세부 사항들을 확인하세요.
        3. 마지막으로, 이 정보들을 바탕으로 3-4문장으로 된 자연스러운 요약을 작성하세요.
        
        뉴스 기사:
        {text}
        
        단계별 사고:
        """
        
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=self.template
        )
        
        # 체인 설정
        self.chain = LLMChain(llm=self.chat_model, prompt=self.prompt)
    
    def summarize(self, text: str) -> str:
        """
        Chain-of-Thought 방식으로 텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            
        Returns:
            str: 요약된 텍스트
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        # CoT 과정 실행
        cot_result = self.chain.run(text=text)
        
        # 최종 요약만 추출 (마지막 단락 또는 "최종 요약:" 이후 부분)
        if "최종 요약:" in cot_result:
            final_summary = cot_result.split("최종 요약:")[1].strip()
        else:
            # 줄바꿈으로 분리하고 마지막 단락 사용
            paragraphs = [p for p in cot_result.split("\n\n") if p.strip()]
            final_summary = paragraphs[-1] if paragraphs else cot_result
        
        return final_summary