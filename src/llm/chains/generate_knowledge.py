"""
Generate Knowledge Prompting 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional

class GenerateKnowledgeSummarizer:
    """Generate Knowledge 접근법을 사용한 요약 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=800):
        """
        Generate Knowledge 요약기 초기화
        
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
        
        # 지식 생성 프롬프트 템플릿 정의
        self.knowledge_template = """
        당신은 뉴스 기사를 분석하고 관련 지식과 배경 정보를 생성하는 전문가입니다.
        다음 뉴스 기사를 읽고, 이 기사를 더 잘 이해하는 데 도움이 될 관련 배경 지식을 5가지 생성해주세요.
        
        뉴스 기사:
        {text}
        
        관련 배경 지식:
        """
        
        self.knowledge_prompt = PromptTemplate(
            input_variables=["text"],
            template=self.knowledge_template
        )
        
        # 요약 프롬프트 템플릿 정의
        self.summary_template = """
        다음 뉴스 기사를 읽고, 제공된 배경 지식을 활용하여 3-4문장으로 요약해주세요.
        
        뉴스 기사:
        {text}
        
        배경 지식:
        {knowledge}
        
        이 정보를 바탕으로, 객관적이고 정확한 요약을 작성해주세요:
        """
        
        self.summary_prompt = PromptTemplate(
            input_variables=["text", "knowledge"],
            template=self.summary_template
        )
        
        # 체인 설정
        self.knowledge_chain = LLMChain(llm=self.chat_model, prompt=self.knowledge_prompt)
        self.summary_chain = LLMChain(llm=self.chat_model, prompt=self.summary_prompt)
    
    def summarize(self, text: str) -> str:
        """
        Generate Knowledge 방식으로 텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            
        Returns:
            str: 요약된 텍스트
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        # 1단계: 관련 배경 지식 생성
        knowledge = self.knowledge_chain.run(text=text)
        
        # 2단계: 생성된 지식을 활용하여 요약 수행
        summary = self.summary_chain.run(text=text, knowledge=knowledge)
        
        return summary.strip()