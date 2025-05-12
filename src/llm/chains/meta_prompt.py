"""
Meta Prompting 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional

class MetaPromptSummarizer:
    """Meta Prompting 접근법을 사용한 요약 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=800):
        """
        Meta Prompt 요약기 초기화
        
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
        
        # 메타 프롬프트 템플릿 정의
        self.meta_template = """
        당신은 효과적인 프롬프트를 작성하는 전문가입니다. 뉴스 기사를 요약하는 최적의 프롬프트를 생성해주세요.
        
        생성할 프롬프트는 다음 기준을 만족해야 합니다:
        - 명확하고 구체적인 지시사항 포함
        - 요약의 길이와 형식 명시
        - 객관성과 중요도에 기반한 요약 강조
        - 불필요한 세부사항 배제 지시
        
        뉴스 기사의 내용에 기반하여 가장 효과적인 요약 프롬프트를 생성해주세요.
        
        뉴스 기사:
        {text}
        
        효과적인 요약 프롬프트:
        """
        
        self.meta_prompt = PromptTemplate(
            input_variables=["text"],
            template=self.meta_template
        )
        
        # 메타 체인 설정
        self.meta_chain = LLMChain(llm=self.chat_model, prompt=self.meta_prompt)
    
    def summarize(self, text: str) -> str:
        """
        Meta Prompting 방식으로 텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            
        Returns:
            str: 요약된 텍스트
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        # 1단계: 메타 프롬프트로 최적의 요약 프롬프트 생성
        custom_prompt_text = self.meta_chain.run(text=text)
        
        # 2단계: 생성된 프롬프트로 새 PromptTemplate 생성
        custom_template = f"{custom_prompt_text}\n\n{text}\n\n"
        custom_prompt = PromptTemplate(
            input_variables=[],
            template=custom_template
        )
        
        # 3단계: 생성된 프롬프트로 요약 수행
        summary_chain = LLMChain(llm=self.chat_model, prompt=custom_prompt)
        summary = summary_chain.run()
        
        return summary.strip()