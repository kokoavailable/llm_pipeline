"""
Reflexion 프롬프팅 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional

class ReflexionSummarizer:
    """Reflexion 접근법을 사용한 요약 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=800, reflections=2):
        """
        Reflexion 요약기 초기화
        
        Args:
            model_name (str): 사용할 OpenAI 모델
            temperature (float): 생성 다양성 조절 (0~1)
            max_tokens (int): 최대 토큰 수
            reflections (int): 성찰 반복 횟수
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reflections = reflections
        
        # 모델 초기화
        self.chat_model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 초기 요약 템플릿 정의
        self.initial_template = """
        다음 뉴스 기사를 읽고 핵심 내용을 3-4문장으로 요약해주세요.
        
        뉴스 기사:
        {text}
        
        요약:
        """
        
        self.initial_prompt = PromptTemplate(
            input_variables=["text"],
            template=self.initial_template
        )
        
        # 평가 템플릿 정의
        self.evaluation_template = """
        다음 뉴스 기사와 요약을 검토하고, 요약의 품질을 평가해주세요.
        
        뉴스 기사:
        {text}
        
        요약:
        {summary}
        
        다음 기준에 따라 평가해주세요:
        1. 정확성: 요약이 기사의 내용을 정확하게 반영하는가?
        2. 완전성: 모든 중요한 정보가 포함되었는가?
        3. 간결성: 불필요한 정보 없이 핵심만 포함하는가?
        4. 명확성: 요약이 명확하고 이해하기 쉬운가?
        
        각 기준에 대해 강점과 약점을 자세히 설명해주세요.
        
        평가:
        """
        
        self.evaluation_prompt = PromptTemplate(
            input_variables=["text", "summary"],
            template=self.evaluation_template
        )
        
        # 성찰 및 개선 템플릿 정의
        self.reflection_template = """
        다음 뉴스 기사, 현재 요약, 그리고 평가를 검토하고, 반성적 사고를 통해 개선된 요약을 작성해주세요.
        
        뉴스 기사:
        {text}
        
        현재 요약:
        {summary}
        
        평가:
        {evaluation}
        
        반성:
        1. 평가에서 지적된 문제점은 무엇인가요?
        2. 어떤 부분이 누락되었거나 부정확했나요?
        3. 어떻게 하면 요약을 개선할 수 있을까요?
        
        개선된 요약:
        """
        
        self.reflection_prompt = PromptTemplate(
            input_variables=["text", "summary", "evaluation"],
            template=self.reflection_template
        )
        
        # 체인 설정
        self.initial_chain = LLMChain(llm=self.chat_model, prompt=self.initial_prompt)
        self.evaluation_chain = LLMChain(llm=self.chat_model, prompt=self.evaluation_prompt)
        self.reflection_chain = LLMChain(llm=self.chat_model, prompt=self.reflection_prompt)
    
    def summarize(self, text: str) -> str:
        """
        Reflexion 방식으로 텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            
        Returns:
            str: 요약된 텍스트
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        # 1단계: 초기 요약 생성
        current_summary = self.initial_chain.run(text=text)
        
        # 2단계: 반복적 성찰 및 개선
        for i in range(self.reflections):
            # 현재 요약 평가
            evaluation = self.evaluation_chain.run(
                text=text,
                summary=current_summary
            )
            
            # 성찰 및 개선
            improved_summary = self.reflection_chain.run(
                text=text,
                summary=current_summary,
                evaluation=evaluation
            )
            
            current_summary = improved_summary
        
        return current_summary.strip()