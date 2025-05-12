"""
Prompt Chaining 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional

class PromptChainingSummarizer:
    """Prompt Chaining 접근법을 사용한 요약 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=500):
        """
        Prompt Chaining 요약기 초기화
        
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
        
        # 체인 1: 핵심 주제 식별
        self.topic_template = """
        다음 뉴스 기사에서 다루는 핵심 주제와 중심 내용을 간결하게 알려주세요.
        
        뉴스 기사:
        {text}
        
        핵심 주제:
        """
        
        self.topic_prompt = PromptTemplate(
            input_variables=["text"],
            template=self.topic_template
        )
        
        self.topic_chain = LLMChain(
            llm=self.chat_model,
            prompt=self.topic_prompt,
            output_key="topic"
        )
        
        # 체인 2: 중요 세부사항 추출
        self.details_template = """
        다음 뉴스 기사의 주요 세부사항과 중요한 사실을 추출해주세요.
        
        뉴스 기사:
        {text}
        
        핵심 주제:
        {topic}
        
        중요 세부사항:
        """
        
        self.details_prompt = PromptTemplate(
            input_variables=["text", "topic"],
            template=self.details_template
        )
        
        self.details_chain = LLMChain(
            llm=self.chat_model,
            prompt=self.details_prompt,
            output_key="details"
        )
        
        # 체인 3: 최종 요약 생성
        self.summary_template = """
        다음 정보를 바탕으로 뉴스 기사의 핵심 내용을 3-4문장으로 통합하여 요약해주세요.
        요약은 불렛포인트 없이 자연스러운 문단 형태로 작성해주세요.
        
        핵심 주제:
        {topic}
        
        중요 세부사항:
        {details}
        
        최종 요약:
        """
        
        self.summary_prompt = PromptTemplate(
            input_variables=["topic", "details"],
            template=self.summary_template
        )
        
        self.summary_chain = LLMChain(
            llm=self.chat_model,
            prompt=self.summary_prompt,
            output_key="summary"
        )
        
        # 순차 체인 설정
        self.overall_chain = SequentialChain(
            chains=[self.topic_chain, self.details_chain, self.summary_chain],
            input_variables=["text"],
            output_variables=["topic", "details", "summary"],
            verbose=False
        )
    
    def summarize(self, text: str) -> str:
        """
        Prompt Chaining 방식으로 텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            
        Returns:
            str: 요약된 텍스트
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        # 순차 체인 실행
        result = self.overall_chain({"text": text})
        
        return result["summary"].strip()
    
    def summarize_with_intermediate(self, text: str) -> Dict[str, str]:
        """
        중간 결과를 포함한 요약 생성
        
        Args:
            text (str): 요약할 텍스트
            
        Returns:
            Dict[str, str]: 중간 결과를 포함한 요약 결과
        """
        if not text or len(text.strip()) < 10:
            return {"topic": "", "details": "", "summary": ""}
        
        # 순차 체인 실행 및 중간 결과 포함 반환
        result = self.overall_chain({"text": text})
        
        return {
            "topic": result["topic"],
            "details": result["details"],
            "summary": result["summary"]
        }