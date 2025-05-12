"""
Zero-shot 프롬프팅 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional

class ZeroShotSummarizer:
    """Zero-shot 접근법을 사용한 요약 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=300):
        """
        Zero-shot 요약기 초기화
        
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
        
        # Zero-shot 프롬프트 템플릿 정의
        # 직접적 예시 없이 무엇을 해야하는지만으로 명령
        self.template = """
        다음 텍스트를 읽고 핵심 내용을 3-4문장으로 요약해주세요.
        요약은 불렛포인트 없이 자연스러운 문단 형태로 작성해주세요.
        
        텍스트: {text}
        
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
        Zero-shot 방식으로 텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            
        Returns:
            str: 요약된 텍스트
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        result = self.chain.run(text=text)
        return result.strip()