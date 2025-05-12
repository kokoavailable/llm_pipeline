"""
Multimodal Chain-of-Thought 프롬프팅 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional

class MultimodalCoTSummarizer:
    """Multimodal Chain-of-Thought 접근법을 사용한 요약 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=800):
        """
        Multimodal CoT 요약기 초기화
        
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
        
        # 멀티모달 CoT 프롬프트 템플릿 정의
        # 주의: 실제 멀티모달 입력이 필요하지만, 텍스트 기반으로 시뮬레이션
        self.template = """
        당신은 텍스트와 시각적 정보를 모두 활용하여 요약하는 AI입니다.
        다음 뉴스 기사를 읽고, 여러 모달리티를 고려한 사고 과정을 통해 요약해주세요.
        
        다음 단계에 따라 생각하세요:
        
        1. 텍스트 분석:
           - 기사의 주요 주제와 핵심 메시지 파악
           - 중요한 사실, 인물, 날짜, 장소 식별
           - 기사의 논리적 구조와 흐름 파악
        
        2. 가상의 시각적 요소 상상:
           - 이 기사가 담고 있는 내용을 이미지나 그래픽으로 표현한다면 어떤 모습일지 상상
           - 주요 개념이나 관계를 다이어그램으로 구조화
           - 핵심 정보를 시각적으로 배치
        
        3. 멀티모달 통합:
           - 텍스트와 가상의 시각적 요소를 함께 고려하여 핵심 내용 통합
           - 두 모달리티 정보의 교차점에서 가장 중요한 요소 식별
        
        4. 최종 요약 생성:
           - 멀티모달 사고 과정을 바탕으로 3-4문장의 통합된 요약 작성
        
        뉴스 기사:
        {text}
        
        단계별 사고 과정:
        """
        
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=self.template
        )
        
        # 체인 설정
        self.chain = LLMChain(llm=self.chat_model, prompt=self.prompt)
    
    def summarize(self, text: str) -> str:
        """
        Multimodal Chain-of-Thought 방식으로 텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            
        Returns:
            str: 요약된 텍스트
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        # Multimodal CoT 과정 실행
        cot_result = self.chain.run(text=text)
        
        # 최종 요약만 추출
        if "최종 요약:" in cot_result:
            final_summary = cot_result.split("최종 요약:")[1].strip()
        else:
            # 마지막 단락 사용
            paragraphs = [p for p in cot_result.split("\n\n") if p.strip()]
            final_summary = paragraphs[-1] if paragraphs else cot_result
        
        return final_summary.strip()