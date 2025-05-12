"""
Tree of Thoughts 프롬프팅 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional

class TreeOfThoughtsSummarizer:
    """Tree of Thoughts 접근법을 사용한 요약 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=1000):
        """
        Tree of Thoughts 요약기 초기화
        
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
        
        # Tree of Thoughts 프롬프트 템플릿 정의
        self.template = """
        당신은 뉴스 기사를 요약하는 AI 도우미입니다. Tree of Thoughts 방법론을 사용하여 여러 가능한 접근법을 탐색하고 최적의 요약을 생성하세요.
        
        다음 단계를 따르세요:

        1. 먼저, 기사 내용에서 추출할 수 있는 주요 관점 3가지를 생각하세요.
        2. 각 관점에 대해, 그 관점을 중심으로 한 가능한 요약을 생성하세요.
        3. 각 요약을 평가하고 가장 좋은 요약을 선택하세요.
        4. 최종 선택된 요약을 3-4문장으로 다듬어 제공하세요.
        
        뉴스 기사:
        {text}
        
        == 사고의 흐름 ==
        """
        
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=self.template
        )
        
        # 체인 설정
        self.chain = LLMChain(llm=self.chat_model, prompt=self.prompt)
    
    def summarize(self, text: str) -> str:
        """
        Tree of Thoughts 방식으로 텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            
        Returns:
            str: 요약된 텍스트
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        # Tree of Thoughts 과정 실행
        tot_result = self.chain.run(text=text)
        
        # 최종 요약만 추출 ("최종 요약:" 또는 "Final Summary:" 이후 부분)
        if "최종 요약:" in tot_result:
            final_summary = tot_result.split("최종 요약:")[1].strip()
        elif "Final Summary:" in tot_result:
            final_summary = tot_result.split("Final Summary:")[1].strip()
        else:
            # 마지막 단락 사용
            paragraphs = [p for p in tot_result.split("\n\n") if p.strip()]
            final_summary = paragraphs[-1] if paragraphs else tot_result
        
        return final_summary