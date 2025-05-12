"""
Directional Stimulus Prompting (DSP) 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional

class DirectionalStimulusPromptingSummarizer:
    """Directional Stimulus Prompting 접근법을 사용한 요약 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=500):
        """
        DSP 요약기 초기화
        
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
        
        # 자극 방향성 템플릿 정의
        self.stimulus_templates = {
            "객관성": """
            당신은 뉴스 기사를 완전히 객관적인 관점에서 요약하는 AI입니다.
            개인적인 의견이나 해석을 절대 포함하지 않고, 오직 기사에 명시된 사실만을 바탕으로 요약해야 합니다.
            모든 표현은 중립적이어야 하며, 감정적인 단어나 편향된 용어를 사용해서는 안 됩니다.
            
            다음 뉴스 기사를 읽고, 완전히 객관적인 관점에서 3-4문장으로 요약해주세요.
            
            뉴스 기사:
            {text}
            
            객관적 요약:
            """,
            
            "간결성": """
            당신은 뉴스 기사를 극도로 간결하게 요약하는 AI입니다.
            불필요한 세부사항이나 부가 설명 없이, 오직 가장 핵심적인 정보만을 포함해야 합니다.
            모든 문장은 짧고 명확해야 하며, 군더더기 없는 표현을 사용해야 합니다.
            
            다음 뉴스 기사를 읽고, 극도로 간결하게 3-4문장으로 요약해주세요.
            
            뉴스 기사:
            {text}
            
            간결한 요약:
            """,
            
            "포괄성": """
            당신은 뉴스 기사의 모든 중요한 측면을 포괄적으로 요약하는 AI입니다.
            기사에서 다루는 주요 주제, 관점, 증거, 결과 등 모든 중요한 요소가 요약에 포함되어야 합니다.
            어떤 핵심 정보도 누락되지 않도록 주의하면서, 전체적인 맥락을 유지해야 합니다.
            
            다음 뉴스 기사를 읽고, 모든 중요한 측면을 포함하여 3-4문장으로 요약해주세요.
            
            뉴스 기사:
            {text}
            
            포괄적 요약:
            """
        }
        
        # 요약 통합 템플릿
        self.integration_template = """
        다음 뉴스 기사에 대한 세 가지 다른 관점의 요약을 검토하고, 이를 통합하여 최상의 요약을 만들어주세요.
        요약은 3-4문장으로, 객관적이고, 간결하며, 포괄적이어야 합니다.
        
        뉴스 기사:
        {text}
        
        객관적 요약:
        {objective_summary}
        
        간결한 요약:
        {concise_summary}
        
        포괄적 요약:
        {comprehensive_summary}
        
        통합된 최종 요약:
        """
        
        self.integration_prompt = PromptTemplate(
            input_variables=["text", "objective_summary", "concise_summary", "comprehensive_summary"],
            template=self.integration_template
        )
        
        # 체인 설정
        self.stimulus_prompts = {}
        self.stimulus_chains = {}
        
        for direction, template in self.stimulus_templates.items():
            self.stimulus_prompts[direction] = PromptTemplate(
                input_variables=["text"],
                template=template
            )
            self.stimulus_chains[direction] = LLMChain(
                llm=self.chat_model,
                prompt=self.stimulus_prompts[direction]
            )
        
        self.integration_chain = LLMChain(llm=self.chat_model, prompt=self.integration_prompt)
    
    def summarize(self, text: str) -> str:
        """
        Directional Stimulus Prompting 방식으로 텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            
        Returns:
            str: 요약된 텍스트
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        # 1단계: 각 방향성에 따른 요약 생성
        summaries = {}
        for direction, chain in self.stimulus_chains.items():
            summaries[direction] = chain.run(text=text)
        
        # 2단계: 요약 통합
        integrated_summary = self.integration_chain.run(
            text=text,
            objective_summary=summaries.get("객관성", ""),
            concise_summary=summaries.get("간결성", ""),
            comprehensive_summary=summaries.get("포괄성", "")
        )
        
        return integrated_summary.strip()