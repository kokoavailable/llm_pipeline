"""
Automatic Reasoning and Tool-use 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from typing import List, Dict, Any, Optional
import re

class EntityExtractionTool(BaseTool):
    """주요 엔티티(개체) 추출 도구"""
    
    name: str = "extract_entities"  # 타입 주석 추가
    description: str = "뉴스 기사에서 주요 개체(사람, 조직, 장소, 날짜 등)를 추출합니다."
    
    def __init__(self, llm):
        """도구 초기화"""
        super().__init__()
        self.llm = llm
        
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            다음 텍스트에서 주요 개체(사람, 조직, 장소, 날짜 등)를 추출하고 분류해주세요.
            각 항목은 "유형: 개체명"의 형식으로 작성해주세요.
            
            텍스트:
            {text}
            
            개체 목록:
            """
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _run(self, text: str) -> str:
        """도구 실행"""
        return self.chain.run(text=text)

class KeyPointExtractionTool(BaseTool):
    """핵심 포인트 추출 도구"""
    
    name: str = "extract_key_points"  # 타입 주석 추가
    description: str = "뉴스 기사에서 핵심 포인트와 주요 주장을 추출합니다."
    
    def __init__(self, llm):
        """도구 초기화"""
        super().__init__()
        self.llm = llm
        
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            다음 텍스트에서 가장 중요한 핵심 포인트 5개를 추출해주세요.
            각 포인트는 간결한 문장 형태로 작성해주세요.
            
            텍스트:
            {text}
            
            핵심 포인트:
            """
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _run(self, text: str) -> str:
        """도구 실행"""
        return self.chain.run(text=text)

class SentimentAnalysisTool(BaseTool):
    """감정 분석 도구"""
    
    name: str = "analyze_sentiment"  # 타입 주석 추가
    description: str = "뉴스 기사의 전반적인 논조와 감정을 분석합니다."
    
    def __init__(self, llm):
        """도구 초기화"""
        super().__init__()
        self.llm = llm
        
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            다음 텍스트의 전반적인 논조(긍정, 부정, 중립)와 감정을 분석해주세요.
            분석은 객관적이어야 하며, 텍스트의 어떤 부분이 해당 논조를 나타내는지 설명해주세요.
            
            텍스트:
            {text}
            
            논조 분석:
            """
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _run(self, text: str) -> str:
        """도구 실행"""
        return self.chain.run(text=text)

class AutoReasoningToolUseSummarizer:
    """Automatic Reasoning and Tool-use 접근법을 사용한 요약 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=800):
        """
        Automatic Reasoning and Tool-use 요약기 초기화
        
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
        
        # 도구 초기화
        self.tools = [
            EntityExtractionTool(self.chat_model),
            KeyPointExtractionTool(self.chat_model),
            SentimentAnalysisTool(self.chat_model)
        ]
        
        # 에이전트 초기화
        self.agent = initialize_agent(
            self.tools,
            self.chat_model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False
        )
        
        # 최종 요약 체인 설정
        self.summary_template = """
        다음 뉴스 기사와 분석 결과를 바탕으로 객관적이고 정확한 요약을 3-4문장으로 작성해주세요.
        
        뉴스 기사:
        {text}
        
        분석 결과:
        {analysis}
        
        최종 요약:
        """
        
        self.summary_prompt = PromptTemplate(
            input_variables=["text", "analysis"],
            template=self.summary_template
        )
        
        self.summary_chain = LLMChain(llm=self.chat_model, prompt=self.summary_prompt)
    
    def summarize(self, text: str) -> str:
        """
        Automatic Reasoning and Tool-use 방식으로 텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            
        Returns:
            str: 요약된 텍스트
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        # 1단계: 에이전트를 사용한 텍스트 분석
        agent_prompt = f"""
        다음 뉴스 기사를 분석하기 위해 사용 가능한 도구들을 활용하세요:
        1. 먼저 extract_entities 도구를 사용하여 주요 개체를 추출하세요.
        2. 다음으로 extract_key_points 도구를 사용하여 핵심 포인트를 추출하세요.
        3. 마지막으로 analyze_sentiment 도구를 사용하여 전반적인 논조를 분석하세요.
        
        뉴스 기사:
        {text}
        """
        
        analysis_result = self.agent.run(agent_prompt)
        
        # 2단계: 분석 결과를 활용한 요약 생성
        summary = self.summary_chain.run(text=text, analysis=analysis_result)
        
        return summary.strip()