"""
Graph Prompting 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional

class GraphPromptingSummarizer:
    """Graph Prompting 접근법을 사용한 요약 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=1000):
        """
        Graph Prompting 요약기 초기화
        
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
        
        # 그래프 요소 추출 템플릿 정의
        self.entity_template = """
        다음 뉴스 기사에서 주요 개체(사람, 조직, 장소, 개념 등)를 추출해주세요.
        각 개체에 대해 간략한 설명을 추가해주세요.
        
        뉴스 기사:
        {text}
        
        주요 개체 목록:
        """
        
        self.entity_prompt = PromptTemplate(
            input_variables=["text"],
            template=self.entity_template
        )
        
        # 관계 추출 템플릿 정의
        self.relation_template = """
        다음 뉴스 기사와 추출된 주요 개체 목록을 바탕으로, 개체 간의 관계를 그래프 형태로 설명해주세요.
        각 관계는 "개체1 -> [관계] -> 개체2" 형식으로 작성해주세요.
        
        뉴스 기사:
        {text}
        
        주요 개체:
        {entities}
        
        개체 간 관계:
        """
        
        self.relation_prompt = PromptTemplate(
            input_variables=["text", "entities"],
            template=self.relation_template
        )
        
        # 그래프 기반 요약 템플릿 정의
        self.summary_template = """
        다음 뉴스 기사와 추출된 개체 및 관계 그래프를 바탕으로 요약을 생성해주세요.
        그래프의 주요 노드(개체)와 엣지(관계)를 중심으로 핵심 내용을 3-4문장으로 요약해주세요.
        
        뉴스 기사:
        {text}
        
        주요 개체:
        {entities}
        
        개체 간 관계:
        {relations}
        
        그래프 기반 요약:
        """
        
        self.summary_prompt = PromptTemplate(
            input_variables=["text", "entities", "relations"],
            template=self.summary_template
        )
        
        # 체인 설정
        self.entity_chain = LLMChain(llm=self.chat_model, prompt=self.entity_prompt)
        self.relation_chain = LLMChain(llm=self.chat_model, prompt=self.relation_prompt)
        self.summary_chain = LLMChain(llm=self.chat_model, prompt=self.summary_prompt)
    
    def summarize(self, text: str) -> str:
       """
       Graph Prompting 방식으로 텍스트 요약
       
       Args:
           text (str): 요약할 텍스트
           
       Returns:
           str: 요약된 텍스트
       """
       if not text or len(text.strip()) < 10:
           return ""
       
       # 1단계: 주요 개체 추출
       entities = self.entity_chain.run(text=text)
       
       # 2단계: 개체 간 관계 추출
       relations = self.relation_chain.run(text=text, entities=entities)
       
       # 3단계: 그래프 기반 요약 생성
       summary = self.summary_chain.run(
           text=text,
           entities=entities,
           relations=relations
       )
       
       return summary.strip()
   

    def get_knowledge_graph(self, text: str) -> Dict[str, Any]:
       """
       텍스트에서 지식 그래프 추출
       
       Args:
           text (str): 분석할 텍스트
           
       Returns:
           Dict[str, Any]: 지식 그래프 데이터
       """
       if not text or len(text.strip()) < 10:
           return {"entities": [], "relations": []}
       
       # 개체 및 관계 추출
       entities = self.entity_chain.run(text=text)
       relations = self.relation_chain.run(text=text, entities=entities)
       
       # 간단한 파싱 (실제 구현에서는 더 정교한 파싱이 필요)
       entity_list = [e.strip() for e in entities.split('\n') if e.strip()]
       relation_list = [r.strip() for r in relations.split('\n') if r.strip()]
       
       return {
           "entities": entity_list,
           "relations": relation_list
       }