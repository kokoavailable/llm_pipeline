"""
Active-Prompt 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional
import re

class ActivePromptSummarizer:
   """Active-Prompt 접근법을 사용한 요약 클래스"""
   
   def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=500, iterations=2):
       """
       Active-Prompt 요약기 초기화
       
       Args:
           model_name (str): 사용할 OpenAI 모델
           temperature (float): 생성 다양성 조절 (0~1)
           max_tokens (int): 최대 토큰 수
           iterations (int): 활성 프롬프팅 반복 횟수
       """
       self.model_name = model_name
       self.temperature = temperature
       self.max_tokens = max_tokens
       self.iterations = iterations
       
       # 모델 초기화
       self.chat_model = ChatOpenAI(
           model_name=model_name,
           temperature=temperature,
           max_tokens=max_tokens
       )
       
       # 초기 요약 프롬프트 템플릿
       self.initial_template = """
       다음 뉴스 기사를 읽고 핵심 내용을 3-4문장으로 요약해주세요.
       요약은 불렛포인트 없이 자연스러운 문단 형태로 작성해주세요.
       
       뉴스 기사:
       {text}
       
       요약:
       """
       
       self.initial_prompt = PromptTemplate(
           input_variables=["text"],
           template=self.initial_template
       )
       
       # 요약 개선 프롬프트 템플릿
       self.feedback_template = """
       다음 뉴스 기사와 현재 요약을 검토해주세요.
       
       뉴스 기사:
       {text}
       
       현재 요약:
       {current_summary}
       
       현재 요약의 문제점을 분석하고, 다음을 고려하여 개선해주세요:
       1. 중요한 정보가 누락되었는가?
       2. 불필요한 정보가 포함되었는가?
       3. 정확성에 문제가 있는가?
       4. 표현이 명확하고 자연스러운가?
       
       개선된 요약:
       """
       
       self.feedback_prompt = PromptTemplate(
           input_variables=["text", "current_summary"],
           template=self.feedback_template
       )
       
       # 체인 설정
       self.initial_chain = LLMChain(llm=self.chat_model, prompt=self.initial_prompt)
       self.feedback_chain = LLMChain(llm=self.chat_model, prompt=self.feedback_prompt)
   
   def summarize(self, text: str) -> str:
       """
       Active-Prompt 방식으로 텍스트 요약
       
       Args:
           text (str): 요약할 텍스트
           
       Returns:
           str: 요약된 텍스트
       """
       if not text or len(text.strip()) < 10:
           return ""
       
       # 1단계: 초기 요약 생성
       current_summary = self.initial_chain.run(text=text)
       
       # 2단계: 반복적 개선
       for i in range(self.iterations):
           # 현재 요약에 대한 피드백 및 개선
           improved_summary = self.feedback_chain.run(
               text=text,
               current_summary=current_summary
           )
           
           current_summary = improved_summary
       
       return current_summary.strip()