"""
ReAct (Reasoning and Acting) 프롬프팅 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional

class ReActSummarizer:
   """ReAct 접근법을 사용한 요약 클래스"""
   
   def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=800):
       """
       ReAct 요약기 초기화
       
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
       
       # ReAct 프롬프트 템플릿 정의
       self.template = """
       당신은 뉴스 기사를 요약하는 AI 도우미입니다. Reasoning과 Acting을 번갈아가며 단계적으로 뉴스 기사를 분석하고 요약해주세요.
       
       다음 형식으로 진행하세요:
       1. Thought: 기사에 대한 생각과 분석
       2. Action: 수행할 작업 (주요 주제 식별, 중요 세부사항 추출, 불필요한 정보 제거 등)
       3. Observation: 작업 결과 관찰
       반복...
       
       마지막에는 "Final Summary:"로 시작하는 3-4문장 길이의 최종 요약을 제공하세요.
       
       뉴스 기사:
       {text}
       
       Thought:
       """
       
       self.prompt = PromptTemplate(
           input_variables=["text"],
           template=self.template
       )
       
       # 체인 설정
       self.chain = LLMChain(llm=self.chat_model, prompt=self.prompt)
   
   def summarize(self, text: str) -> str:
       """
       ReAct 방식으로 텍스트 요약
       
       Args:
           text (str): 요약할 텍스트
           
       Returns:
           str: 요약된 텍스트
       """
       if not text or len(text.strip()) < 10:
           return ""
       
       # ReAct 과정 실행
       react_result = self.chain.run(text=text)
       
       # 최종 요약만 추출 ("Final Summary:" 이후 부분)
       if "Final Summary:" in react_result:
           final_summary = react_result.split("Final Summary:")[1].strip()
       else:
           # 발견되지 않으면 전체 반환
           final_summary = react_result
       
       return final_summary