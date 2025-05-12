"""
Program-Aided Language Models (PALM) 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional
import re
import numpy as np

class ProgramAidedLanguageModelSummarizer:
    """Program-Aided Language Models 접근법을 사용한 요약 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=800):
        """
        PALM 요약기 초기화
        
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
        
        # 프로그램 생성 템플릿 정의
        self.program_template = """
        당신은 텍스트 요약을 위한 파이썬 프로그램을 작성하는 AI 프로그래머입니다.
        다음 뉴스 기사를 요약하기 위한 파이썬 프로그램을 작성해주세요.
        
        프로그램은 다음 단계를 포함해야 합니다:
        1. 텍스트 전처리 (불필요한 부분 제거, 문장 분리 등)
        2. 중요 문장 식별 (예: TF-IDF, 문장 위치 등을 고려)
        3. 핵심 키워드 추출
        4. 선택된 정보를 바탕으로 3-4문장 요약 생성
        
        프로그램은 실행 가능해야 하며, 기사 텍스트를 변수로 저장하고 처리해야 합니다.
        간단한 NLP 기법을 사용하되, 복잡한 외부 라이브러리는 최소화해주세요.
        
        뉴스 기사:
        {text}
        
        파이썬 프로그램:
        ```python
        """
        
        self.program_prompt = PromptTemplate(
            input_variables=["text"],
            template=self.program_template
        )
        
        # 코드 해석 및 요약 적용 템플릿
        self.execution_template = """
        다음은 뉴스 기사를 요약하기 위한 파이썬 프로그램입니다. 이 프로그램의 로직을 분석하고, 
        동일한 로직을 따라 기사를 3-4문장으로 요약해주세요.
        
        뉴스 기사:
        {text}
        
        요약 프로그램:
        {program}
        
        프로그램 로직을 따른 요약:
        """
        
        self.execution_prompt = PromptTemplate(
            input_variables=["text", "program"],
            template=self.execution_template
        )
        
        # 체인 설정
        self.program_chain = LLMChain(llm=self.chat_model, prompt=self.program_prompt)
        self.execution_chain = LLMChain(llm=self.chat_model, prompt=self.execution_prompt)
    
    def _extract_python_code(self, text: str) -> str:
        """
        텍스트에서 파이썬 코드 추출
        
        Args:
            text (str): 코드가 포함된 텍스트
            
        Returns:
            str: 추출된 파이썬 코드
        """
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        else:
            # 코드 블록 없이 직접 코드가 포함된 경우
            lines = text.split("\n")
            code_lines = []
            in_code = False
            
            for line in lines:
                if line.strip().startswith("def ") or line.strip().startswith("import "):
                    in_code = True
                
                if in_code:
                    code_lines.append(line)
            
            return "\n".join(code_lines) if code_lines else text
    
    def summarize(self, text: str) -> str:
        """
        Program-Aided Language Models 방식으로 텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            
        Returns:
            str: 요약된 텍스트
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        # 1단계: 요약 프로그램 생성
        program_result = self.program_chain.run(text=text)
        program_code = self._extract_python_code(program_result)
        
        # 2단계: 프로그램 로직 기반 요약 실행
        summary = self.execution_chain.run(text=text, program=program_code)
        
        return summary.strip()