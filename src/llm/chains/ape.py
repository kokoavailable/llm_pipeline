"""
Automatic Prompt Engineer (APE) 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional
import numpy as np

class AutomaticPromptEngineerSummarizer:
    """Automatic Prompt Engineer 접근법을 사용한 요약 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=500, generations=3):
        """
        APE 요약기 초기화
        
        Args:
            model_name (str): 사용할 OpenAI 모델
            temperature (float): 생성 다양성 조절 (0~1)
            max_tokens (int): 최대 토큰 수
            generations (int): 생성할 프롬프트 변형 수
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.generations = generations
        
        # 모델 초기화
        self.chat_model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # APE 프롬프트 생성 템플릿
        self.prompt_generation_template = """
        당신은 언어 모델의 성능을 최적화하는 프롬프트 엔지니어입니다.
        뉴스 기사를 효과적으로 요약하는 최적의 프롬프트를 만들어주세요.
        
        요약 작업의 목표:
        - 객관적이고 정확한 요약을 3-4문장으로 작성
        - 핵심 정보만 포함하고 불필요한 세부사항 제외
        - 자연스러운 문단 형태로 작성 (불렛포인트 없음)
        
        예시 기사:
        {text}
        
        다음 형식으로 {num_generations}개의 서로 다른 프롬프트 변형을 생성해주세요:
        
        프롬프트 1: [프롬프트 내용]
        프롬프트 2: [프롬프트 내용]
        ...
        
        각 프롬프트는 독특한 접근 방식이나 전략을 취해야 합니다.
        """
        
        self.prompt_generation_prompt = PromptTemplate(
            input_variables=["text", "num_generations"],
            template=self.prompt_generation_template
        )
        
        # 프롬프트 평가 템플릿
        self.evaluation_template = """
        다음 뉴스 기사를 읽고, 주어진 프롬프트를 사용하여 생성된 요약의 품질을 1-10 점수로 평가해주세요.
        
        뉴스 기사:
        {text}
        
        프롬프트:
        {prompt}
        
        생성된 요약:
        {summary}
        
        평가 기준:
        - 정확성: 요약이 기사의 주요 내용을 정확하게 포함하는가?
        - 간결성: 요약이 불필요한 정보 없이 핵심만 포함하는가?
        - 완전성: 요약이 기사의 중요한 측면을 모두 다루는가?
        - 자연스러움: 요약이 자연스럽고 읽기 쉬운가?
        
        점수(1-10):
        """
        
        self.evaluation_prompt = PromptTemplate(
            input_variables=["text", "prompt", "summary"],
            template=self.evaluation_template
        )
        
        # 체인 설정
        self.prompt_generation_chain = LLMChain(llm=self.chat_model, prompt=self.prompt_generation_prompt)
        self.evaluation_chain = LLMChain(llm=self.chat_model, prompt=self.evaluation_prompt)
    
    def _parse_prompts(self, generation_output: str) -> List[str]:
        """
        프롬프트 생성 출력에서 각 프롬프트 추출
        
        Args:
            generation_output (str): 프롬프트 생성 출력
            
        Returns:
            List[str]: 추출된 프롬프트 목록
        """
        prompts = []
        lines = generation_output.strip().split('\n')
        
        current_prompt = ""
        for line in lines:
            if line.startswith("프롬프트") and ":" in line:
                if current_prompt:
                    prompts.append(current_prompt.strip())
                current_prompt = line.split(":", 1)[1].strip()
            elif current_prompt:
                current_prompt += " " + line.strip()
        
        if current_prompt:
            prompts.append(current_prompt.strip())
        
        return prompts
    
    def _extract_score(self, evaluation_output: str) -> float:
        """
        평가 출력에서 점수 추출
        
        Args:
            evaluation_output (str): 평가 출력
            
        Returns:
            float: 추출된 점수
        """
        try:
            # 마지막 줄에서 숫자 추출
            lines = evaluation_output.strip().split('\n')
            for line in reversed(lines):
                if "점수" in line and ":" in line:
                    score_text = line.split(":", 1)[1].strip()
                    return float(score_text)
                
                # 숫자만 있는 경우
                if line.strip().isdigit():
                    return float(line.strip())
            
            # 텍스트에서 숫자 추출
            import re
            numbers = re.findall(r'\d+', evaluation_output)
            if numbers:
                return float(numbers[-1])
            
            return 5.0  # 기본값
        except:
            return 5.0  # 오류 시 기본값
    
    def summarize(self, text: str) -> str:
        """
        Automatic Prompt Engineer 방식으로 텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            
        Returns:
            str: 요약된 텍스트
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        # 1단계: 다양한 프롬프트 생성
        generation_result = self.prompt_generation_chain.run(
            text=text[:1000],  # 예시로 앞부분만 사용
            num_generations=self.generations
        )
        
        prompts = self._parse_prompts(generation_result)
        if not prompts:
            # 프롬프트 추출 실패 시 기본 프롬프트 사용
            prompts = [
                "다음 뉴스 기사를 3-4문장으로 요약하시오. 중요한 정보만 포함하고 자연스러운 문단 형태로 작성하시오."
            ]
        
        # 2단계: 각 프롬프트로 요약 생성 및 평가
        best_summary = ""
        best_score = -1
        
        for prompt in prompts:
            # 프롬프트로 요약 생성
            summary_prompt = f"{prompt}\n\n{text}"
            summary = self.chat_model.predict(summary_prompt)
            
            # 요약 평가
            evaluation_result = self.evaluation_chain.run(
                text=text,
                prompt=prompt,
                summary=summary
            )
            
            score = self._extract_score(evaluation_result)
            
            # 최고 점수 갱신
            if score > best_score:
                best_score = score
                best_summary = summary
        
        return best_summary.strip()