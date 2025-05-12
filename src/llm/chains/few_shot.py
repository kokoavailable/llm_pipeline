"""
Few-shot 프롬프팅 기법을 사용한 요약 체인 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from typing import List, Dict, Any, Optional

class FewShotSummarizer:
    """Few-shot 접근법을 사용한 요약 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=300):
        """
        Few-shot 요약기 초기화
        
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
        
        # Few-shot 예제 정의
        self.examples = [
            {
                "text": "코로나19 팬데믹이 시작된 이후 3년 만에 마스크 착용 의무가 전면 해제되었다. 방역당국은 오늘 0시부터 실내에서도 마스크를 쓰지 않아도 된다고 발표했다. 다만 병원과 약국, 요양시설 등 고위험군이 많은 시설에서는 여전히 마스크 착용이 필요하다. 이번 조치는 세계보건기구(WHO)가 코로나19에 대한 국제 비상사태 종료를 선언한 데 따른 것이다.",
                "summary": "코로나19 팬데믹 이후 3년 만에 실내 마스크 착용 의무가 전면 해제되었다. 방역당국은 0시부터 실내에서도 마스크 의무를 해제한다고 발표했으나, 병원과 약국 등 고위험군이 많은 시설에서는 여전히 착용이 필요하다. 이번 조치는 WHO의 코로나19 국제 비상사태 종료 선언에 따른 결정이다."
            },
            {
                "text": "애플이 새로운 아이폰15 시리즈를 공개했다. 이번 모델은 USB-C 포트를 채택했으며, 프로 모델은 티타늄 프레임과 더 얇은 베젤을 특징으로 한다. 또한 카메라 성능이 대폭 향상되어 4800만 화소 메인 카메라와 5배 광학 줌을 지원한다. 가격은 전작과 비슷한 수준으로 책정되었으며, 다음 주부터 전 세계에 순차적으로 출시될 예정이다.",
                "summary": "애플이 USB-C 포트를 채택한 아이폰15 시리즈를 발표했다. 프로 모델은 티타늄 프레임과 더 얇은 베젤을 갖추고 있으며, 4800만 화소 메인 카메라와 5배 광학 줌으로 카메라 성능이 크게 개선되었다. 가격은 전작과 유사하게 책정되었으며 다음 주부터 글로벌 출시가 시작된다."
            }
        ]
        
        # 예제 형식 정의
        self.example_formatter_template = """
        텍스트: {text}
        
        요약: {summary}
        """
        
        self.example_prompt = PromptTemplate(
            input_variables=["text", "summary"],
            template=self.example_formatter_template
        )
        
        # Few-shot 프롬프트 템플릿 정의
        self.few_shot_prompt = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=self.example_prompt,
            prefix="다음은 뉴스 기사와 그 요약의 예시입니다. 요약은 불렛포인트 없이 자연스러운 문단 형태로 작성되어 있습니다:",
            suffix="텍스트: {text}\n\n요약:",
            input_variables=["text"],
            example_separator="\n\n"
        )
        
        # 체인 설정
        self.chain = LLMChain(llm=self.chat_model, prompt=self.few_shot_prompt)
    
    def summarize(self, text: str) -> str:
        """
        Few-shot 방식으로 텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            
        Returns:
            str: 요약된 텍스트
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        result = self.chain.run(text=text)
        return result.strip()