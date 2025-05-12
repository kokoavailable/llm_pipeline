"""
기본 프롬프트 템플릿 관리 모듈

역할 설정

형식 제약

구조적 포맷 명시.

"""
from langchain.prompts import PromptTemplate

class BasePromptTemplates:
    """기본 프롬프트 템플릿 관리 클래스"""
    
    @staticmethod
    def get_summarization_template() -> PromptTemplate:
        """        
        [프롬프트 엔지니어링 포인트]
        - 역할 지시(Role Prompting): LLM이 '뉴스 요약 AI'로 동작하도록 명시
        - 출력 제약(Output Constraint): '3~4문장, 불렛포인트 금지' 등 명확한 출력 포맷 요구
        - 자연어 명령어 디자인: 사람이 읽기에도 자연스러운 프롬프트 작성
        """

        template = """
        당신은 뉴스 기사를 읽고, 핵심 내용을 부드럽고 자연스러운 단락으로 요약하는 AI입니다.
        요약은 객관적이며 불렛포인트 없이, 사람이 쓴 것처럼 자연스럽게 이어지는 3~4문장으로 작성되어야 합니다.

        다음은 뉴스 기사입니다. 중요한 내용을 중심으로, 자연스럽고 흐름이 끊기지 않는 단락으로 3~4문장 정도로 요약해주세요. 
        요약은 마치 사람이 쓴 것처럼 자연스럽게 이어지고, 불렛포인트나 나열식 표현 없이 작성되어야 합니다.

        기사 원문:
        {text}

        요약:
        """
        
        # 템플릿과 입력 변수 바인딩 객체 생성
        return PromptTemplate(
            input_variables=["text"],
            template=template
        )
    
    @staticmethod
    def get_keyword_extraction_template() -> PromptTemplate:
        """        
        [프롬프트 엔지니어링 포인트]
        - 역할 지시(Role Prompting): '핵심 키워드를 추출하는 AI'로 모델 역할 설정
        - 출력 제약(Output Constraint):
            - 정확히 5개 키워드 요구
            - 쉼표로 구분
            - 문장/불릿포인트/숫자 사용 금지 → 출력 파싱 용이성 향상
        - 클린한 결과를 위한 포맷 지시: "한 줄로 출력하세요" 등 명확한 기대 결과 유도
        """
        template = """
        당신은 뉴스 기사를 읽고, 핵심 키워드를 추출하는 AI입니다.
        
        다음은 뉴스 기사입니다. 이 기사에서 가장 중요한 핵심 키워드 5개를 추출해주세요.  
        - 각 키워드는 단어 또는 짧은 구문 형태여야 합니다.  
        - 키워드 간에는 쉼표(,)로 구분해주세요.  
        - 불렛포인트나 숫자 없이, 키워드만 한 줄로 출력해주세요.  
        - 문장은 작성하지 마세요.  

        기사 내용:  
        {text}

        결과:
        """
        
        # 템플릿과 입력 변수 바인딩 객체 생성
        return PromptTemplate(
            input_variables=["text"],
            template=template
        )