"""
RAG(Retrieval-Augmented Generation) 기반 요약 구현
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from typing import List, Dict, Any, Optional
import pandas as pd

class RAGSummarizer:
    """RAG(Retrieval-Augmented Generation) 접근법을 사용한 요약 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, max_tokens=300):
        """
        RAG 요약기 초기화
        
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
        
        # 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings()
        
        # 텍스트 스플리터 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # RAG 프롬프트 템플릿 정의
        self.template = """
        당신은 뉴스 기사를 요약하는 AI 도우미입니다.
        
        다음 뉴스 기사와 관련 맥락을 기반으로 핵심 내용을 3-4문장으로 요약해주세요.
        요약은 불렛포인트 없이 자연스러운 문단 형태로 작성해주세요.
        
        뉴스 기사:
        {text}
        
        관련 맥락:
        {context}
        
        요약:
        """
        
        self.prompt = PromptTemplate(
            input_variables=["text", "context"],
            template=self.template
        )
        
        # 체인 설정
        self.chain = LLMChain(llm=self.chat_model, prompt=self.prompt)
    
    def create_knowledge_base(self, texts: List[str]) -> None:
        """
        지식 베이스 생성
        
        Args:
            texts (List[str]): 지식 베이스에 추가할 텍스트 목록
        """
        # 텍스트 분할
        all_chunks = []
        for text in texts:
            chunks = self.text_splitter.split_text(text)
            all_chunks.extend(chunks)
        
        # 벡터 스토어 생성
        self.vectorstore = FAISS.from_texts(all_chunks, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def create_knowledge_base_from_df(self, df: pd.DataFrame, column_name: str) -> None:
        """
        데이터프레임에서 지식 베이스 생성
        
        Args:
            df (pd.DataFrame): 소스 데이터프레임
            column_name (str): 텍스트 데이터가 있는 컬럼 이름
        """
        texts = df[column_name].dropna().tolist()
        self.create_knowledge_base(texts)
    
    def summarize(self, text: str) -> str:
        """
        RAG 방식으로 텍스트 요약
        
        Args:
            text (str): 요약할 텍스트
            
        Returns:
            str: 요약된 텍스트
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        if not hasattr(self, 'retriever'):
            # 지식 베이스가 없는 경우, 텍스트만으로 요약
            return self.chat_model.predict(f"다음 뉴스 기사를 3-4문장으로 요약해주세요: {text}")
        
        # 관련 맥락 검색
        retrieved_docs = self.retriever.get_relevant_documents(text)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # RAG 기반 요약 생성
        result = self.chain.run(text=text, context=context)
        return result.strip()
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str, 
                          use_kb: bool = True) -> pd.DataFrame:
        """
        DataFrame의 각 기사에 대해 RAG 기반 요약 수행
        
        Args:
            df (pd.DataFrame): 기사 데이터가 포함된 DataFrame
            text_column (str): 텍스트 데이터가 있는 컬럼 이름
            use_kb (bool): 지식 베이스 사용 여부
            
        Returns:
            pd.DataFrame: 요약이 추가된 DataFrame
        """
        # 지식 베이스 생성 (필요한 경우)
        if use_kb and not hasattr(self, 'retriever'):
            self.create_knowledge_base_from_df(df, text_column)
        
        # 결과 컬럼 추가
        df['rag_summary'] = ""
        
        # 각 행에 대해 요약 수행
        for idx, row in df.iterrows():
            content = row[text_column]
            if isinstance(content, str) and content.strip():
                summary = self.summarize(content)
                df.at[idx, 'rag_summary'] = summary
        
        return df