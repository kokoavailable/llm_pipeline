# 뉴스 요약 & 주가 예측 & 프롬프트 엔지니어링 파이프라인

이 프로젝트는 다양한 뉴스 매체에서 데이터를 수집하고, 자연어 처리(NLP) 모델을 활용하여 뉴스 내용을 요약하며, 분석 결과를 시각적으로 제시하는 자동화된 데이터 파이프라인입니다. airflow 를 적용하여 전 자동화를 달성하였습니다.
GRU 모델을 통해 주가 시퀀스 데이터를 학습하여 주가를 예측하였습니다.

코드에 대한 시연은 notebooks 폴더 하의 주피터 노트북 파일에서 가능합니다

## 프로젝트 개요

다음과 같은 엔드 투 엔드 데이터 파이프라인을 구현하였습니다.

1. 다양한 소스로부터 뉴스 기사를 크롤링 한다.
2. 로우 텍스트 데이터를 전처리 한다.
3. 전처리한 텍스트로 NLP(LLM)을 활용해 본문 요약과 키워드 요약을 한다.
4. 생성한 처리 데이터를 가지고 시각화한다.

아파치 에어플로우를 통해 해당파이프라인의 스케줄링과 의존성 관리 등을 자동화하였습니다.

## 디렉토리 구조
```
├── airflow # 스케줄링 airflow
│   ├── airflow
│   │   ├── airflow-webserver.pid
│   │   ├── airflow.cfg
│   │   ├── airflow.db
│   │   └── dags
│   │       ├── news_summary_dag.py
│   │       └── stock_prediction_dag.py
│   ├── airflow.db
│   ├── plugins
│   └── webserver_config.py
├── config
│   ├── crawler_config.DEV.yaml
│   └── model_config.yaml
├── data
│   ├── processed
│   │   └── processed_news_articles.csv
│   ├── raw
│   │   ├── news_articles.csv
│   │   ├── news_articles.json
│   │   ├── news_summaries.csv
│   │   ├── pipeline_summary.json
│   │   ├── processed_news_articles.csv
│   │   └── processed_news_articles.json
│   ├── runs
│   ├── summaries
│   │   ├── news_summaries.csv
│   │   ├── news_summaries1.csv
│   │   └── news_summaries1.json
│   └── visualizations
│       ├── keywords_wordcloud.png
│       ├── monthly_articles.png
│       └── source_distribution.png
├── dev_run.py # 메인 실행용 스크립트
├── docs
├── jobs
├── logs
├── notebooks
│   ├── kdx_news_pipeline.ipynb
│   ├── pipeline_flowchart
│   └── pipeline_flowchart.png
├── requirements.txt
├── scheduler.log
├── setup.py
├── src
│   ├── data
│   │   ├── news
│   │   │   ├── crawler.py
│   │   │   └── preprocessor.py
│   │   └── stock
│   │       ├── loader.py
│   │       └── preprocessor.py
│   ├── llm
│   │   └── gpt_summarizer.py
│   ├── models
│   │   ├── gru_model.py
│   │   └── trainer.py
│   ├── pipelines
│   │   ├── news_pipeline.py
│   │   └── stock_pipeline.py
│   ├── utils
│   │   ├── helpers.py
│   │   └── stopwords_kr.py
│   └── visualization
│       ├── news
│       │   ├── fonts
│       │   │   └── NanumGothic-Bold.ttf
│       │   └── plotter.py
│       └── stock
│           └── plotter.py
├── tests
│   └── test.py
```
# 설치

파이썬 3.9 이상
Apache Airflow 

## 필요 구성
```
annotated-types==0.7.0
anyio==4.9.0
appnope==0.1.4
asttokens==3.0.0
attrs==25.3.0
beautifulsoup4==4.13.4
bs4==0.0.2
certifi==2025.1.31
charset-normalizer==3.4.1
comm==0.2.2
contourpy==1.3.0
cycler==0.12.1
debugpy==1.8.14
decorator==5.2.1
distro==1.9.0
dotenv==0.9.9
exceptiongroup==1.2.2
executing==2.2.0
fonttools==4.57.0
h11==0.14.0
httpcore==1.0.8
httpx==0.28.1
idna==3.10
importlib_metadata==8.6.1
importlib_resources==6.5.2
ipykernel==6.29.5
ipython==8.18.1
jedi==0.19.2
jiter==0.9.0
jupyter_client==8.6.3
jupyter_core==5.7.2
kiwisolver==1.4.7
lxml==5.4.0
matplotlib==3.9.4
matplotlib-inline==0.1.7
nest-asyncio==1.6.0
numpy==2.0.2
openai==1.76.0
outcome==1.3.0.post0
packaging==25.0
pandas==2.2.3
parso==0.8.4
pexpect==4.9.0
pillow==11.2.1
platformdirs==4.3.7
prompt_toolkit==3.0.51
psutil==7.0.0
ptyprocess==0.7.0
pure_eval==0.2.3
pydantic==2.11.3
pydantic_core==2.33.1
Pygments==2.19.1
pyparsing==3.2.3
PySocks==1.7.1
python-dateutil==2.9.0.post0
python-dotenv==1.1.0
pytz==2025.2
PyYAML==6.0.2
pyzmq==26.4.0
requests==2.32.3
seaborn==0.13.2
selenium==4.31.0
six==1.17.0
sniffio==1.3.1
sortedcontainers==2.4.0
soupsieve==2.7
stack-data==0.6.3
tornado==6.4.2
tqdm==4.67.1
traitlets==5.14.3
trio==0.30.0
trio-websocket==0.12.2
typing-inspection==0.4.0
typing_extensions==4.13.2
tzdata==2025.2
urllib3==2.4.0
wcwidth==0.2.13
webdriver-manager==4.0.2
websocket-client==1.8.0
wordcloud==1.9.4
wsproto==1.2.0
zipp==3.21.0
```

# 설정
1. 레포지토리 클론
```
bashgit clone https://github.com/kokoavailable/llm_pipeline.git
cd llm_pipeline
```

2. 파이썬 가상환경 생성과 사용
```
python -m venv llm_pipeline
source llm_pipeline/bin/activate  # On Windows: venv\Scripts\activate
```

3. 필요 의존성 설치
```
pip install -r requirements.txt
```


# 사용
```
llm_pipeline/notebooks/
```
주피터 노트북 또는 IPython 환경에서 순차적으로 실행합니다.

1. 환경설정
2. 출력 디렉토리 설정
3. 뉴스 기사 수집
4. 데이터 전처리
5. 뉴스 요약 생성
6. 시각화 생성
7. 주요 시각화 결과 확인
8. 파이프라인 실행 요약 저장


# 설정


config/ 디렉토리.

crawler_config.DEV.yaml
model_config.yaml

# 구성요소

데이터 수집
src/data/news/crawler.py
src/data/stock/loader.py 

로우 데이터 전처리 
src/data/news/preprocessor.py
src/data/stock/preprocessor.py

ML/DL 모델
src/data/models/gru_model.py
src/data/models/trainer.py

데이터 요약
src/llm/gpt_summarizer.py

데이터 시각화
src/visualization/news/plotter.py
src/visualization/stock/plotter.py

설정
crawler_config.DEV.yaml
model_config.yaml

# 개발
## 추가  개발

크롤러 사이트를 확장하고 싶다면 디렉토리에 코드를 추가하여 사용하시면 됩니다.
crawler_config.DEV.yaml 설정 추가

# 테스트
pytest tests/