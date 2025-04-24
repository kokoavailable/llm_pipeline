"""
뉴스 데이터 크롤링 모듈
"""

import os
import time
import json
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import pandas as pd
import yaml
import sys
import urllib.parse

import time

from ..utils.helpers import init_session_with_cookies

# # 프로젝트 루트 경로 추가 (상대 경로 임포트를 위한 설정)
# project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# 모듈화된 코드 임포트
from ..utils.helpers import logger


# 로깅 설정




class NewsCrawler:
    """관련 뉴스를 크롤링하는 클래스입니다."""

    def __init__(self, config_path=None):
        """
        NewsCrawler 초기화

        Args:
            config_path (str): 설정 파일 경로
        """
        # 기본 설정

        app_env = os.getenv('APP_ENV', 'DEV')

        if config_path is None:
            config_path = f"config/crawler_config.{app_env}.yaml"

        self.config = self._load_config(config_path)

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        self.sessions = {
            "mk": requests.Session(),
        }


    def _load_config(self, config_path):
        """설정 파일 로드"""
        default_config = {
            'search_keyword': ['KDX한국데이터거래소'],
            'news_source': ['mk'],
            'max_articles_per_source': 10,
            'date_range_days': 30
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return {**default_config, **config}
        
        return default_config
    
    ### 뉴스 크롤링 메소드 ###
    def crawl_mk_news(self, keyword, start_date=None, end_date= None, max_articles=None):
        """
        MK 뉴스 크롤링

        Args:
            keyword (str): 검색 키워드
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
            max_articles (int): 최대 기사 수, None이면 전체 크롤링

        Returns:
            list: 기사 정보(URL, HTML) 목록
        """
        logger.info(f"MK 뉴스 크롤링 시작: {keyword}")

        BASE_URL = 'https://www.mk.co.kr/'
        SEARCH_URL = f'{BASE_URL}search/'

        # 날짜 파라미터 설정
        date_params = {}
        if start_date:
            date_params['startDate'] = start_date.replace('-', '')
        if end_date:
            date_params['endDate'] = end_date.replace('-', '')

        # 세션 준비
        mk_session = self.sessions.get("mk")
        mk_session.headers.update(self.headers)

        if not mk_session:
            mk_session = requests.Session()
            mk_session.headers.update(self.headers)
            self.sessions["mk"] = mk_session



        # 1) 첫 페이지 접속해서 API 경로와 총 기사 수 확인
        search_params = {"word": keyword, **date_params}
        try:
            response = mk_session.get(SEARCH_URL, params=search_params, timeout=10)
            logger.debug(f"[요청] 검색 URL: {response.url}")
            logger.debug(f"[응답] 상태 코드: {response.status_code}, 응답 길이: {len(response.text)}")
            response.raise_for_status()
            html = response.text
            soup = BeautifulSoup(html, "lxml")

            # API 경로 추출
            api_button = soup.select_one("button[data-source-selector]")
            logger.debug(f"api_button: {api_button}")
            if not api_button:
                logger.error("API 버튼을 찾을 수 없습니다.")
                return []
                
            selector = api_button.get("data-source-selector", "")
            logger.debug(f"[2] 추출된 selector: {selector}")
            
            api_input = soup.select_one(selector)
            logger.debug(f"[3] 추출된 api_input: {api_input}")
            if not api_input:
                logger.error(f"API 입력({selector})을 찾을 수 없습니다.")
                return []
                
            api_path = api_input.get("value", "")
            api_path = api_path.replace("//www.mk.co.kr/", "")  # 결과: "_CP/243"
            logger.debug(f"[4] 추출된 api_path: {api_path}")

            # # 총 기사 수 확인
            # input_tag = soup.select_one("input#api_243")
            # total_count_str = input_tag.get("data-total") if input_tag else "0"
            # logger.debug(f"[5] 추출된 input_tag: {input_tag}")
            # logger.debug(f"[5] 추출된 total_count_str: {total_count_str}")
            # total_count = int(total_count_str.replace(",", "")) if total_count_str else 0

            # if max_articles is None or max_articles > total_count:
            #     max_articles = total_count
                
            # logger.info(f"총 기사 수: {total_count}, 수집 목표: {max_articles}")
            
        except Exception as e:
            logger.error(f"첫 페이지 접속 오류: {str(e)}")
            return []
        
        # 2) page=1부터 순차적으로 호출하여 기사 URL 수집
        article_data = []
        links = []
        page = 1

        headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
        "Referer": f"{BASE_URL}search?word={urllib.parse.quote(keyword)}",
        "X-Requested-With": "XMLHttpRequest",  # 비동기 요청처럼 보이게!
        "Accept": "application/json, text/javascript, */*; q=0.01",
        }

        # 쿠키 세션
        cookie_url = f"{SEARCH_URL}?word={urllib.parse.quote(keyword)}"
        init_session_with_cookies(mk_session, cookie_url)

        # 비동기 요청용 헤더
        mk_session.headers.update(self.headers)

        while True:
            try:
                api_params = {
                    "word": keyword,
                    "sort": "desc",
                    "dateType": "all",
                    "searchField": "all",
                    "newsType": "all",
                    "page": page,
                    "highlight": "Y",
                    "page_size": "null",           
                    "id": "null"
                    # **date_params
                }
                

                logger.debug(f"API 호출: {BASE_URL}{api_path}, 페이지: {page}")
                response = mk_session.get(
                    f"{BASE_URL}{api_path}",
                    headers=headers,
                    params=api_params,
                    timeout=10
                )
                response.raise_for_status()

                items_soup = BeautifulSoup(response.text, "lxml")
                items = items_soup.select("li.news_node a.news_item")

                if not items:
                    logger.info(f"페이지 {page}에서 더 이상 기사가 없습니다.")
                    break
                
                # URL 추출 및 정규화
                for a in items:
                    url = a.get("href", "")
                    if url:
                        if url.startswith('//'):
                            url = 'https:' + url
                        elif not url.startswith(('http://', 'https://')):
                            url = urllib.parse.urljoin(BASE_URL, url)

                        links.append(url)
                        # if len(links) >= max_articles:
                        #     break

                page += 1
                time.sleep(0.5)
                

            except Exception as e:
                logger.error(f"API 호출 오류(페이지 {page}): {str(e)}")
                break

        # 3) 각 기사 URL에서 HTML 수집        
        for i, url in enumerate(links):
            try:
                logger.debug(f"기사 HTML 수집 ({i+1}/{len(links)}): {url}")
                response = mk_session.get(url, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "lxml")


                # ───── ① 제목 ─────
                #   <h2 class="news_ttl">, og:title 둘 다 대비
                title = (
                    soup.select_one("h2.news_ttl")           or
                    soup.select_one("meta[property='og:title']")
                )
                title = title.get_text(strip=True) if title else ""

                # ───── ② 본문 ─────
                #   본문은 <div class="news_cnt_detail_wrap"> 안의 <p> 태그들이므로
                body_wrap = soup.select_one("div.news_cnt_detail_wrap")
                if body_wrap:
                    # 광고 삭제
                    for ad in body_wrap.select("div.ad_wrap, .ad_wide"):
                        ad.decompose()
                
                    paragraphs = body_wrap.find_all("p") if body_wrap else []
                    content = "\n".join(p.get_text(strip=True) for p in paragraphs)
                else:
                    content = ""
                    logger.warning(f"[본문 없음] {url}")
                    logger.debug(f"[응답 HTML 일부]\n{soup.prettify()[:500]}")

                # ───── ③ 날짜 ─────
                date = soup.select_one("meta[property='article:published_time']")
                date = date.get("content", "").strip() if date and date.has_attr("content") \
                    else date.get_text(strip=True)   if date else ""

                # ───── ④ 결과 누적 ─────
                article_data.append(
                    {
                        "title":   title,
                        "content": content,
                        "date":    date,
                        "url":     url,
                        "source":  "매일경제",
                    }
                )

                time.sleep(0.3)

            except Exception as e:
                logger.error(f"기사 HTML 수집 오류({url}): {str(e)}")

        logger.info(f"매일경제 뉴스 크롤링 완료: {len(article_data)}개 기사 수집")
        return article_data

    def crawl_all_sources(self):
        all_articles = []

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['date_range_days'])

        for keyword in self.config['search_keyword']:
            for source in self.config['news_source']:
                articles = []
                if source == 'mk':
                    articles = self.crawl_mk_news(
                        keyword=keyword,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        max_articles=self.config['max_articles_per_source']
                    )
                    all_articles.extend(articles)
                

        return all_articles



    def save_articles(self, articles, output_path):
        """
        크롤링한 기사를 JSON 파일로 저장

        Args:
            articles (list): 크롤링한 기사 리스트
            output_path (str): 저장할 파일 경로
        """
        df = pd.DataFrame(articles)

        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # CSV 파일로 저장
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        # JSON 파일로 저장
        json_path = output_path.replace('.csv', '.json')
        df.to_json(json_path, orient='records', lines=True, force_ascii=False) # 레코드 형식으로 읽어서, 이터러블한 객체 line, 한글을 읽을 수 있는 아스키 비활성

        logger.info(f"기사 {len(articles)}개 저장 완료: {output_path}")

    def run(self, output_path='data/raw/news_articles.csv'):
        """
        뉴스 크롤링 실행

        Args:
            output_path (str): 저장할 파일 경로

        Returns:
            pd.DataFrame: 크롤링한 기사 데이터프레임
        """
        articles = self.crawl_all_sources()
        self.save_articles(articles, output_path)
        return pd.DataFrame(articles)