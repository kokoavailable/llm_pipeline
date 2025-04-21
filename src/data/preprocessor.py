import re
import logging
from bs4 import BeautifulSoup

class NewsPreprocessor:
    """크롤링한 raw 뉴스 데이터를 전처리하는 클래스"""
    def __init__(self):
        """뉴스 전처리 클래스 초기화"""
        pass
    def process_mk_articles(self, article_data_list):
        """
        MK 뉴스 기사 전처리

        Args:
            article_data_list (list): 크롤러에서 수집한 URL/HTML 데이터 리스트

        Returns:
            list: 전처리된 기사 데이터 목록
        """
        processed_articles = []
        for article_data in article_data_list:
            try:
                soup = BeautifulSoup(article_data['html'], 'html.parser')
                title = soup.find('h1').get_text(strip=True)
                content = soup.find('div', class_='article').get_text(strip=True)
                date = soup.find('span', class_='date').get_text(strip=True)

                # 불필요한 HTML 태그 제거
                content = re.sub(r'<[^>]+>', '', content)

                processed_articles.append({
                    'title': title,
                    'content': content,
                    'date': date,
                    'url': article_data['url']
                })
            except Exception as e:
                logging.error(f"Error processing article: {e}")
        
        return processed_articles