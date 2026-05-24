import os
import re
import json
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from newspaper import Article
from openai import OpenAI

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =====================================================
# 기본 설정
# =====================================================
st.set_page_config(
    page_title="LENS RAG | 실시간 뉴스 관점 분석",
    page_icon="⚖️",
    layout="wide"
)

NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"]
NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

REACTION_FILE = "user_reactions.csv"


# =====================================================
# 세션 상태
# =====================================================
if "articles" not in st.session_state:
    st.session_state.articles = []

if "final_df" not in st.session_state:
    st.session_state.final_df = None

if "rag_result" not in st.session_state:
    st.session_state.rag_result = None

if "query_saved" not in st.session_state:
    st.session_state.query_saved = ""

if "reaction_saved" not in st.session_state:
    st.session_state.reaction_saved = False

if "related_keywords" not in st.session_state:
    st.session_state.related_keywords = []

if "selected_related_keyword" not in st.session_state:
    st.session_state.selected_related_keyword = ""
    
if "article_reactions" not in st.session_state:
    st.session_state.article_reactions = {}

if "user_perspective_report" not in st.session_state:
    st.session_state.user_perspective_report = None


# =====================================================
# 유틸 함수
# =====================================================
def format_pub_date(date_str):
    try:
        dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
        return dt.strftime("%Y.%m.%d %H:%M")
    except Exception:
        return date_str


def get_press_name(url):
    try:
        domain = urlparse(url).netloc
        return domain.replace("www.", "")
    except Exception:
        return "언론사 정보 없음"


def remove_html(text):
    if not text:
        return ""
    return re.sub(r"<.*?>", "", text)


def clean_news_text(text):
    if not text:
        return ""

    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[가-힣]{2,4}\s?기자", "", text)
    text = re.sub(r"\[.*?기자\]", "", text)
    text = re.sub(r"무단 전재 및 재배포 금지", "", text)
    text = re.sub(r"Copyright.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"ⓒ.*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_article(text):
    if len(text.strip()) < 300:
        return False
    sentences = re.split(r"[.!?。]|다\.|요\.", text)
    return len(sentences) >= 3


def make_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    return [str(value)]


def save_reaction(query, reaction, article_count):
    data = {
        "query": query,
        "reaction": reaction,
        "article_count": article_count,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    df = pd.DataFrame([data])
    df.to_csv(
        REACTION_FILE,
        mode="a",
        header=not os.path.exists(REACTION_FILE),
        index=False,
        encoding="utf-8-sig"
    )


def load_reactions():
    if os.path.exists(REACTION_FILE):
        return pd.read_csv(REACTION_FILE)
    return pd.DataFrame(columns=["query", "reaction", "article_count", "created_at"])


def safe_json_loads(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None



def get_text_list(value, limit=None):
    """LLM 결과가 list/string/NaN 어느 형태로 와도 화면 출력용 list로 변환"""
    items = make_list(value)
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    return cleaned[:limit] if limit else cleaned


def get_perspective_cards(rag_result):
    """주제별 자동 라벨이 있으면 사용하고, 없으면 안전한 기본 라벨로 대체"""
    cards = rag_result.get("perspective_cards")
    if isinstance(cards, list) and len(cards) > 0:
        normalized = []
        for idx, card in enumerate(cards[:3]):
            if isinstance(card, dict):
                normalized.append({
                    "icon": card.get("icon", ["🛡️", "⚖️", "🏫"][idx % 3]),
                    "label": card.get("label", f"관점 {idx + 1}"),
                    "content": card.get("content", "관련 관점이 충분히 수집되지 않았습니다.")
                })
            else:
                normalized.append({
                    "icon": ["🛡️", "⚖️", "🏫"][idx % 3],
                    "label": f"관점 {idx + 1}",
                    "content": str(card)
                })
        return normalized

    perspectives = get_text_list(rag_result.get("perspective_differences"), limit=3)
    fallback_labels = ["핵심 쟁점 관점", "반대·우려 관점", "현장·사회 영향 관점"]
    fallback_icons = ["🧭", "⚖️", "👥"]
    return [
        {
            "icon": fallback_icons[i],
            "label": fallback_labels[i],
            "content": perspectives[i] if i < len(perspectives) else "관련 관점이 충분히 수집되지 않았습니다."
        }
        for i in range(3)
    ]


def keyword_counter_from_df(df, top_n=15):
    """간단한 한글/영문 키워드 빈도 추출. 불용어 제거 후 상위 단어 반환"""
    stopwords = {
        "그리고", "그러나", "하지만", "또한", "대한", "관련", "이번", "지난", "있는", "없는", "했다", "한다", "것으로",
        "기자", "뉴스", "기사", "보도", "통해", "위해", "이는", "에서", "으로", "에게", "까지", "부터", "보다", "이라", "라고",
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were"
    }
    words = []
    for text in df.get("cleaned_text", pd.Series(dtype=str)).fillna(""):
        tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", str(text))
        for token in tokens:
            token = token.strip().lower()
            if token not in stopwords and len(token) >= 2:
                words.append(token)
    if not words:
        return pd.DataFrame(columns=["키워드", "빈도"])
    counts = pd.Series(words).value_counts().head(top_n).reset_index()
    counts.columns = ["키워드", "빈도"]
    return counts


def expression_temperature(row):
    """감정적/단정적 표현 수를 기반으로 기사 표현 강도 라벨 생성"""
    emotional = len(get_text_list(row.get("emotional_expressions")))
    assertive = len(get_text_list(row.get("assertive_expressions")))
    total = emotional + assertive
    if total >= 5:
        return "🔥 표현 강도 높음", min(100, 70 + total * 5)
    if total >= 3:
        return "🌡️ 표현 강도 보통", min(100, 45 + total * 8)
    return "🧊 비교적 중립적", min(100, 25 + total * 8)


def safe_join_items(value):
    items = get_text_list(value)
    return ", ".join(items) if items else "-"

def calculate_text_diversity_score(df):
    """
    기사 본문 간 의미 차이 계산.
    TF-IDF 기반 코사인 유사도를 사용해서 기사들이 서로 얼마나 다른 내용을 다루는지 측정.
    OpenAI 임베딩보다 가볍고 로컬에서 계산 가능.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        texts = df["cleaned_text"].fillna("").astype(str).tolist()

        if len(texts) < 2:
            return 0

        vectorizer = TfidfVectorizer(max_features=1000)
        vectors = vectorizer.fit_transform(texts)

        sim_matrix = cosine_similarity(vectors)

        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarities.append(sim_matrix[i][j])

        avg_similarity = sum(similarities) / len(similarities)

        diversity_score = (1 - avg_similarity) * 100
        return max(0, min(100, diversity_score))

    except Exception:
        return 0


def calculate_overall_balance_score(final_df):
    df = final_df.copy()

    # 1. 기사 간 의미 다양성: 실제 기사 본문이 서로 얼마나 다른가
    text_diversity = calculate_text_diversity_score(df)

    # 2. 언론사 다양성: 서로 다른 출처가 많을수록 증가
    if "press" in df.columns and len(df) > 0:
        press_diversity = min(100, (df["press"].nunique() / len(df)) * 100)
    else:
        press_diversity = 0

    # 3. 톤 다양성: 중립/우려/비판/긍정/강조 등 톤이 다양할수록 증가
    if "tone_label" in df.columns:
        tone_count = df["tone_label"].fillna("중립").nunique()
        tone_diversity = min(100, (tone_count / 5) * 100)
    else:
        tone_diversity = 0

    # 4. 관점 위치 차이: 기사별 입장 위치가 얼마나 분산되어 있는가
    if "issue_position_score" in df.columns:
        position_scores = pd.to_numeric(
            df["issue_position_score"],
            errors="coerce"
        ).fillna(50)
        position_diversity = min(100, position_scores.std() * 3)
    else:
        position_diversity = 0

    # 5. 누락 관점 적절성: 빠져 있을 수 있는 관점을 포착했는지
    if "missing_perspectives" in df.columns:
        missing_avg = df["missing_perspectives"].apply(
            lambda x: len(make_list(x))
        ).mean()
        missing_score = min(100, missing_avg * 20)
    else:
        missing_score = 0

    final_score = (
    text_diversity * 0.45 +
    position_diversity * 0.20 +
    tone_diversity * 0.15 +
    press_diversity * 0.10 +
    missing_score * 0.10)

    return int(round(max(0, min(100, final_score))))


def calculate_article_balance_score(row):
    emotional_count = len(make_list(row.get("emotional_expressions")))
    assertive_count = len(make_list(row.get("assertive_expressions")))
    missing_count = len(make_list(row.get("missing_perspectives")))

    tone = str(row.get("tone_label", "중립"))

    score = 70

    # 감정·단정 표현 많으면 감소
    score -= min(emotional_count * 4, 16)
    score -= min(assertive_count * 5, 20)

    # 빠진 관점 많으면 균형성 감소
    score -= min(missing_count * 6, 18)

    # 톤 보정
    if tone == "중립":
        score += 10

    return int(max(0, min(100, round(score))))

def build_user_perspective_report(final_df, reactions):
    rows = []

    for idx, reaction in reactions.items():
        if idx < len(final_df):
            row = final_df.iloc[idx]

            rows.append({
                "reaction": reaction,
                "title": row.get("title", ""),
                "press": row.get("press", ""),
                "tone": row.get("tone_label", "중립"),
                "claim": row.get("main_claim", ""),
                "framing": row.get("framing", ""),
                "keywords": row.get("key_keywords", []),
                "missing": row.get("missing_perspectives", [])
            })

    if len(rows) == 0:
        return None

    report_df = pd.DataFrame(rows)

    agree = report_df[report_df["reaction"] == "동의"]
    disagree = report_df[report_df["reaction"] == "동의하지 않음"]
    hold = report_df[report_df["reaction"] == "판단 보류"]

    def top_tones(df):
        if len(df) == 0:
            return "-"
        return ", ".join(df["tone"].value_counts().head(3).index.tolist())

    def collect_unique(series, limit=6):
        items = []
        for value in series.tolist():
            items.extend(make_list(value))
        items = [str(x).strip() for x in items if str(x).strip()]
        return list(dict.fromkeys(items))[:limit]

    agreed_claims = agree["claim"].dropna().head(4).tolist()
    disagreed_claims = disagree["claim"].dropna().head(4).tolist()
    hold_claims = hold["claim"].dropna().head(4).tolist()

    agreed_frames = agree["framing"].dropna().head(3).tolist()
    disagreed_frames = disagree["framing"].dropna().head(3).tolist()

    agreed_keywords = collect_unique(agree["keywords"], limit=8)
    disagreed_keywords = collect_unique(disagree["keywords"], limit=8)

    missing_from_hold = collect_unique(hold["missing"], limit=6)
    missing_from_disagree = collect_unique(disagree["missing"], limit=6)

    return {
        "total": len(report_df),

        "agree_count": len(agree),
        "disagree_count": len(disagree),
        "hold_count": len(hold),

        "agree_tones": top_tones(agree),
        "disagree_tones": top_tones(disagree),
        "hold_tones": top_tones(hold),

        "agreed_claims": agreed_claims,
        "disagreed_claims": disagreed_claims,
        "hold_claims": hold_claims,

        "agreed_frames": agreed_frames,
        "disagreed_frames": disagreed_frames,

        "agreed_keywords": agreed_keywords,
        "disagreed_keywords": disagreed_keywords,

        "missing_from_hold": missing_from_hold,
        "missing_from_disagree": missing_from_disagree
    }

# =====================================================
# 네이버 뉴스 수집
# =====================================================
def search_naver_news(query, display=30, sort="sim"):
    url = "https://openapi.naver.com/v1/search/news.json"

    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }

    params = {
        "query": query,
        "display": display,
        "sort": sort
    }

    response = requests.get(url, headers=headers, params=params, timeout=10)

    if response.status_code != 200:
        st.error(f"네이버 API 오류: {response.status_code}")
        return []

    return response.json().get("items", [])


def collect_articles(query, max_articles=5):
    items = search_naver_news(query, display=40, sort="sim")

    articles = []
    seen_urls = set()

    for item in items:
        if len(articles) >= max_articles:
            break

        title = remove_html(item.get("title", ""))
        description = remove_html(item.get("description", ""))
        article_url = item.get("originallink") or item.get("link")

        if not article_url or article_url in seen_urls:
            continue

        seen_urls.add(article_url)

        try:
            article = Article(article_url, language="ko")
            article.download()
            article.parse()

            raw_text = article.text
            if not raw_text:
                continue

            cleaned_text = clean_news_text(raw_text)

            if not is_valid_article(cleaned_text):
                continue

            articles.append({
                "query": query,
                "title": article.title if article.title else title,
                "description": description,
                "url": article_url,
                "press": get_press_name(article_url),
                "pub_date": item.get("pubDate", ""),
                "cleaned_text": cleaned_text,
                "collected_at": datetime.now().strftime("%Y.%m.%d %H:%M")
            })

        except Exception:
            continue

    return articles


# =====================================================
# 실시간 뉴스 RAG 구성
# =====================================================
def create_documents_from_articles(articles):
    docs = []

    for idx, article in enumerate(articles, start=1):
        content = f"""
[기사 번호]
{idx}

[검색어]
{article['query']}

[제목]
{article['title']}

[언론사]
{article['press']}

[날짜]
{format_pub_date(article['pub_date'])}

[URL]
{article['url']}

[본문]
{article['cleaned_text']}
"""

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "article_id": idx,
                    "title": article["title"],
                    "press": article["press"],
                    "url": article["url"],
                    "pub_date": article["pub_date"]
                }
            )
        )

    return docs


def create_realtime_vectorstore(articles):
    docs = create_documents_from_articles(articles)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )

    split_docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore


def format_docs_for_prompt(docs):
    formatted = []

    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title", "제목 없음")
        press = doc.metadata.get("press", "언론사 없음")
        url = doc.metadata.get("url", "")
        content = doc.page_content[:1800]

        formatted.append(
            f"""
[검색 근거 {i}]
제목: {title}
언론사: {press}
URL: {url}
내용:
{content}
"""
        )

    return "\n\n".join(formatted)


def analyze_issue_with_rag(query, vectorstore, k=8):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(query)
    context = format_docs_for_prompt(retrieved_docs)

    prompt = ChatPromptTemplate.from_template(
        """
너는 미디어 리터러시 교육을 위한 뉴스 관점 분석 AI야.

사용자가 입력한 이슈는 다음과 같아.
[검색어]
{query}

아래는 이 검색어와 관련해 실시간 수집된 뉴스 기사들을 벡터화한 뒤,
의미적으로 관련도가 높은 내용만 검색해서 가져온 근거야.

[검색된 뉴스 근거]
{context}

중요한 원칙:
- 정치 성향을 단정하지 마.
- 기사 내용의 진위 여부를 판정하지 마.
- 특정 관점을 정답처럼 제시하지 마.
- 검색된 기사 근거 안에서만 분석해.
- 사용자가 다양한 관점에서 뉴스를 이해하도록 도와줘.
- 대답은 한국어로 작성해.

반드시 아래 JSON 형식으로만 답해.

{{
  "common_core": "여러 기사에서 공통적으로 다루는 핵심 사건 요약",
  "perspective_cards": [
    {{"icon": "🛡️", "label": "이 이슈에 맞는 관점 라벨 1", "content": "관점 설명 1"}},
    {{"icon": "⚖️", "label": "이 이슈에 맞는 관점 라벨 2", "content": "관점 설명 2"}},
    {{"icon": "🏫", "label": "이 이슈에 맞는 관점 라벨 3", "content": "관점 설명 3"}}
  ],
  "perspective_differences": [
    "관점 차이 1",
    "관점 차이 2",
    "관점 차이 3"
  ],
  "emphasis_differences": [
    "어떤 기사들은 무엇을 강조하는지 1",
    "어떤 기사들은 무엇을 강조하는지 2",
    "어떤 기사들은 무엇을 강조하는지 3"
  ],
  "missing_perspectives": [
    "빠져 있을 수 있는 관점 1",
    "빠져 있을 수 있는 관점 2",
    "빠져 있을 수 있는 관점 3"
  ],
  "thinking_questions": [
    "사용자가 생각해볼 질문 1",
    "사용자가 생각해볼 질문 2",
    "사용자가 생각해볼 질문 3"
  ],
  "one_line_comment": "이 이슈를 균형 있게 읽기 위한 한 줄 조언"
}}
"""
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    chain = prompt | llm | StrOutputParser()
    result_text = chain.invoke({"query": query, "context": context})
    result_json = safe_json_loads(result_text)

    if result_json is None:
        result_json = {
            "common_core": result_text,
            "perspective_cards": [],
            "perspective_differences": [],
            "emphasis_differences": [],
            "missing_perspectives": [],
            "thinking_questions": [],
            "overall_balance_score": 50,
            "one_line_comment": "JSON 변환에 실패했지만, 원문 분석 결과를 표시합니다."
        }

    return result_json, retrieved_docs


# =====================================================
# 기사별 LLM 분석
# =====================================================
def analyze_one_article(article):
    prompt = f"""
너는 미디어 리터러시 교육을 위한 뉴스 분석 AI야.

아래 기사 1개를 분석해줘.

중요한 원칙:
- 정치 성향을 단정하지 마.
- 기사 내용의 진위 여부를 판정하지 마.
- 특정 관점을 정답처럼 제시하지 마.
- 분석 결과는 참고 정보로 작성해.
- 사용자가 다양한 관점에서 기사를 이해하도록 도와줘.

[기사 제목]
{article['title']}

[언론사]
{article['press']}

[기사 날짜]
{format_pub_date(article['pub_date'])}

[기사 URL]
{article['url']}

[기사 본문]
{article['cleaned_text'][:3000]}

반드시 아래 JSON 형식으로만 답해.

{{
  "summary": "기사의 핵심 사건 요약",
  "main_claim": "기사에서 가장 강조하는 주장",
  "framing": "기사가 사건을 어떤 관점 중심으로 바라보는지 짧은 프레이밍 라벨만 생성. 예: 경제 성장 중심 / 갈등 중심 / 정책 효과 중심 / 피해자 관점 중심 / 산업 혁신 중심",
  "tone_label": "중립/우려/비판/긍정/강조 중 하나",
  "key_keywords": ["핵심 키워드 1", "핵심 키워드 2", "핵심 키워드 3"],
  "issue_position_score": "0~100 사이 정수. 기사 관점의 위치를 나타내는 참고 점수. 한쪽 주장만 강하게 강조하면 낮거나 높게, 여러 관점을 함께 제시하면 중간에 가깝게 평가",
  "emotional_expressions": ["감정적 표현 예시 1", "감정적 표현 예시 2"],
  "assertive_expressions": ["단정적 표현 예시 1", "단정적 표현 예시 2"],
  "missing_perspectives": ["빠져 있을 수 있는 관점 1", "빠져 있을 수 있는 관점 2"],
  "balance_score": "0~100 사이 정수. 기사 안에서 반론, 이해관계자, 누락 가능성, 표현 균형이 얼마나 드러나는지 평가",
  "one_line_comment": "사용자에게 보여줄 한 줄 설명"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "너는 뉴스 기사 표현, 프레이밍, 관점 차이를 분석하는 미디어 리터러시 AI야."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    result_text = response.choices[0].message.content
    result_json = safe_json_loads(result_text)
    

    if result_json is None:
        return {
            "summary": result_text,
            "main_claim": "",
            "framing": "",
            "tone_label": "중립",
            "key_keywords": [],
            "issue_position_score": 50,
            "emotional_expressions": [],
            "assertive_expressions": [],
            "missing_perspectives": [],
            "balance_score": 50,
            "one_line_comment": "JSON 형식 변환에 실패했습니다."
        }
    try:
        result_json["issue_position_score"] = int(float(result_json.get("issue_position_score", 50)))
    except:
        result_json["issue_position_score"] = 50
    
    try:
        result_json["balance_score"] = int(float(result_json.get("balance_score", 50)))
    except:
        result_json["balance_score"] = 50
        
    result_json["issue_position_score"] = max(0, min(100, result_json["issue_position_score"]))
    result_json["balance_score"] = max(0, min(100, result_json["balance_score"]))

    return result_json


# =====================================================
# 관련 검색어 추천
# =====================================================
def generate_related_keywords(query, max_keywords=8):
    """입력 검색어를 더 구체화할 수 있는 뉴스 검색용 관련어 추천"""
    prompt = f"""
사용자가 뉴스 관점 분석을 위해 입력한 검색어는 다음과 같습니다.

[검색어]
{query}

이 검색어와 함께 검색하면 좋은 관련 검색어를 {max_keywords}개 추천해주세요.

조건:
- 너무 넓은 단어는 피하고, 실제 뉴스 검색에 적합한 구체적인 표현으로 작성
- 한국어 중심으로 작성
- 서로 겹치는 키워드는 피하기
- 정치적 성향을 단정하는 표현은 피하기
- 쉼표로만 구분해서 출력

예시:
촉법소년 연령 하향, 촉법소년 처벌 강화, 소년법 개정, 청소년 범죄 예방
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 뉴스 검색 키워드 추천 전문가야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.45
        )

        text = response.choices[0].message.content.strip()
        text = text.replace("\n", ",")
        keywords = [k.strip(" -•\t") for k in text.split(",") if k.strip(" -•\t")]

        unique_keywords = []
        for keyword in keywords:
            if keyword not in unique_keywords and len(keyword) >= 2:
                unique_keywords.append(keyword)

        return unique_keywords[:max_keywords]

    except Exception as e:
        st.warning(f"관련 검색어 생성 중 오류가 발생했습니다: {e}")
        return []


# =====================================================
# UI/UX 강화 CSS
# =====================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Pretendard', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 55%, #020617 100%);
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.hero-container {
    padding: 2.5rem 1rem 1.8rem 1rem;
    text-align: center;
}

.hero-title {
    font-size: 64px;
    font-weight: 900;
    background: linear-gradient(90deg, #60a5fa, #38bdf8, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -3px;
    margin-bottom: 10px;
}

.hero-copy {
    font-size: 18px;
    color: #cbd5e1;
    font-weight: 500;
    line-height: 1.7;
}

.notice-box {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 20px;
    padding: 18px 22px;
    margin-bottom: 22px;
    color: #d1d5db;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.20);
    backdrop-filter: blur(12px);
}

.card, .summary-card, .mini-card {
    background: rgba(30, 41, 59, 0.62);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 22px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.20);
    backdrop-filter: blur(14px);
}

.card-title {
    font-size: 19px;
    font-weight: 850;
    margin-bottom: 12px;
    color: #f8fafc;
}

.meta-text {
    color: #cbd5e1;
    font-size: 14px;
    line-height: 1.8;
}

.summary-card-title {
    font-size: 22px;
    font-weight: 850;
    margin-bottom: 12px;
    color: #93c5fd;
}

.summary-card-body {
    font-size: 16px;
    line-height: 1.85;
    color: #e5e7eb;
}

.score-badge {
    display: inline-block;
    background: rgba(37, 99, 235, 0.25);
    color: #dbeafe;
    border: 1px solid rgba(96, 165, 250, 0.35);
    padding: 7px 13px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 800;
    margin: 6px 0 14px 0;
}

.article-link {
    color: #93c5fd !important;
    text-decoration: none !important;
    font-weight: 800;
}

.stButton > button {
    background: linear-gradient(90deg, #2563eb, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.7rem 1.5rem !important;
    font-weight: 800 !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.32);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(14, 165, 233, 0.38);
}

[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px;
    padding: 16px;
}

[data-testid="stMetricValue"] {
    font-weight: 900 !important;
    color: #60a5fa !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    height: 46px;
    background-color: rgba(255, 255, 255, 0.06);
    border-radius: 12px 12px 0 0;
    color: #cbd5e1;
    padding: 0 18px;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    background-color: rgba(37, 99, 235, 0.22) !important;
    color: #ffffff !important;
    border-bottom: 2px solid #60a5fa !important;
}

.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #60a5fa, #38bdf8) !important;
}

.related-box {
    background: rgba(15, 23, 42, 0.58);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 16px 18px;
    margin: 8px 0 20px 0;
}

.related-title {
    color: #bfdbfe;
    font-weight: 850;
    margin-bottom: 6px;
}

.related-desc {
    color: #cbd5e1;
    font-size: 0.92rem;
}

/* text_input, slider 라벨 흰색 */
.stTextInput label,
.stSlider label {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* slider 숫자도 같이 밝게 */
.stSlider span {
    color: white !important;
}

/* metric 제목 흰색 */
[data-testid="stMetricLabel"] {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* metric 값 (하늘색 유지) */
[data-testid="stMetricValue"] {
    color: #60a5fa !important;
    font-weight: 900 !important;
}

/* ==============================
   전체 텍스트 가독성 보정
============================== */

/* 기본 텍스트 */
.stMarkdown, .stMarkdown p, .stMarkdown span,
.stWrite, p, span, div {
    color: #e5e7eb;
}

/* 제목류 */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* caption 설명문 */
[data-testid="stCaptionContainer"] {
    color: #cbd5e1 !important;
}

/* expander 내부 텍스트 */
[data-testid="stExpander"] * {
    color: #e5e7eb !important;
}

/* st.text()로 출력되는 기사 원문 */
[data-testid="stText"] {
    color: #e5e7eb !important;
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 14px;
    padding: 16px;
    line-height: 1.8;
}

/* Plotly 축/설명 근처 Streamlit 텍스트 보정 */
.js-plotly-plot text {
    fill: #cbd5e1 !important;
}

/* selectbox, radio 라벨 */
.stSelectbox label,
.stRadio label {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* info/success/warning 박스 내부 글씨 */
[data-testid="stAlert"] * {
    color: #f8fafc !important;
}

</style>
""",
unsafe_allow_html=True
)


# =====================================================
# 상단 화면
# =====================================================
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">⚖️ LENS RAG</div>
        <div class="hero-copy">
            실시간 뉴스 수집 · 즉시 벡터화 · RAG 기반 관점 분석 서비스<br>
            같은 이슈를 여러 기사에서 비교해 균형 있게 읽도록 돕습니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="notice-box">
        <b>📌 분석 안내</b><br><br>
        • 검색어를 입력하면 네이버 뉴스 API에서 관련 주요 뉴스를 실시간으로 수집합니다.<br>
        • 수집된 기사 본문은 전처리 후 chunk 단위로 분할되고 OpenAI Embedding으로 벡터화됩니다.<br>
        • FAISS 벡터DB에서 관련 높은 기사 내용을 검색한 뒤, LLM이 근거 기반 관점 분석을 생성합니다.<br>
        • 분석 결과는 참고용이며, 기사 내용의 사실 여부나 정치적 성향을 확정하지 않습니다.
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# 검색 영역
# =====================================================
search_col1, search_col2 = st.columns([3, 1])

with search_col1:
    query = st.text_input(
        "검색할 사건/이슈",
        value=st.session_state.selected_related_keyword,
        placeholder="예: 반도체 랠리, 촉법소년, 의대 증원, 전세사기",
        key="news_query_input"
    )

with search_col2:
    max_articles = st.slider(
        "수집 기사 수",
        min_value=3,
        max_value=7,
        value=5
    )

# =====================================================
# 관련 검색어 추천 UI
# =====================================================
keyword_col1, keyword_col2 = st.columns([1, 3])

with keyword_col1:
    recommend_button = st.button("✨ 관련 검색어 추천", use_container_width=True)

with keyword_col2:
    st.markdown(
        """
        <div class="related-box">
            <div class="related-title">검색어를 더 구체화하면 수집 기사 품질이 좋아집니다.</div>
            <div class="related-desc">예: 촉법소년 → 촉법소년 연령 하향, 소년법 개정, 청소년 범죄 예방</div>
        </div>
        """,
        unsafe_allow_html=True
    )

if recommend_button:
    if not query.strip():
        st.warning("관련 검색어를 추천받으려면 먼저 검색어를 입력해주세요.")
    else:
        with st.spinner("AI가 관련 검색어를 추천하는 중입니다..."):
            st.session_state.related_keywords = generate_related_keywords(query.strip())

if st.session_state.related_keywords:
    st.markdown("### 🔎 관련 검색어 추천")
    st.caption("버튼을 누르면 검색창에 자동으로 반영됩니다. 이후 분석 시작 버튼을 눌러주세요.")

    keyword_cols = st.columns(4)
    for i, keyword in enumerate(st.session_state.related_keywords):
        with keyword_cols[i % 4]:
            if st.button(keyword, key=f"related_keyword_button_{i}", use_container_width=True):
                st.session_state.selected_related_keyword = keyword
                st.session_state.related_keywords = st.session_state.related_keywords
                st.rerun()
    if st.session_state.selected_related_keyword:
        st.markdown(
    f"""
<div style="
background:rgba(59,130,246,0.10);
border:1px solid rgba(96,165,250,0.22);
border-radius:10px;
padding:8px 14px;
margin-top:6px;
margin-bottom:12px;
display:inline-block;
">

<span style="
color:#bfdbfe;
font-size:14px;
font-weight:600;
">
선택된 검색어:
</span>

<span style="
color:white;
font-size:14px;
font-weight:800;
">
{st.session_state.selected_related_keyword}
</span>

</div>
""",
    unsafe_allow_html=True
)

start_button = st.button("🔍 실시간 뉴스 RAG 분석 시작", use_container_width=True)


# =====================================================
# 분석 실행 - UX 개선 버전
# =====================================================
if start_button:
    query = query.strip()
    if not query:
        st.warning("🔍 분석할 키워드를 입력해 주세요.")
    else:
        st.session_state.selected_related_keyword = query
        st.session_state.rag_result = None
        st.session_state.final_df = None
        st.session_state.reaction_saved = False
        st.session_state.query_saved = query
        st.session_state.article_reactions = {}
        st.session_state.user_perspective_report = None

        with st.status("🚀 LENS AI가 실시간 뉴스를 분석하고 있습니다...", expanded=True) as status:
            st.write("📡 네이버 API에서 관련 기사를 수집 중...")
            articles = collect_articles(query, max_articles=max_articles)

            if len(articles) == 0:
                status.update(label="❌ 수집된 기사가 없습니다.", state="error", expanded=True)
                st.error("수집된 기사가 없습니다. 검색어를 바꾸거나 다시 시도해보세요.")
            else:
                st.session_state.articles = articles
                st.write(f"✅ {len(articles)}개 기사 수집 완료")

                st.write("🧠 수집 기사 본문을 벡터DB로 변환 중...")
                vectorstore = create_realtime_vectorstore(articles)

                st.write("⚖️ RAG 기반 전체 관점 분석 중...")
                rag_result, retrieved_docs = analyze_issue_with_rag(query, vectorstore, k=8)
                st.session_state.rag_result = rag_result
                st.session_state.retrieved_docs = retrieved_docs

                st.write("📝 기사별 세부 분석 중...")
                analysis_results = []
                progress = st.progress(0)

                for i, article in enumerate(articles):
                    st.write(f"- 기사 {i + 1} 분석 중: {article['press']}")
                    result = analyze_one_article(article)
                    analysis_results.append(result)
                    progress.progress((i + 1) / len(articles))

                article_df = pd.DataFrame(articles)
                analysis_df = pd.DataFrame(analysis_results)
                final_df = pd.concat([article_df.reset_index(drop=True), analysis_df.reset_index(drop=True)],axis=1)
                
                # 기사별 다양성 점수 계산 (LLM 점수 덮어쓰기)
                final_df["balance_score"] = final_df.apply(calculate_article_balance_score,axis=1)
                
                # 전체 관점 다양성 점수 계산
                calculated_score = calculate_overall_balance_score(final_df)
                st.session_state.rag_result["overall_balance_score"] = calculated_score
                st.session_state.final_df = final_df
                status.update(label="✅ 분석 완료!", state="complete", expanded=False)


# =====================================================
# 결과 화면
# =====================================================
if st.session_state.final_df is not None and st.session_state.rag_result is not None:
    final_df = st.session_state.final_df
    rag_result = st.session_state.rag_result
    query_saved = st.session_state.query_saved
    score = rag_result.get("overall_balance_score", 0)

    st.markdown("---")
    st.subheader("📊 실시간 뉴스 RAG 분석 결과")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("검색 키워드", query_saved)

    with metric_col2:
        st.metric("수집 기사 수", f"{len(final_df)}개")

    with metric_col3:
        st.metric("종합 관점 다양성 지표", f"{score}/100")

    tabs = st.tabs(["💡 RAG 종합 분석", "📰 수집 기사 목록", "🧐 기사별 상세 분석"])

    # =================================================
    # 1. RAG 종합 분석 탭 - 고도화 버전
    # =================================================
    with tabs[0]:
        st.markdown("## 🔎 RAG 기반 전체 관점 분석")

        summary_col, gauge_col = st.columns([1.55, 1])

        with summary_col:
            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="summary-card-title">핵심 요약</div>
                    <div class="summary-card-body">
                        {rag_result.get("common_core", "")}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.info(f"**AI 제언:** {rag_result.get('one_line_comment', '')}")

        with gauge_col:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": "관점 다양성 지표", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar": {"color": "#60a5fa"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 2,
                    "bordercolor": "#334155",
                    "steps": [
                        {"range": [0, 40], "color": "#7f1d1d"},
                        {"range": [40, 70], "color": "#78350f"},
                        {"range": [70, 100], "color": "#14532d"},
                    ],
                }
            ))
            gauge.update_layout(
                height=290,
                margin=dict(l=10, r=10, t=45, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f9fafb", "family": "Pretendard"}
            )
            st.plotly_chart(gauge, use_container_width=True, key="rag_overall_balance_gauge")
            st.caption("전체 관점 다양성 지표는 기사들 사이의 관점 차이·언론사 다양성·표현 톤 등을 종합 계산한 참고 지표입니다.")

        st.divider()

        # 관점 자동 라벨 카드
        st.markdown("## 🧭 주제별 관점 비교")
        st.caption("경제·정치·사회 이슈에 따라 LLM이 관점 라벨을 자동으로 구성합니다.")
        perspective_cards = get_perspective_cards(rag_result)
        p_cols = st.columns(3)
        for i, col in enumerate(p_cols):
            card = perspective_cards[i] if i < len(perspective_cards) else {"icon": "🧭", "label": f"관점 {i+1}", "content": "관련 관점이 충분히 수집되지 않았습니다."}
            with col:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-title">{card.get('icon', '🧭')} {card.get('label', f'관점 {i + 1}')}</div>
                        <div class="summary-card-body">{card.get('content', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.divider()

        # 기사별 비교표 + 키워드 차트
        st.markdown("## 📌 기사 비교 대시보드")
        dash_col1, dash_col2 = st.columns([1.35, 1])

        with dash_col1:
            compare_columns = ["press", "title", "main_claim", "tone_label", "balance_score"]
            for col_name in compare_columns:
                if col_name not in final_df.columns:
                    final_df[col_name] = "" if col_name != "balance_score" else 50

            compare_df = final_df[compare_columns].copy()
            compare_df.columns = ["언론사", "기사 제목", "핵심 주장", "톤", "다양성 점수"]
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

        with dash_col2:
            keyword_df = keyword_counter_from_df(final_df, top_n=12)
            if len(keyword_df) > 0:
                fig_keyword = px.bar(keyword_df, x="빈도", y="키워드", orientation="h", title="수집 기사 핵심 키워드")
                fig_keyword.update_layout(
                    height=360,
                    margin=dict(l=10, r=10, t=45, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#f9fafb", "family": "Pretendard"}
                )
                st.plotly_chart(fig_keyword, use_container_width=True, key="keyword_bar_chart")
            else:
                st.info("키워드를 추출할 수 없습니다.")

        # 기사별 관점 위치 스펙트럼
        if "issue_position_score" in final_df.columns:
            st.markdown("## 🧪 기사별 관점 스펙트럼")
            spectrum_df = final_df[["press", "title", "issue_position_score"]].copy()
            spectrum_df["issue_position_score"] = pd.to_numeric(spectrum_df["issue_position_score"], errors="coerce").fillna(50)
            spectrum_df["기준선"] = "기사별 위치"
            fig_spectrum = px.scatter(
                spectrum_df,
                x="issue_position_score",
                y="기준선",
                text="press",
                hover_data={"title": True, "issue_position_score": True, "기준선": False},
                range_x=[0, 100],
                title="0에 가까울수록 한쪽 관점, 100에 가까울수록 다른 관점으로 해석"
            )
            fig_spectrum.update_traces(textposition="top center", marker=dict(size=18))
            fig_spectrum.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f9fafb", "family": "Pretendard"}
            )
            st.plotly_chart(fig_spectrum, use_container_width=True, key="article_position_spectrum")

        st.divider()

        # 강조점 차이
        st.markdown("## 🧩 기사들이 강조한 내용")
        emphasis_items = get_text_list(rag_result.get("emphasis_differences"))
        for item in emphasis_items:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="summary-card-body">{item}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        missing_col, question_col = st.columns(2)

        with missing_col:
            st.markdown("## 👀 빠져 있을 수 있는 관점")
            for item in get_text_list(rag_result.get("missing_perspectives")):
                st.warning(item)

        with question_col:
            st.markdown("## ❓ 생각해볼 질문")
            for item in get_text_list(rag_result.get("thinking_questions")):
                st.info(item)

        st.divider()

        with st.expander("🔍 RAG 검색 근거 확인"):
            for i, doc in enumerate(st.session_state.retrieved_docs, start=1):
                st.markdown(f"### 근거 {i}")
                st.markdown(f"**제목:** {doc.metadata.get('title', '')}")
                st.markdown(f"**언론사:** {doc.metadata.get('press', '')}")
                st.markdown(f"**URL:** {doc.metadata.get('url', '')}")
                st.text(doc.page_content[:1200])
                st.divider()
        
        st.markdown("## 🧭 나의 관점 패턴 리포트")
        st.caption("기사별 상세 분석 탭에서 각 기사에 대한 반응을 선택하면 이곳에서 관점 패턴을 확인할 수 있습니다.")
        
        if st.button("🧭 나의 관점 패턴 분석하기", use_container_width=True):
            st.session_state.user_perspective_report = build_user_perspective_report(
                final_df,
                st.session_state.article_reactions)
            
        if st.session_state.user_perspective_report:
            report = st.session_state.user_perspective_report
            
            st.markdown("### 📊 나의 관점 패턴 리포트")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("동의", report["agree_count"])
            c2.metric("동의하지 않음", report["disagree_count"])
            c3.metric("판단 보류", report["hold_count"])
            
            st.info(
                f"""
                내가 주로 동의한 기사 톤은 **{report['agree_tones']}** 입니다.  
                거리감을 느낀 기사 톤은 **{report['disagree_tones']}** 이고,  
                판단을 보류한 기사 톤은 **{report['hold_tones']}** 입니다.
                """)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### 👍 내가 수용한 주장")
                if report["agreed_claims"]:
                    for claim in report["agreed_claims"]:
                        st.markdown(f"- {claim}")
                else:
                    st.caption("아직 동의한 기사가 없습니다.")
                st.markdown("#### 🔑 내가 반응한 핵심 키워드")
                if report["agreed_keywords"]:
                    st.write(", ".join(report["agreed_keywords"]))
                else:
                    st.caption("아직 확인된 키워드가 없습니다.")
            with col_b:
                st.markdown("#### 🤔 내가 거리감을 느낀 주장")
                if report["disagreed_claims"]:
                    for claim in report["disagreed_claims"]:
                        st.markdown(f"- {claim}")
                else:
                    st.caption("아직 동의하지 않은 기사가 없습니다.")
                    
                st.markdown("#### 🧭 거리감을 느낀 프레이밍")
                if report["disagreed_frames"]:
                    for frame in report["disagreed_frames"]:
                        st.markdown(f"- {frame}")
                else:
                    st.caption("아직 확인된 프레이밍이 없습니다.")
            st.markdown("#### 👀 추가로 읽어볼 필요가 있는 관점")
            missing_items = (report["missing_from_hold"] + 
                             report["missing_from_disagree"])
            missing_items = list(dict.fromkeys(missing_items))
            
            if missing_items:
                for item in missing_items:
                    st.markdown(f"- {item}")
            else:
                st.caption("현재 선택만으로는 추가 확인 관점이 충분히 도출되지 않았습니다.")

    # =================================================
    # 2. 수집 기사 목록 탭
    # =================================================
    with tabs[1]:
        st.markdown("## 📰 실시간 수집 기사 목록")
        st.caption("이번 분석에 실제로 사용된 기사 목록입니다.")

        for i, row in final_df.iterrows():
            st.markdown(
                f"""
                <div class="card">
                    <div style="display:flex; justify-content:space-between; gap:12px; align-items:center;">
                        <span style="color:#93c5fd; font-weight:850;">기사 {i + 1} · {row['press']}</span>
                        <span style="color:#94a3b8; font-size:0.85rem;">수집 {row['collected_at']}</span>
                    </div>
                    <div class="card-title" style="margin-top:10px;">{row['title']}</div>
                    <div class="meta-text">
                        날짜: {format_pub_date(row['pub_date'])}<br>
                        설명: {str(row.get('description', ''))[:180]}...
                    </div>
                    <br>
                    <a class="article-link" href="{row['url']}" target="_blank">원문 읽기 →</a>
                </div>
                """,
                unsafe_allow_html=True
            )

    # =================================================
    # 3. 기사별 상세 분석 탭
    # =================================================
    with tabs[2]:
        st.markdown("## 🧐 기사별 상세 관점 분석")

        selected_title = st.selectbox(
            "분석할 기사를 선택하세요",
            final_df["title"].tolist(),
            key="selected_article_title"
        )

        row = final_df[final_df["title"] == selected_title].iloc[0]
        row_idx = int(final_df[final_df["title"] == selected_title].index[0])

        left_col, main_col = st.columns([1, 2.3])

        with left_col:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">기사 정보</div>
                    <div class="meta-text">
                        <b>언론사/출처</b><br>{row['press']}<br><br>
                        <b>기사 날짜</b><br>{format_pub_date(row['pub_date'])}<br><br>
                        <b>수집 시간</b><br>{row['collected_at']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(f"[🔗 기사 원문 보기]({row['url']})")

            article_score = int(row.get("balance_score", 0))
            article_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=article_score,
                title={"text": "기사 표현 균형 지표", "font": {"size": 15}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#60a5fa"},
                    "steps": [
                        {"range": [0, 40], "color": "#7f1d1d"},
                        {"range": [40, 70], "color": "#78350f"},
                        {"range": [70, 100], "color": "#14532d"},
                    ],
                }
            ))
            article_gauge.update_layout(
                height=220,
                margin=dict(l=10, r=10, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f9fafb", "family": "Pretendard"}
            )
            st.plotly_chart(article_gauge, use_container_width=True, key=f"article_gauge_{row_idx}")

        with main_col:
            st.markdown(f"### {row['title']}")
            st.markdown(
                f"""
                <span class="score-badge">기사 표현 균형 지표 {row['balance_score']}/100</span>
                """,
                unsafe_allow_html=True
            )

            st.markdown("#### 🧾 핵심 요약")
            st.success(row["summary"])

            st.markdown("#### 🗣️ 주요 주장")
            st.info(row["main_claim"])

            st.markdown("#### 🧭 프레이밍 분석")
            st.warning(row["framing"])

            exp_col1, exp_col2 = st.columns(2)

            with exp_col1:
                st.markdown("#### 감정적 표현")
                for item in make_list(row["emotional_expressions"]):
                    st.markdown(f"- {item}")

            with exp_col2:
                st.markdown("#### 단정적 표현")
                for item in make_list(row["assertive_expressions"]):
                    st.markdown(f"- {item}")

            st.markdown("#### 👀 빠져 있을 수 있는 관점")
            for item in get_text_list(row["missing_perspectives"]):
                st.markdown(f"- {item}")

            st.markdown("#### 🧪 표현 온도계")
            temp_label, temp_score = expression_temperature(row)
            st.progress(temp_score / 100)
            st.caption(f"{temp_label} · 감정적/단정적 표현 개수 기반 참고 지표")

            if "key_keywords" in row and safe_join_items(row.get("key_keywords")) != "-":
                st.markdown("#### #️⃣ 기사 핵심 키워드")
                st.write(safe_join_items(row.get("key_keywords")))

            st.caption(
                "※ 위 분석은 기사 원문을 바탕으로 LLM이 생성한 참고 정보이며, 기사 내용의 사실 여부나 정치적 성향을 확정하지 않습니다."
            )
            st.divider()
            st.markdown("#### 🙋 이 기사에 대한 나의 반응")
            
            reaction = st.radio(
                "이 기사 관점에 대한 나의 반응을 선택하세요",
                ["동의", "동의하지 않음", "판단 보류"],
                horizontal=True,
                key=f"article_reaction_{row_idx}")
            
            st.session_state.article_reactions[row_idx] = reaction
            st.caption("선택한 반응은 RAG 종합 분석 탭의 '나의 관점 패턴 리포트'에 반영됩니다.")

    csv = final_df.to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        label="📥 분석 결과 CSV 다운로드",
        data=csv,
        file_name=f"news_rag_analysis_{query_saved}.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_result_csv"
    )
