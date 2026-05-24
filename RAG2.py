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

from difflib import SequenceMatcher


# =====================================================
# 기본 설정
# =====================================================
st.set_page_config(
    page_title="LENS | 실시간 뉴스 관점 분석",
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

if "viewpoint_queries" not in st.session_state:
    st.session_state.viewpoint_queries = None


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

def is_relevant_to_query(query, title, description, text):
    """
    기존 코드는 제목/설명에 검색어가 반드시 포함되어야 해서
    반대관점 검색어로 확장했을 때 좋은 기사가 많이 탈락했습니다.
    이제는 제목+설명+본문 앞부분에서 핵심 토큰이 일부만 일치해도 통과시킵니다.
    """
    query = query.strip()
    combined = f"{title} {description} {text[:2500]}"

    tokens = [t for t in re.findall(r"[가-힣A-Za-z0-9]{2,}", query) if len(t) >= 2]
    if not tokens:
        return True

    hit_count = sum(1 for t in tokens if t in combined)

    # 핵심어가 1개라도 들어가면 후보로 인정.
    # 너무 엄격하게 걸러내면 반대 관점 기사가 사라짐.
    if hit_count < 1:
        return False

    return True

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
    text = str(row.get("cleaned_text", ""))

    emotional_count = len(make_list(row.get("emotional_expressions")))
    assertive_count = len(make_list(row.get("assertive_expressions")))
    missing_count = len(make_list(row.get("missing_perspectives")))

    tone = str(row.get("tone_label", "중립"))
    text_len = max(len(text), 1)

    strong_words = [
        "논란", "비판", "우려", "강력", "충격", "심각", "반발",
        "피해", "갈등", "위기", "논쟁", "분노", "불안", "문제"
    ]

    strong_word_count = sum(text.count(word) for word in strong_words)
    strong_word_density = strong_word_count / text_len * 1000

    score = 82

    score -= min(emotional_count * 2, 8)
    score -= min(assertive_count * 2.5, 10)
    score -= min(missing_count * 2.5, 8)

    # 이슈 특성상 자주 나오는 강한 단어는 약하게만 반영
    score -= min(strong_word_density * 1.5, 6)

    if tone == "중립":
        score += 5
    elif tone == "우려":
        score -= 1
    elif tone in ["비판", "강조"]:
        score -= 3

    # 기사별 차이를 위한 약한 보정
    score += min(3, max(-3, (text_len - 1500) / 700))

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
# 관점별 검색어 생성
# =====================================================
def generate_viewpoint_search_queries(query):
    """
    사용자 검색어를 그대로 검색하지 않고,
    찬성/반대/중립 관점이 드러날 수 있는 검색어 묶음으로 확장합니다.
    이 부분이 기존 벡터 검색 구조에서 가장 크게 바뀐 핵심입니다.
    """
    prompt = f"""
사용자가 뉴스 관점 분석을 위해 입력한 검색어는 다음과 같습니다.

[검색어]
{query}

이 이슈에 대해 서로 다른 관점의 뉴스를 찾기 위한 검색어를 만들어주세요.

반드시 JSON 형식으로만 답하세요.

{{
  "supportive_queries": ["찬성·긍정·필요성 관점 뉴스 검색어 1", "찬성·긍정·필요성 관점 뉴스 검색어 2"],
  "opposing_queries": ["반대·우려·비판 관점 뉴스 검색어 1", "반대·우려·비판 관점 뉴스 검색어 2"],
  "neutral_queries": ["배경·팩트·중립 설명 뉴스 검색어 1", "배경·팩트·중립 설명 뉴스 검색어 2"]
}}

조건:
- 검색어는 한국어 뉴스 검색에 바로 넣을 수 있게 짧게 작성
- 정치 성향을 단정하는 단어는 피하기
- 검색어마다 원래 이슈 키워드를 포함하기
"""
    fallback = {
        "supportive_queries": [f"{query} 필요성", f"{query} 기대 효과"],
        "opposing_queries": [f"{query} 반대", f"{query} 우려", f"{query} 비판"],
        "neutral_queries": [f"{query} 배경", f"{query} 쟁점", f"{query} 정리"]
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 뉴스 검색어를 관점별로 설계하는 전문가야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.25
        )
        data = safe_json_loads(response.choices[0].message.content)
        if not isinstance(data, dict):
            return fallback

        for key in ["supportive_queries", "opposing_queries", "neutral_queries"]:
            if key not in data or not isinstance(data[key], list) or len(data[key]) == 0:
                data[key] = fallback[key]
            data[key] = [str(x).strip() for x in data[key] if str(x).strip()][:3]

        return data
    except Exception:
        return fallback


def flatten_viewpoint_queries(plan):
    mapping = [
        ("찬성·긍정 관점", plan.get("supportive_queries", [])),
        ("반대·우려 관점", plan.get("opposing_queries", [])),
        ("중립·배경 관점", plan.get("neutral_queries", [])),
    ]
    rows = []
    for label, queries in mapping:
        for q in queries:
            rows.append((label, q))
    return rows

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

def collect_articles(query, max_articles=6):
    """
    기존: 특정 키워드를 하드코딩해서 비슷한 기사만 수집
    변경: LLM이 만든 찬성/반대/중립 검색어로 나누어 수집
    결과: 사용자가 검색어 하나만 넣어도 반대의견 뉴스까지 의도적으로 탐색
    """
    viewpoint_plan = generate_viewpoint_search_queries(query)
    st.session_state.viewpoint_queries = viewpoint_plan

    search_plan = flatten_viewpoint_queries(viewpoint_plan)
    articles = []
    seen_urls = set()
    seen_title_list = []
    seen_text_list = []
    press_count = {}
    group_count = {"찬성·긍정 관점": 0, "반대·우려 관점": 0, "중립·배경 관점": 0}

    per_group_limit = max(1, max_articles // 3)

    for viewpoint_group, search_query in search_plan:
        if len(articles) >= max_articles:
            break

        # 특정 그룹이 너무 많이 차지하지 않도록 제한
        if group_count.get(viewpoint_group, 0) >= per_group_limit + 1:
            continue

        items = (
            search_naver_news(search_query, display=15, sort="sim")
            + search_naver_news(search_query, display=10, sort="date")
        )

        for item in items:
            if len(articles) >= max_articles:
                break

            title = remove_html(item.get("title", ""))
            description = remove_html(item.get("description", ""))
            article_url = item.get("originallink") or item.get("link")
            press = get_press_name(article_url)

            if not article_url or article_url in seen_urls:
                continue

            # 한 언론사 기사만 몰리지 않게 함
            if press_count.get(press, 0) >= 1 and len(press_count) < max_articles:
                continue

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

                if not is_relevant_to_query(query, title + " " + search_query, description, cleaned_text):
                    continue

                if any(SequenceMatcher(None, title, old).ratio() >= 0.72 for old in seen_title_list):
                    continue

                current_sample = cleaned_text[:800]
                if any(SequenceMatcher(None, current_sample, old).ratio() >= 0.55 for old in seen_text_list):
                    continue

                seen_urls.add(article_url)
                seen_title_list.append(title)
                seen_text_list.append(current_sample)
                press_count[press] = press_count.get(press, 0) + 1
                group_count[viewpoint_group] = group_count.get(viewpoint_group, 0) + 1

                articles.append({
                    "query": query,
                    "source_search_query": search_query,
                    "viewpoint_group": viewpoint_group,
                    "title": article.title if article.title else title,
                    "description": description,
                    "url": article_url,
                    "press": press,
                    "pub_date": item.get("pubDate", ""),
                    "cleaned_text": cleaned_text,
                    "collected_at": datetime.now().strftime("%Y.%m.%d %H:%M")
                })

                break

            except Exception:
                continue

    return articles

# =====================================================
# 실시간 뉴스 관점 분석 구성
# =====================================================
class SimpleSourceDoc:
    """화면에서 근거를 확인하기 위한 간단한 문서 객체"""
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def create_source_docs_from_articles(articles):
    docs = []
    for idx, article in enumerate(articles, start=1):
        content = f"""
[기사 번호]
{idx}

[수집 관점]
{article.get('viewpoint_group', '분류 없음')}

[수집 검색어]
{article.get('source_search_query', article.get('query', ''))}

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
            SimpleSourceDoc(
                page_content=content,
                metadata={
                    "article_id": idx,
                    "title": article["title"],
                    "press": article["press"],
                    "url": article["url"],
                    "pub_date": article["pub_date"],
                    "viewpoint_group": article.get("viewpoint_group", "분류 없음"),
                    "source_search_query": article.get("source_search_query", article.get("query", ""))
                }
            )
        )
    return docs


def format_articles_for_prompt(articles, max_chars=1800):
    formatted = []
    for i, article in enumerate(articles, start=1):
        formatted.append(
            f"""
[근거 기사 {i}]
수집 관점: {article.get('viewpoint_group', '분류 없음')}
검색어: {article.get('source_search_query', article.get('query', ''))}
제목: {article.get('title', '')}
언론사: {article.get('press', '')}
날짜: {format_pub_date(article.get('pub_date', ''))}
URL: {article.get('url', '')}
본문 근거:
{article.get('cleaned_text', '')[:max_chars]}
"""
        )
    return "\n\n".join(formatted)


def analyze_issue_with_llm(query, articles):
    """
    RAG/벡터DB 없이 동작하는 핵심 분석 함수.
    1) 네이버 API로 관점별 기사 수집
    2) 수집된 기사 원문을 LLM에 근거로 제공
    3) 찬성·반대·중립 관점을 비교 분석
    """
    context = format_articles_for_prompt(articles)

    prompt = f"""
너는 미디어 리터러시 교육을 위한 뉴스 관점 분석 AI야.

사용자가 입력한 이슈는 다음과 같아.
[검색어]
{query}

아래는 이 검색어에 대해 찬성·반대·중립 관점 검색어를 확장하여 실시간으로 수집한 뉴스 기사들이야.

[수집된 뉴스 근거]
{context}

중요한 원칙:
- 정치 성향을 단정하지 마.
- 기사 내용의 진위 여부를 판정하지 마.
- 특정 관점을 정답처럼 제시하지 마.
- 수집된 기사 근거 안에서만 분석해.
- 찬성/반대/중립 중 어느 한쪽을 정답처럼 쓰지 말고, 쟁점 구조를 비교해.
- 사용자가 다양한 관점에서 뉴스를 이해하도록 도와줘.
- 대답은 한국어로 작성해.

반드시 아래 JSON 형식으로만 답해.

{{
  "common_core": "여러 기사에서 공통적으로 다루는 핵심 사건 요약",
  "perspective_cards": [
    {{"icon": "🟦", "label": "찬성·긍정 관점", "content": "찬성 또는 긍정 관점의 핵심 주장과 근거"}},
    {{"icon": "🟥", "label": "반대·우려 관점", "content": "반대 또는 우려 관점의 핵심 주장과 근거"}},
    {{"icon": "⬜", "label": "중립·배경 관점", "content": "배경, 사실관계, 절충 관점의 핵심 내용"}}
  ],
  "perspective_differences": [
    "찬성과 반대가 갈리는 핵심 기준 1",
    "찬성과 반대가 갈리는 핵심 기준 2",
    "찬성과 반대가 갈리는 핵심 기준 3"
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

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 뉴스 관점 차이와 프레이밍을 분석하는 미디어 리터러시 AI야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        result_text = response.choices[0].message.content
        result_json = safe_json_loads(result_text)
    except Exception as e:
        result_json = None
        result_text = f"분석 중 오류가 발생했습니다: {e}"

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

    source_docs = create_source_docs_from_articles(articles)
    return result_json, source_docs

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
  "summary": "기사를 읽지 않아도 이해할 수 있도록 핵심 배경, 사건 내용, 주요 이해관계자, 쟁점을 포함해 3~4문장으로 요약",
  "main_claim": "기사에서 가장 강조하는 주장",
  "framing": "기사가 사건을 어떤 관점 중심으로 바라보는지 짧은 프레이밍 라벨만 생성. 예: 경제 성장 중심 / 갈등 중심 / 정책 효과 중심 / 피해자 관점 중심 / 산업 혁신 중심",
  "tone_label": "중립/우려/비판/긍정/강조 중 하나",
  "key_keywords": ["핵심 키워드 1", "핵심 키워드 2", "핵심 키워드 3"],
  "issue_position_score": "0~100 사이 정수. 기사 관점의 위치를 나타내는 참고 점수. 한쪽 주장만 강하게 강조하면 낮거나 높게, 여러 관점을 함께 제시하면 중간에 가깝게 평가",
  "emotional_expressions": "기사에 실제로 등장한 감정적 표현만 배열로 작성. 없으면 빈 배열 []",
  "assertive_expressions": "기사에 실제로 등장한 단정적 표현만 배열로 작성. 없으면 빈 배열 []",
  "missing_perspectives": "기사에서 충분히 다루지 않은 관점만 배열로 작성. 없으면 빈 배열 []",
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
# UI/UX 강화 CSS - Light Professional Theme
# =====================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Pretendard', sans-serif;
}

.stApp {
    background: #f8fafc;
    color: #0f172a;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1180px;
}

/* 상단 */
.hero-container {
    padding: 2rem 0 1.2rem 0;
    text-align: left;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 48px;
    font-weight: 900;
    color: #0f172a;
    letter-spacing: -2px;
    margin-bottom: 8px;
}

.hero-title span {
    color: #2563eb;
}

.hero-copy {
    font-size: 17px;
    color: #475569;
    font-weight: 500;
    line-height: 1.7;
}

/* 안내 박스 */
.notice-box {
    background: #ffffff;
    border-left: 5px solid #2563eb;
    border-top: 1px solid #dbeafe;
    border-right: 1px solid #dbeafe;
    border-bottom: 1px solid #dbeafe;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 24px;
    color: #334155;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}

/* 카드 */
.card, .summary-card, .mini-card {
    background: #ffffff;
    border:1px solid #dbeafe;
    border-left:6px solid #2563eb;
    border-radius:18px;
    padding:22px;
    margin-bottom:18px;
    box-shadow:
    0 8px 22px rgba(15,23,42,.05);
}

.card{
    height:310px;
    display:flex;
    flex-direction:column;
    justify-content:flex-start;
}

.card-title {
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 12px;
    color: #0f172a;
}

.meta-text {
    color: #475569;
    font-size: 14px;
    line-height: 1.8;
}

.summary-card-title {
    font-size: 22px;
    font-weight: 850;
    margin-bottom: 12px;
    color: #1d4ed8;
}

.summary-card-body {
    font-size: 16px;
    line-height: 1.85;
    color: #1e293b;
}

/* 점수 배지 */
.score-badge {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    padding: 7px 13px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 800;
    margin: 6px 0 14px 0;
}

.article-link {
    color: #2563eb !important;
    text-decoration: none !important;
    font-weight: 800;
}

/* 버튼 */
.stButton > button {
    background:#ffffff !important;
    color:#1e3a8a !important;
    border:2px solid #2563eb !important;
    border-radius:14px !important;
    font-weight:700 !important;
    box-shadow:
    0 4px 12px rgba(37,99,235,.08);
    transition:.25s;
}

.stButton > button:hover {
    background:#f1f5f9 !important;
    border-color:#94a3b8 !important;
    color:#1e293b !important;
    transform:none;
    box-shadow:none;
}

.stButton > button:active {
    background:#e2e8f0 !important;
    border-color:#94a3b8 !important;
    color:#1e293b !important;
    box-shadow:none !important;
}

.related-button{
    background:#ffffff;
    border:2px solid #2563eb;
    color:#1e293b;
}

/* 입력창/슬라이더 */
.stTextInput label,
.stSlider label,
.stSelectbox label,
.stRadio label {
    color: #0f172a !important;
    font-weight: 700 !important;
}

.stSlider span {
    color: #334155 !important;
}

/* Metric */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dbeafe;
    border-left: 5px solid #3b82f6;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
}

[data-testid="stMetricLabel"] {
    color: #334155 !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 900 !important;
}

/* 탭 */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #e2e8f0;
}

.stTabs [data-baseweb="tab"] {
    height: 46px;
    background-color: #f1f5f9;
    border-radius: 12px 12px 0 0;
    color: #334155;
    padding: 0 18px;
    font-weight: 800;
}

.stTabs [aria-selected="true"] {
    background-color: #dbeafe !important;
    color: #1d4ed8 !important;
    border-bottom: 3px solid #2563eb !important;
}

/* 관련 검색어 박스 */
.related-box {
    background: #ffffff;
    border: 1px solid #dbeafe;
    border-left: 5px solid #38bdf8;
    border-radius: 16px;
    padding: 16px 18px;
    margin: 8px 0 20px 0;
}

.related-title {
    color: #0f172a;
    font-weight: 850;
    margin-bottom: 6px;
}

.related-desc {
    color: #64748b;
    font-size: 0.92rem;
}

/* 텍스트 가독성 */
.stMarkdown, .stMarkdown p, .stMarkdown span,
.stWrite, p, span, div {
    color: #1e293b;
}

h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
}

[data-testid="stCaptionContainer"] {
    color: #64748b !important;
}

[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
}

[data-testid="stExpander"] * {
    color: #1e293b !important;
}

[data-testid="stText"] {
    color: #1e293b !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px;
    line-height: 1.8;
}

[data-testid="stDataFrame"]{
    border:1px solid #dbeafe;
    border-radius:16px;
}

/* Plotly */
.js-plotly-plot text {
    fill: #334155 !important;
}

/* 알림 박스 */
[data-testid="stAlert"] * {
    color: #0f172a !important;
}

/* 다운로드 버튼 */
.stDownloadButton > button {
    background: #ffffff !important;
    color: #2563eb !important;
    border: 1px solid #93c5fd !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
}
.report-card, .report-section {
    background:#ffffff;
    border:1px solid #dbeafe;
    border-left:6px solid #2563eb;
    border-radius:18px;
    padding:22px;
    margin-bottom:20px;
    box-shadow:0 8px 22px rgba(15,23,42,.05);
}

.report-card {
    min-height:260px;
}

.report-title {
    font-size:20px;
    font-weight:850;
    color:#0f172a;
    margin-bottom:16px;
}

.report-item {
    background:#f8fafc;
    border:1px solid #e2e8f0;
    border-radius:12px;
    padding:14px 16px;
    margin-bottom:12px;
    line-height:1.7;
    color:#1e293b;
}

.report-item.orange {
    background:#fff7ed;
    border-color:#fed7aa;
}

.report-empty {
    color:#94a3b8;
    font-size:14px;
}

.keyword-row {
    display:flex;
    flex-wrap:wrap;
    gap:10px;
}

.keyword-pill {
    display:inline-flex;
    background:#eff6ff;
    border:1px solid #bfdbfe;
    color:#2563eb;
    border-radius:999px;
    padding:7px 13px;
    font-size:13px;
    font-weight:750;
    white-space:nowrap;
}

.frame-item {
    background:#f8fafc;
    border-left:4px solid #2563eb;
    border-radius:10px;
    padding:13px 15px;
    margin-bottom:10px;
    line-height:1.7;
}

.missing-grid {
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:12px;
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
        <div class="hero-title">⚖️ LENS</div>
        <div class="hero-copy">
            실시간 뉴스 수집 · 반대관점 탐색 · LLM 기반 통합 분석 서비스<br>
            같은 이슈의 찬성·반대·중립 관점을 함께 찾아 균형 있게 읽도록 돕습니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        margin-top:-8px;
        margin-bottom:18px;
        padding-left:6px;
        font-size:14px;
        font-weight:500;
        line-height:1.8;
        color:#64748b;
    ">
        검색어를 구체화하면 관련 기사 정확도가 높아집니다.<br>
        예: 촉법소년 → 촉법소년 연령 하향 · 소년법 개정 · 청소년 범죄 예방
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
    recommend_button = st.button("관련 검색어 추천", use_container_width=True)

if recommend_button:
    if not query.strip():
        st.warning("관련 검색어를 추천받으려면 먼저 검색어를 입력해주세요.")
    else:
        with st.spinner("AI가 관련 검색어를 추천하는 중입니다..."):
            st.session_state.related_keywords = generate_related_keywords(query.strip())

if st.session_state.related_keywords:
    st.markdown("### 관련 검색어 추천")
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
color:#1e3a8a;
font-size:14px;
font-weight:700;
">
선택된 검색어:
</span>

<span style="
color:#2563eb;
font-size:16px;
font-weight:900;
">
{st.session_state.selected_related_keyword}
</span>

</div>
""",
    unsafe_allow_html=True
)

start_button = st.button("실시간 뉴스 관점 균형 분석 시작", use_container_width=True)


# =====================================================
# 분석 실행 - UX 개선 버전
# =====================================================
if start_button:
    query = query.strip()
    if not query:
        st.warning("분석할 키워드를 입력해 주세요.")
    else:
        st.session_state.selected_related_keyword = query
        st.session_state.rag_result = None
        st.session_state.final_df = None
        st.session_state.reaction_saved = False
        st.session_state.query_saved = query
        st.session_state.article_reactions = {}
        st.session_state.user_perspective_report = None

        with st.status("LENS AI가 실시간 뉴스와 반대관점을 분석하고 있습니다...", expanded=True) as status:
            st.write("네이버 API에서 관련 기사를 수집 중...")
            articles = collect_articles(query, max_articles=max_articles)
            

            if len(articles) == 0:
                status.update(label="❌ 수집된 기사가 없습니다.", state="error", expanded=True)
                st.error("수집된 기사가 없습니다. 검색어를 바꾸거나 다시 시도해보세요.")
            else:
                st.session_state.articles = articles
                st.write(f"✅ {len(articles)}개 기사 수집 완료")

                if st.session_state.viewpoint_queries:
                    st.write("✅ 반대관점 검색어 확장 완료")

                st.write("⚖️ 검색 기반 전체 관점 통합 분석 중...")
                rag_result, retrieved_docs = analyze_issue_with_llm(query, articles)
                st.session_state.rag_result = rag_result
                st.session_state.retrieved_docs = retrieved_docs

                st.write("기사별 세부 분석 중...")
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
                
                final_df["llm_balance_score"] = pd.to_numeric(
                    final_df["balance_score"],
                    errors="coerce"
                ).fillna(60)
                
                final_df["balance_score"] = final_df.apply(
                    calculate_article_balance_score,
                    axis=1)
                
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
    st.subheader("실시간 뉴스 관점 균형 분석 결과")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("검색 키워드", query_saved)

    with metric_col2:
        st.metric("수집 기사 수", f"{len(final_df)}개")

    with metric_col3:
        st.metric("종합 관점 다양성 지표", f"{score}/100")

    if st.session_state.viewpoint_queries:
        with st.expander("이번 분석에 사용한 관점별 검색어"):
            vq = st.session_state.viewpoint_queries
            st.write("찬성·긍정 관점:", ", ".join(vq.get("supportive_queries", [])))
            st.write("반대·우려 관점:", ", ".join(vq.get("opposing_queries", [])))
            st.write("중립·배경 관점:", ", ".join(vq.get("neutral_queries", [])))

    tabs = st.tabs(["관점 종합 분석", "수집 기사 목록", "기사별 상세 분석"])

    # =================================================
    # 1. 종합 관점 분석 탭 - 고도화 버전
    # =================================================
    with tabs[0]:
        st.markdown("## 검색 기반 전체 관점 분석")

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
            score = int(score)
            
            with st.container(border=True):
                st.markdown("#### 관점 다양성 지표")
                st.metric(label="종합 점수",value=f"{score}/100")
                st.progress(score / 100)
                st.caption("기사들 사이의 관점 차이, 언론사 다양성, 표현 톤을 종합 계산한 참고 지표입니다.")
        st.divider()
                
                # 관점 자동 라벨 카드
        st.markdown("## 주제별 관점 비교")
        st.caption("경제·정치·사회 이슈에 따라 LLM이 관점 라벨을 자동으로 구성합니다.")
        
        perspective_cards = get_perspective_cards(rag_result)
        p_cols = st.columns(3)
        
        for i, col in enumerate(p_cols):
            card = perspective_cards[i] if i < len(perspective_cards) else {
                "icon": "",
                "label": f"관점 {i + 1}",
                "content": "관련 관점이 충분히 수집되지 않았습니다."}
            
            with col:
                st.markdown(
                    f"""
                    <div class="card">
                    <div class="card-title">{card.get('label', f'관점 {i + 1}')}</div>
                    <div class="summary-card-body">{card.get('content', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True)

        st.divider()

        # 기사별 비교표 + 키워드 차트
        st.markdown("## 기사 간 관점 비교")
        st.caption("같은 이슈를 기사들이 어떤 관점으로 다루는지 비교합니다.")
        
        compare_df = final_df[
            ["viewpoint_group","press","main_claim","framing","tone_label","balance_score"]].copy()
        
        compare_df.columns = ["수집 관점","언론사","핵심 주장","프레이밍","톤","표현 균형"]
        
        st.dataframe(compare_df,use_container_width=True,hide_index=True)

    

        # 강조점 차이
        st.markdown("## 기사들이 강조한 내용")
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
            st.markdown("## 빠져 있을 수 있는 관점")
            for item in get_text_list(rag_result.get("missing_perspectives")):
                st.warning(item)

        with question_col:
            st.markdown("## 생각해볼 질문")
            for item in get_text_list(rag_result.get("thinking_questions")):
                st.info(item)

        st.divider()

        with st.expander("분석 근거 기사 확인"):
            for i, doc in enumerate(st.session_state.retrieved_docs, start=1):
                st.markdown(f"### 근거 {i}")
                st.markdown(f"**제목:** {doc.metadata.get('title', '')}")
                st.markdown(f"**언론사:** {doc.metadata.get('press', '')}")
                st.markdown(f"**URL:** {doc.metadata.get('url', '')}")
                st.text(doc.page_content[:1200])
                st.divider()
        
        st.markdown("## 나의 관점 패턴 리포트")
        st.caption("기사별 상세 분석 탭에서 각 기사에 대한 반응을 선택하면 이곳에서 관점 패턴을 확인할 수 있습니다.")
        
        if st.button("🧭 나의 관점 패턴 분석하기", use_container_width=True):
            st.session_state.user_perspective_report = build_user_perspective_report(
                final_df,
                st.session_state.article_reactions)
            
        if st.session_state.user_perspective_report:
            report = st.session_state.user_perspective_report
            
            st.markdown("### 나의 관점 패턴 리포트")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("동의", report["agree_count"])
            c2.metric("동의하지 않음", report["disagree_count"])
            c3.metric("판단 보류", report["hold_count"])
            
            agreed_html = "".join(
                [f"<div class='report-item'>{claim}</div>" for claim in report["agreed_claims"][:3]]
            ) or "<div class='report-empty'>아직 동의한 기사가 없습니다.</div>"
            
            disagreed_html = "".join(
                [f"<div class='report-item orange'>{claim}</div>" for claim in report["disagreed_claims"][:3]]
            ) or "<div class='report-empty'>아직 동의하지 않은 기사가 없습니다.</div>"
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(
                    f"""
                    <div class="report-card">
                        <div class="report-title">내가 수용한 주장</div>
                        {agreed_html}
                    </div>
                    """,
                    unsafe_allow_html=True)
                
            with col_b:
                st.markdown(
                    f"""
                    <div class="report-card">
                        <div class="report-title">거리감을 느낀 주장</div>
                        {disagreed_html}
                    </div>
                    """,
                    unsafe_allow_html=True)
                
            keyword_tags = "".join(
                [
                    f"<span class='keyword-pill'>#{keyword}</span>"
                    for keyword in report["agreed_keywords"][:8]
                    ]
            ) or "<span class='report-empty'>확인된 키워드가 없습니다.</span>"
            
            frame_items = "".join(
                [
                    f"<div class='frame-item'>{frame}</div>"
                    for frame in report["disagreed_frames"][:4]
                    ]
            ) or "<div class='report-empty'>확인된 프레이밍이 없습니다.</div>"
            
            col_c, col_d = st.columns(2)
            
            with col_c:
                st.markdown(
                    f"""
                    <div class="report-section">
                        <div class="report-title">반응한 핵심 키워드</div>
                        <div class="keyword-row">
                            {keyword_tags}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True)
                
            with col_d:
                st.markdown(
                    f"""
                    <div class="report-section">
                        <div class="report-title">거리감을 느낀 프레이밍</div>
                        {frame_items}
                    </div>
                    """,
                    unsafe_allow_html=True)
                
            missing_items = report["missing_from_hold"] + report["missing_from_disagree"]
            missing_items = list(dict.fromkeys(missing_items))
            
            missing_html = "".join(
                [f"<div class='report-item'>{item}</div>" for item in missing_items[:6]]
            ) or "<div class='report-empty'>현재 선택만으로는 추가 확인 관점이 충분히 도출되지 않았습니다.</div>"
            
            st.markdown(
                f"""
                <div class="report-section">
                    <div class="report-title">추가로 읽어볼 필요가 있는 관점</div>
                    <div class="missing-grid">
                        {missing_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True)

    # =================================================
    # 2. 수집 기사 목록 탭
    # =================================================
    with tabs[1]:
        st.markdown("## 실시간 수집 기사 목록")
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
        st.markdown("## 기사별 상세 관점 분석")

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
                        <b>수집 관점</b><br>{row.get('viewpoint_group', '-')}<br><br>
                        <b>검색어</b><br>{row.get('source_search_query', '-')}<br><br>
                        <b>기사 날짜</b><br>{format_pub_date(row['pub_date'])}<br><br>
                        <b>수집 시간</b><br>{row['collected_at']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(f"[기사 원문 보기]({row['url']})")

            article_score = int(row.get("balance_score", 0))
            with st.container(border=True):
                st.markdown("##### 기사 표현 균형 지표")
                st.metric(label="기사 점수",value=f"{article_score}/100")
                st.progress(article_score/100)
                if article_score >= 70:
                    st.caption("다양한 관점과 반론이 비교적 함께 제시된 기사입니다.")
                elif article_score >= 40:
                    st.caption("일부 관점은 포함되었지만 특정 강조가 존재할 수 있습니다.")
                else:
                    st.caption("특정 주장이나 표현이 강하게 강조된 기사입니다.")

        with main_col:
            st.markdown(f"### {row['title']}")
            st.markdown(
                f"""
                <span class="score-badge">기사 표현 균형 지표 {row['balance_score']}/100</span>
                """,
                unsafe_allow_html=True
            )

            st.markdown("#### 핵심 요약")
            st.markdown(
                f"""
                <div class="mini-card">
                <div class="summary-card-body">
                {row.get("summary", "")}
                </div>
                </div>
                """,
                unsafe_allow_html=True)
            
            st.markdown("#### 주요 주장")
            st.markdown(
                f"""
                <div style="
                background:#eff6ff;
                border:1px solid #bfdbfe;
                border-left:5px solid #2563eb;
                border-radius:14px;
                padding:18px;
                font-size:16px;
                line-height:1.8;
                color:#0f172a;
                margin-bottom:18px;
                ">
                {row.get("main_claim", "")}
                </div>
                """,
                unsafe_allow_html=True)
            
            st.markdown("#### 기사 관점 분석")
            emotion_items = get_text_list(row.get("emotional_expressions"))
            assert_items = get_text_list(row.get("assertive_expressions"))
            missing_items = get_text_list(row.get("missing_perspectives"))
            
            emotion_html = (
                "".join([f"<li>{x}</li>" for x in emotion_items])
                if emotion_items
                else "<span style='color:#94a3b8'>해당 표현 없음</span>")
            
            assert_html = (
                "".join([f"<li>{x}</li>" for x in assert_items])
                if assert_items
                else "<span style='color:#94a3b8'>해당 표현 없음</span>")
            
            missing_html = (
                "".join([f"<li>{x}</li>" for x in missing_items])
                if missing_items
                else "<span style='color:#94a3b8'>추가 관점 없음</span>")
            
            analysis_box_html = f"""
            <div style="
            background:#ffffff;
            border:1px solid #dbeafe;
            border-left:6px solid #2563eb;
            border-radius:18px;
            padding:20px;
            margin-bottom:22px;
            ">
            
            <div style="
            display:grid;
            grid-template-columns:1fr 1fr;
            gap:16px;
            ">
            
            <div style="background:#f8fafc;border-radius:14px;padding:16px;">
            <div style="font-size:15px;font-weight:850;color:#1d4ed8;margin-bottom:10px;">
            프레이밍
            </div>
            
            <div style="line-height:1.8;">
            {row.get("framing","-")}
            </div>
            </div>
            
            <div style="background:#f8fafc;border-radius:14px;padding:16px;">
            <div style="font-size:15px;font-weight:850;color:#1d4ed8;margin-bottom:10px;">
            빠져 있을 수 있는 관점
            </div>
            
            <ul style="padding-left:18px;margin:0;">
            {missing_html}
            </ul>
            </div>
            
            <div style="background:#f8fafc;border-radius:14px;padding:16px;">
            <div style="font-size:15px;font-weight:850;color:#1d4ed8;margin-bottom:10px;">
            감정적 표현
            </div>
            
            <ul style="padding-left:18px;margin:0;">
            {emotion_html}
            </ul>
            </div>
            
            <div style="background:#f8fafc;border-radius:14px;padding:16px;">
            <div style="font-size:15px;font-weight:850;color:#1d4ed8;margin-bottom:10px;">
            단정적 표현
            </div>
            
            <ul style="padding-left:18px;margin:0;">
            {assert_html}
            </ul>
            </div>
            </div>
            </div>
            """
            
            st.markdown(analysis_box_html, unsafe_allow_html=True)
            
            st.markdown("#### 표현 강도")
            temp_label, temp_score = expression_temperature(row)
            st.progress(temp_score / 100)
            st.caption(
                "감정적 표현과 단정적 표현의 개수를 바탕으로, 기사 문장이 얼마나 강한 어조로 쓰였는지 보여주는 참고 지표입니다.")
            
            if "key_keywords" in row and safe_join_items(row.get("key_keywords")) != "-":
                tags = []
                for keyword in get_text_list(row.get("key_keywords")):
                    tags.append(
                        f"""
                    <span style="
                        display:inline-flex;
                        align-items:center;
                        justify-content:center;
                        background:#eff6ff;
                        border:1px solid #bfdbfe;
                        color:#2563eb;
                        border-radius:999px;
                        padding:8px 14px;
                        font-size:13px;
                        font-weight:700;
                    ">
                    #{keyword}
                    </span>
                    """
                    )
                    
                keyword_html = "".join(tags)
                st.markdown(
                    f"""
                    <div style="
                        display:flex;
                        flex-wrap:wrap;
                        gap:10px;
                        margin-top:18px;
                        margin-bottom:25px;
                    ">
                        {keyword_html}
                    </div>
                    """,
                    unsafe_allow_html=True)

            st.caption(
                "※ 위 분석은 기사 원문을 바탕으로 LLM이 생성한 참고 정보이며, 기사 내용의 사실 여부나 정치적 성향을 확정하지 않습니다."
            )
            st.divider()
            st.markdown("#### 이 기사에 대한 나의 반응")
            
            reaction = st.radio(
                "이 기사 관점에 대한 나의 반응을 선택하세요",
                ["동의", "동의하지 않음", "판단 보류"],
                horizontal=True,
                key=f"article_reaction_{row_idx}")
            
            st.session_state.article_reactions[row_idx] = reaction
            st.caption("선택한 반응은 관점 종합 분석 탭의 '나의 관점 패턴 리포트'에 반영됩니다.")

    csv = final_df.to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        label="📥 분석 결과 CSV 다운로드",
        data=csv,
        file_name=f"news_viewpoint_analysis_{query_saved}.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_result_csv"
    )
