# API and Model Configurations
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.3

# VectorDB and Retriever Configurations
VECTORDB_PATH = "vectordb"
RETRIEVER_SEARCH_K = 5
SIMILARITY_THRESHOLD = 0.3 # 유사도 점수 기준값 (0~1 사이, 높을수록 유사함)

# Application Logic Configurations
SCORE_THRESHOLD = 7
MIN_REVIEW_COUNT = 5  # 가중 평점 계산을 위한 최소 리뷰 수 (m)
GLOBAL_AVERAGE_SCORE = 6.0  # 전체 가게의 평균 평점으로 가정하는 값 (C)
CHAIN_RESTAURANTS = ["베가보쌈", "원할머니보쌈", "만배아리랑보쌈"]

# Preprocessing Keyword Scores
KEYWORD_SCORES = {
    "내돈내산": 10,
    "재방문": 9,
    "맛집": 8,
    "존맛": 7,
    "최고": 6,
    "친절": 4,
    "서비스": 3,
    "추천": 3,
    "깔끔": 2,

    "별로": -6,
    "별로였음": -7,
    "실망": -8,
    "불친절": -9,
    "비추천": -7,
    "기대 이하": -7,
    "돈값 못함": -8,
    "맛없음": -9,
    "다시는 안 옴": -10,
    "한 번이면 됨": -6,
    "광고": -10,
    "협찬": -12,
    "광고 아님": 8,
    "광고 아님 진심 후기": 8,
    "홍보 아니에요": 7,
    "지인 추천으로 감": 7,
    "솔직 후기": 7,
    "내 돈 주고 먹은 거예요": 9,
    "내가 직접 찾아간 곳": 8,

    "맛있음": 8,
    "만족": 8,
    "또 가고 싶음": 9,
    "진짜 맛있다": 9
}

# Preprocessing Stopwords
STOPWORDS = [
    "은", "는", "이", "가", "을", "를", "의", "에", "에서", "와", "과", "도", "으로", "로",
    "하다", "되다", "이다", "있다", "없다", "같다",
    "그리고", "하지만", "그러나", "그래서", "그런데", "또", "또한",
    "것", "수", "저", "제", "저희", "때", "그", "이것", "저것", "그것"
]
