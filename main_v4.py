import os
import warnings
import config
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

# 경고 메시지를 내용으로 필터링하여 숨깁니다.
warnings.filterwarnings("ignore", message=".*deprecated.*")

# 🔹 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 🔹 프롬프트 템플릿 정의
explanation_prompt_template = """
'{question}'은(는) 평균 점수 {average_score:.1f}점으로 '{classification}'으로 판별되었습니다.

아래 리뷰 요약을 바탕으로, 이 판별에 대한 구체적인 이유를 긍정적, 부정적 측면에서 요약해 주십시오.
'내돈내산', '재방문' 등의 신뢰도 관련 키워드가 있다면 강조해서 언급해주세요.

리뷰 요약:
{context}

출력 형식:
'{question}'는 [{classification}]으로 판별됩니다. (평균 점수: {average_score:.1f}점)
이유:
1. 긍정적 표현 요약:
2. 부정적 표현 요약:
3. 신뢰도 판단 근거:
"""

query_expansion_prompt_template = """
당신은 사용자의 검색어를 확장하여 검색 성능을 높이는 AI 어시스턴트입니다.
사용자의 원래 검색어와 관련된, 검색에 도움이 될 만한 추가 검색어 3개를 쉼표(,)로 구분하여 생성해주세요.

예시:
- 입력: "만득이네 두루치기"
- 출력: "만득이네, 만득이네 후기, 남가좌동 만득이네"

- 입력: "스타벅스"
- 출력: "스타벅스 후기, 스타벅스 메뉴, 스타벅스 리뷰"

이제 다음 검색어에 대한 추가 검색어를 생성해주세요.
입력: "{question}"
출력:
"""

def run_rag(store_name, vectordb_path=config.VECTORDB_PATH):
    llm = ChatOpenAI(model=config.OPENAI_CHAT_MODEL, openai_api_key=OPENAI_API_KEY, temperature=config.OPENAI_TEMPERATURE)

    # 1. 쿼리 확장
    print(f"🔄 '{store_name}'에 대한 쿼리 확장 중...")
    query_expansion_prompt = PromptTemplate.from_template(query_expansion_prompt_template)
    query_expansion_chain = query_expansion_prompt | llm | StrOutputParser()
    expanded_queries_str = query_expansion_chain.invoke({"question": store_name})
    
    search_queries = [store_name] + [q.strip() for q in expanded_queries_str.split(',')]
    print(f"🔍 확장된 검색어: {search_queries}")

    # 2. 확장된 쿼리로 문서 검색 및 취합
    emb = OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=vectordb_path, embedding_function=emb)
    
    unique_docs = {}
    for query in search_queries:
        retrieved_docs_with_scores = db.similarity_search_with_relevance_scores(
            query=query, 
            k=config.RETRIEVER_SEARCH_K
        )
        for doc, score in retrieved_docs_with_scores:
            if score >= config.SIMILARITY_THRESHOLD:
                # 문서 내용과 메타데이터를 기반으로 고유 키 생성
                doc_key = (doc.page_content, doc.metadata.get('score'), doc.metadata.get('label'))
                if doc_key not in unique_docs:
                    unique_docs[doc_key] = doc

    # 3. 재점수화(Re-ranking): 추출된 가게이름이 일치하는 문서를 우선순위로 정렬
    priority_docs = []
    other_docs = []
    for doc in unique_docs.values():
        # 사용자의 검색어에 추출된 가게 이름이 포함되어 있는지 확인 (부분 일치 허용)
        if store_name in doc.metadata.get('extracted_name', ''):
            priority_docs.append(doc)
        else:
            other_docs.append(doc)
    
    relevant_docs = priority_docs + other_docs
    print(f"✨ 재점수화 완료: 우선순위 문서 {len(priority_docs)}개 / 전체 {len(relevant_docs)}개")

    # 4. 관련성 높은 문서가 없는 경우, 사용자에게 안내 메시지 반환
    if not relevant_docs:
        return f"'{store_name}'에 대한 리뷰 데이터를 찾을 수 없어, 답변을 생성할 수 없습니다. 가게 이름을 다시 확인해주세요."

    # --- (이하 로직은 기존과 동일) ---

    # 5. 검색된 문서의 점수 및 리뷰 수 계산
    v = len(relevant_docs)
    total_score = sum(doc.metadata.get('score', 0) for doc in relevant_docs)
    R = total_score / v if v > 0 else 0

    # 6. 가중 평점(Weighted Rating) 계산
    m = config.MIN_REVIEW_COUNT
    C = config.GLOBAL_AVERAGE_SCORE
    
    average_score = (v / (v + m)) * R + (m / (v + m)) * C

    # 7. 점수 기준으로 '맛집'/'비맛집' 분류
    classification = "맛집" if average_score >= config.SCORE_THRESHOLD else "비맛집"

    # 8. LLM을 통해 판별 이유 생성
    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = PromptTemplate(
        template=explanation_prompt_template,
        input_variables=["question", "average_score", "classification", "context"]
    )
    
    chain = prompt | llm
    
    result = chain.invoke({
        "question": store_name,
        "average_score": average_score,
        "classification": classification,
        "context": context
    })

    return result.content

if __name__ == "__main__":
    store = input("안녕하세요. 저는 맛집 판별 AI🍚입니다. 판별하고자하는 가게이름을 입력해주세요.\n> ")
    print("\n🍚 맛집 판별 중...\n")
    print(run_rag(store))