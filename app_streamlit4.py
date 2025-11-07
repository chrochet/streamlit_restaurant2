import streamlit as st
from main_v4 import run_rag
import config

st.set_page_config(
    page_title="맛집 판별 AI",
    page_icon="🍚",
)

st.title("🍚 맛집 판별 AI ")

st.markdown("""안녕하세요! 저는 맛집 판별 AI입니다.  
판별하고자 하는 가게 이름을 입력하고 '판별 시작' 버튼을 눌러주세요.
(리뷰 데이터가 없는 식당은 판별이 불가능할 수 있습니다)
""")

store_name = st.text_input("가게 이름", placeholder="예: 가타쯔무리, 만득이네")
branch_name = ""

# 사용자가 입력한 가게 이름이 체인점 목록에 있는지 확인
if store_name in config.CHAIN_RESTAURANTS:
    branch_name = st.text_input("지점명", placeholder="예: 강남점 (체인점은 지점명을 입력해주세요)")

if st.button("판별 시작"):
    if not store_name:
        st.warning("가게 이름을 입력해주세요.")
    else:
        # 체인점의 경우 지점명까지 합쳐서 최종 검색어 생성
        final_query = store_name
        if branch_name:
            final_query = f"{store_name} {branch_name}"
        
        with st.spinner(f"'{final_query}'에 대한 리뷰를 분석 중입니다... 잠시만 기다려주세요."):
            result = run_rag(final_query, vectordb_path="vectordb4")
            st.divider()
            st.markdown(result)
