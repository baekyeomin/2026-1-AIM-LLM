import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# =========================
# 로직 :
# 1. professor_labinfo.csv에서 교수별 정보를 하나의 텍스트로 묶음
# 2. TFIDF 벡터화 (텍스트 -> 숫자)
# 3. 대충 하드웨어적인 교수님 / 비전 및 멀티모달 / 추천 및 데이터 / 시스템네트워크 등 class를 대표하는 키워드 리스트를 만들었어용
# 4. 키워드 묶음 TFIDF 벡터화
# 5. 코사인 유사도로 각 교수님이 어느 class와 비슷한지 계산
# 6. 애매할 수도 있으니 애매하면 키워드 등장빈도로 분류
# 7. 교수님끼리도 가까운 사람 KNN으로 (K=3) 찾았습니당
# =========================

# =========================
# 출력결과 :
'''

=== 계열별 교수 분류 결과 ===

[하드웨어계열]
- 권은지
- 김민규
- 주용수

[비전/멀티모달계열]
- 김영욱
- 김장호
- 김준호 
- 안인규
- 윤상민

[추천/데이터마이닝계열]
- 박하명
- 배홍균
- 이현기

[시스템/네트워크계열]
- 김상철
- 박수현
- 윤명근
- 임세민
- 최상혁

=== 교수별 상세 결과 ===
   교수님 성함 predicted_category  category_score final_category  \
0     권은지             하드웨어계열        0.367498         하드웨어계열   
1     김민규             하드웨어계열        0.102458         하드웨어계열   
2     김영욱          비전/멀티모달계열        0.339398      비전/멀티모달계열   
3     김장호          비전/멀티모달계열        0.141410      비전/멀티모달계열   
4     김상철         시스템/네트워크계열        0.099943     시스템/네트워크계열   
5    김준호           비전/멀티모달계열        0.249754      비전/멀티모달계열   
6     박수현         시스템/네트워크계열        0.214338     시스템/네트워크계열   
7     박하명        추천/데이터마이닝계열        0.443481    추천/데이터마이닝계열   
8     배홍균        추천/데이터마이닝계열        0.080082    추천/데이터마이닝계열   
9     안인규          비전/멀티모달계열        0.275632      비전/멀티모달계열   
10    윤상민          비전/멀티모달계열        0.155482      비전/멀티모달계열   
11    윤명근         시스템/네트워크계열        0.018694     시스템/네트워크계열   
12    이현기        추천/데이터마이닝계열        0.055652    추천/데이터마이닝계열   
13    임세민         시스템/네트워크계열        0.024024     시스템/네트워크계열   
14    주용수             하드웨어계열        0.039680         하드웨어계열   
15    최상혁         시스템/네트워크계열        0.024500     시스템/네트워크계열   

                               비슷한 교수 TOP3  
0    김영욱 (0.122), 김민규 (0.084), 임세민 (0.073)  
1    윤상민 (0.130), 임세민 (0.130), 박하명 (0.112)  
2    권은지 (0.122), 안인규 (0.113), 배홍균 (0.085)  
3    박하명 (0.111), 이현기 (0.101), 김민규 (0.084)  
4    윤명근 (0.286), 배홍균 (0.147), 김장호 (0.057)  
5    윤상민 (0.308), 권은지 (0.064), 김영욱 (0.034)  
6    최상혁 (0.026), 임세민 (0.024), 안인규 (0.017)  
7    김민규 (0.112), 이현기 (0.111), 김장호 (0.111)  
8    김상철 (0.147), 윤명근 (0.110), 김영욱 (0.085)  
9    김영욱 (0.113), 김민규 (0.058), 임세민 (0.054)  
10  김준호  (0.308), 김민규 (0.130), 임세민 (0.064)  
11   김상철 (0.286), 배홍균 (0.110), 김장호 (0.036)  
12   박하명 (0.111), 김장호 (0.101), 김민규 (0.089)  
13   김민규 (0.130), 권은지 (0.073), 윤상민 (0.064)  
14   권은지 (0.067), 임세민 (0.039), 김민규 (0.025)  
15   김영욱 (0.049), 박하명 (0.039), 김민규 (0.032)  
'''
# =========================

df = pd.read_excel("professor_labinfo.csv.xlsx")

name_col = "교수님 성함"
text_cols = [
    "연구실명",
    "사이트에 소개된 관심분야",
    "수업하시는 교과목 (2026-1기준)",
    "우리가 추가한 교수님 특징?"
]
num_col = "학부연구생 수"

# 결측치
df = df[df[name_col].notna()].copy()

for col in text_cols:
    df[col] = df[col].fillna("")

df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
df[num_col] = df[num_col].fillna(df[num_col].median())

# 텍스트전처리
def clean_text(text):
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-zA-Z0-9가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["combined_text"] = (
    df["연구실명"] + " " +
    df["사이트에 소개된 관심분야"] + " " +
    df["수업하시는 교과목 (2026-1기준)"] + " " +
    df["우리가 추가한 교수님 특징?"]
).apply(clean_text)

# 키워드..feat gpt
category_keywords = {
    "하드웨어계열": """
        hardware ai acceleration accelerator edge ai efficient deep learning
        model optimization design automation chip semiconductor embedded system
        low power on-device ai hardware-aware compression pruning quantization
        lightweight 경량화 하드웨어 임베디드 반도체 칩 최적화 가속기
    """,

    "비전/멀티모달계열": """
        computer vision visual computing image video graphics mixed reality
        multimodal medical ai autonomous driving geospatial ai physical ai
        robot audition sound aware multimodal llm pattern recognition
        vision language vision-language cv vr ar mr 영상 이미지 비전 멀티모달
        오디오 그래픽스 컴퓨터비전
    """,

    "추천/데이터마이닝계열": """
        recommender system recommendation collaborative filtering ranking
        data mining large scale data mining machine learning user modeling
        personalization retrieval search analytics graph mining
        추천시스템 추천 데이터마이닝 대규모데이터 데이터분석 랭킹 개인화
    """,

    "시스템/네트워크계열": """
        network networking wireless communication v2x edge computing mobile edge ai
        iot underwater iot distributed system cloud computing operating system
        system security protocol routing communication convergence latency
        네트워크 통신 시스템 엣지컴퓨팅 분산시스템 클라우드 모바일컴퓨팅
        무선통신 수중통신 보안
    """
}

# TF-IDF
# 교수 텍스트 + 계열 키워드 문서를 함께 학습

seed_docs = [clean_text(v) for v in category_keywords.values()]
all_docs = df["combined_text"].tolist() + seed_docs

vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),
    stop_words="english"
)

X_all = vectorizer.fit_transform(all_docs)

X_prof = X_all[:len(df)]
X_seed = X_all[len(df):]

# 숫자형 정보(학부연구생 수) 반영
# 덜 중요한 정보라고 생각해서 약하게 반영
scaler = StandardScaler()
X_num = scaler.fit_transform(df[[num_col]])

# 텍스트 유사도 기반 분류 (코사인 유사도)
sim_matrix = cosine_similarity(X_prof, X_seed)   # (교수 수, 카테고리 수)

category_names = list(category_keywords.keys())
df["predicted_category"] = [category_names[i] for i in sim_matrix.argmax(axis=1)]
df["category_score"] = sim_matrix.max(axis=1)

# 애매한 경우 보정 (텍스트에 특정 키워드많으면 점수올려)
def heuristic_override(text, current_cat):
    t = text.lower()

    hardware_kw = ["hardware", "acceleration", "chip", "semiconductor", "embedded", "경량화", "하드웨어"]
    vision_kw = ["vision", "multimodal", "graphics", "image", "video", "mixed reality", "medical ai", "robot audition", "오디오", "비전", "멀티모달"]
    recomm_kw = ["recommender", "recommendation", "data mining", "ranking", "personalization", "추천", "데이터마이닝"]
    system_kw = ["network", "communication", "iot", "edge computing", "distributed", "wireless", "v2x", "수중", "네트워크", "통신", "시스템"]

    scores = {
        "하드웨어계열": sum(k in t for k in hardware_kw),
        "비전/멀티모달계열": sum(k in t for k in vision_kw),
        "추천/데이터마이닝계열": sum(k in t for k in recomm_kw),
        "시스템/네트워크계열": sum(k in t for k in system_kw),
    }

    best_cat = max(scores, key=scores.get)
    if scores[best_cat] >= 2:   # 키워드가 충분히 많이 걸리면 덮어쓰기
        return best_cat
    return current_cat

df["final_category"] = df.apply(
    lambda row: heuristic_override(row["combined_text"], row["predicted_category"]),
    axis=1
)

# 같은 class 안에서 비슷한 교수 찾기 (KNN)
X_prof_dense = X_prof.toarray()

# 대학원생 수 비중을 너무 크게 하지 않기 위해 0.3 가중치
X_total = np.hstack([X_prof_dense, 0.3 * X_num])

nn = NearestNeighbors(n_neighbors=min(4, len(df)), metric="cosine")
nn.fit(X_total)

distances, indices = nn.kneighbors(X_total)

similar_professors = []
for i in range(len(df)):
    neighbors = []
    for j in range(1, min(4, len(df))):  # 자기 자신 제외
        idx = indices[i][j]
        sim = 1 - distances[i][j]
        neighbors.append(f"{df.iloc[idx][name_col]} ({sim:.3f})")
    similar_professors.append(", ".join(neighbors))

df["비슷한 교수 TOP3"] = similar_professors

# 결과출력
print("\n=== 계열별 교수 분류 결과 ===")
for cat in category_names:
    members = df[df["final_category"] == cat][name_col].tolist()
    print(f"\n[{cat}]")
    for m in members:
        print("-", m)

print("\n=== 교수별 상세 결과 ===")
print(df[[name_col, "predicted_category", "category_score", "final_category", "비슷한 교수 TOP3"]])

# 저장
df.to_csv("professor_category_grouped.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: professor_category_grouped.csv")