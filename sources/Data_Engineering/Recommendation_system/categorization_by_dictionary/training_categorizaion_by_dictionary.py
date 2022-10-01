import os
# from ckonlpy.tag import Twitter as Okt
import pymysql
import pandas as pd
import numpy as np
import re

folder_path = "./Data_Engineering/Recommendation_system/categorization_by_dictionary/"

def get_categorization_dictionary_it(df):
    word_list = ['ict', 'iot', 'ai', 'sw', '스마트', '인공지능', '첨단', '데이터', '클러스트', '로봇산업', '로봇', '드론', '메타', '메타버스', '빅데이터', '반도체']
    return df.loc[df['공고명'].apply(lambda x: x.lower()).str.contains('|'.join(word_list))].reset_index(drop=True)


def get_categorization_dictionary_echo(df):
    word_list = ['에너지', '탄소', '그린', '녹색', '수소', '친환경', '공기', '신재생에너지', '신재생 에너지']
    return df.loc[df['공고명'].apply(lambda x: x.lower()).str.contains('|'.join(word_list))].reset_index(drop=True)


def get_categorization_dictionary_bio(df):
    word_list = ['바이오', '화장품', '마이크로바이옴', '마이크로 바이옴', '의료기기', '메디케어', '헬스', '병원', '의료', '향체', '의약품', '생명공학', '치의학', '안과']
    return df.loc[df['공고명'].apply(lambda x: x.lower()).str.contains('|'.join(word_list))].reset_index(drop=True)


def get_categorization_dictionary_smallcap(df):
    word_list = ["중소기업", "소기업", "스타트업", "벤처기업", "벤처 기업" "엔젤", "엔젤투자", "엔젤 투자", "벤처투자", "벤처 투자"]
    return df.loc[df['공고명'].apply(lambda x: x.lower()).str.contains('|'.join(word_list))].reset_index(drop=True)


theme_dictionary = {
    "4차산업": get_categorization_dictionary_it,
    "친환경": get_categorization_dictionary_echo,
    "bio&health": get_categorization_dictionary_bio,
    "중소기업": get_categorization_dictionary_smallcap,
}

mysql_id = "ict1"
mysql_pwd = "ict1"
db_name = "ict1"
table_name = "mlv2_20220929"

# create data pipeline
try:
    conn = pymysql.connect(
        host='localhost',  # 호출변수 정의
        user=mysql_id,
        password=mysql_pwd,
        db="ict1",
        charset='utf8mb4'
    )
except:
    print("ERROR : DB Connection")

try:
    with conn.cursor() as cursor:
        query = "SELECT * FROM " + table_name
        cursor.execute(query)
        df = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
except:
    print("SQL ERROR")
conn.close()

df = df.fillna(np.nan)
df["등록일자"] = pd.to_datetime(df["등록일자"])
df["신청시작일자"] = pd.to_datetime(df["신청시작일자"])
df["신청종료일자"] = pd.to_datetime(df["신청종료일자"])
df = df.sort_values("등록일자", ascending=False)
df.shape
df.tail

# === export samples (export raw string) & define cluster property ===
cluster_sample_path = folder_path + "cluster_sample/"
if not os.path.exists(cluster_sample_path):
    os.makedirs(cluster_sample_path)

for i in theme_dictionary.keys():
    cluster_path = cluster_sample_path + i + "/"
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)
    tmp_df = theme_dictionary[i](df.copy())
    tmp_df["dictionary_cluster"] = i
    for j in tmp_df.sample(min(len(tmp_df), 50), random_state=42).iterrows():
        with open(cluster_path + "idx_" + str(j[0]) + "&title_" + j[1]["공고명"].replace("/", "_") + ".txt", "w", encoding="utf8") as f:
            f.write(re.sub(r'\n+', '\n', j[1]["attm_text"]).strip())
    tmp_df.to_csv(cluster_path + "output.csv", index=False, encoding="utf8")

tmp_df.head()

# print(tmp_df.info())
