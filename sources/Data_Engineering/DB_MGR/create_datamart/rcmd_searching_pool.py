import numpy as np
import pandas as pd
import pickle
from pandas import DataFrame as dataframe, Series as series
import pymysql
from tqdm import tqdm

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 200)

folder_path = "/Data_Engineering/DB_MGR/"

mysql_id = "ict1"
mysql_pwd = "ict1"
db_name = "ict1"
table_name = "mlv2_20220929"
new_table_name = "searching_20220929"

try:
    conn = pymysql.connect(
        host='localhost',  # 호출변수 정의
        user=mysql_id,
        password=mysql_pwd,
        db=db_name,
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



df = df.fillna(np.nan)
df["등록일자"] = pd.to_datetime(df["등록일자"])
df["신청시작일자"] = pd.to_datetime(df["신청시작일자"])
df["신청종료일자"] = pd.to_datetime(df["신청종료일자"])
df["grant_rawstring"] = df["grant_rawstring"].fillna("")
df["grant_prepstring"] = df["grant_prepstring"].fillna("")
df = df.sort_values("등록일자", ascending=False)
df.head()

df.shape
# ori_shape = df.shape




cluster_root_path = "C:\\Users\\take\\PycharmProjects\\ICT_competition_mysql\\Data_Engineering\\Recommendation_system\\categorization_by_dictionary\\cluster_sample\\"
df_rcmd_keyword = ["4차산업", "친환경", "bio&health", "중소기업"]
df_rcmd = []
for i in range(len(df_rcmd_keyword)):
    print("cluster", i)
    tmp_df = pd.read_csv(cluster_root_path + df_rcmd_keyword[i] + "\\output.csv", encoding="utf8")
    tmp_df["cluster_keyword"] = df_rcmd_keyword[i]
    df_rcmd.append(tmp_df)
df_rcmd = pd.concat(df_rcmd, axis=0, ignore_index=True)

df["cluster_keyword"] = ""
for idx, value in enumerate(df["번호"]):
    tmp = df_rcmd.loc[(df_rcmd["번호"] == value), "cluster_keyword"]
    if len(tmp) > 0:
        df["cluster_keyword"].iloc[idx] =  tmp.iloc[0]
    else:
        df["cluster_keyword"].iloc[idx] = "none"


cluster_root_path = "C:\\Users\\take\\PycharmProjects\\ICT_competition_mysql\\Data_Engineering\\Recommendation_system\\categorization_tfidf_kmeans\\지원분야_첨부파일내용_tfidf_svd_v1_inf_20220930\\cluster_sample\\"
df_rcmd_keyword = ["연구개발", "none", "사회적기업", "자금지원", "none", "박람회", "해외수출", "창업", "일자리", "소상공인"]
df_rcmd = []
for i in range(len(df_rcmd_keyword)):
    print("cluster", i)
    tmp_df = pd.read_csv(cluster_root_path + "cluster_" + str(i) + "\\output.csv", encoding="utf8")
    tmp_df["cluster_kmeans"] = df_rcmd_keyword[i]
    df_rcmd.append(tmp_df)
    # break
df_rcmd = pd.concat(df_rcmd, axis=0, ignore_index=True)


df["cluster_kmeans"] = ""
for idx, value in enumerate(df["번호"]):
    tmp = df_rcmd.loc[(df_rcmd["번호"] == value), "cluster_kmeans"]
    if len(tmp) > 0:
        df["cluster_kmeans"].iloc[idx] = tmp.iloc[0]
    else:
        df["cluster_kmeans"].iloc[idx] = "none"

df.info()
df.head()
df = df.sort_values("번호")


# copy and create new table
try:
    with conn.cursor() as cursor:
        cursor.execute(f"create table {new_table_name} select * from {table_name}")
        cursor.execute(f"TRUNCATE TABLE {new_table_name}")
except:
    print("DB Copy Error - recreate")
    with conn.cursor() as cursor:
        cursor.execute(f"drop table {new_table_name}")
        cursor.execute(f"create table {new_table_name} select * from {table_name}")
        cursor.execute(f"TRUNCATE TABLE {new_table_name}")

# with conn.cursor() as cursor:
#     cursor.execute(f"TRUNCATE TABLE {new_table_name}")


# add new columns & rename columns
try:
    with conn.cursor() as cursor:
        cursor.execute(f"ALTER TABLE {new_table_name} ADD cluster_keyword VARCHAR(32)")
        cursor.execute(f"ALTER TABLE {new_table_name} ADD cluster_kmeans VARCHAR(32)")
except:
    print("DB alter error")

params = ",".join(["%s"] * len(df.columns))
# add values to new columns
try:
    with conn.cursor() as cursor:
        for idx, value in tqdm(df.iterrows()):
            sql = f"INSERT INTO {new_table_name} VALUES " \
                  f"({params})"
            cursor.execute(sql, tuple([None if pd.isna(i) else str(i) for i in value.to_list()]))
except:
    print("Value insert error")

conn.commit()
conn.close()


