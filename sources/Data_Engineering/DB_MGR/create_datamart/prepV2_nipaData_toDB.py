import os
import sys
# sys.path.append("/data_prep/DB_MGR")
# sys.path.append("/data_prep/Classifier_지원분야")
sys.path.append("C:\\Users\\take\\PycharmProjects\\ICT_competition_mysql\\Data_Engineering\\DB_MGR")
sys.path.append("C:\\Users\\take\\PycharmProjects\\ICT_competition_mysql\\Data_Engineering\\Classifier_지원분야\\sources")

import numpy as np
import pandas as pd
import pickle
from pandas import DataFrame as dataframe, Series as series
import pymysql
from tqdm import tqdm
from main_category_inference import get_category
from bizinfo_getAttachment_fromURL_FUNC_v2 import bizinfo_get_attm_string_from_file
from main_get_grant_from_string_v2 import get_grant_from_string
from main_get_date_from_string_v2 import get_date_from_string

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 200)

folder_path = "/Data_Engineering/"

mysql_id = "ict1"
mysql_pwd = "ict1"
db_name = "ict1"
table_name = "raw_inpa_20220929"
new_table_name = "prepV2_inpa_20220929"

# load raw dataset
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
        df_test_x = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
except:
    print("SQL ERROR")

df_test_x.head()

# preprocessing data
tmp_df = df_test_x[["주관부처", "공고명", "attm_text"]]
tmp_df.columns = ["inst", "title", "attm_text"]

df_test_x["지원분야"] = get_category(tmp_df)

print(df_test_x.head())
print(df_test_x.shape)


from ckonlpy.tag import Twitter as Okt
okt = Okt()
okt.add_dictionary(words=["rd", "ict", "cloud", "bigdata", "digital", "ai", "4차산업", "메타버스"
                          "metaverse", "인공지능", "ar", "vr", "arvr", "블록체인", "빅데이터"], tag='Noun')

stop_words = ["년", "년도", "차", "및", "모집", "추가모집", "추가", "연장", "공고", "재공고", "안내"]
# for test
tmp_words = [okt.nouns(i) for i in df_test_x["공고명"]]
new_tmp_words = []
for i in tmp_words:
    tmp = []
    for j in i:
        if len(j) > 1:
            tmp.append(j)
    new_tmp_words.append(" ".join(tmp))
df_test_x["공고명_명사추출"] = series(new_tmp_words).apply(lambda x: " ".join([i for i in x.split() if (i not in stop_words) & (len(i) > 1)]))
df_test_x.head()

df_test_x["attm_text"] = df_test_x["attm_text"].apply(lambda x: "" if x is None else x)

tmp = get_date_from_string(df_test_x["attm_text"])

df_test_x[["신청시작일자", "신청종료일자"]] = dataframe(tmp).fillna(pd.to_timedelta(0))
# df_test_x.head()
#
# dataframe(tmp).iloc[0,0]

tmp = get_grant_from_string(df_test_x["attm_text"])

# for idx, file_str in enumerate(tqdm(df_test_x["attm_text"])):
#     try:
#         tmp_df = " ".join(file_str.replace("\n", " ").replace("’", "`").replace("‘", "`").replace("∼", "~").replace(":", ":").replace("`", "20").split())
#         # break
#     except:
#         print(idx)
# df_test_x["attm_text"].iloc[229:235]
# df_test_x["attm_text"].iloc[230] is None

df_test_x["grant_rawstring"] = [i[0] if i[0] is not None else "" for i in tmp]
df_test_x["grant_prepstring"] = [i[2] if i[2] is not None else "" for i in tmp]

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

# add new columns & rename columns
try:
    with conn.cursor() as cursor:
        cursor.execute(f"ALTER TABLE {new_table_name} RENAME COLUMN 주관부처 TO 수행기관")
        cursor.execute(f"ALTER TABLE {new_table_name} ADD 지원분야 VARCHAR(32)")
        cursor.execute(f"ALTER TABLE {new_table_name} ADD 공고명_명사추출 LONGTEXT")
        cursor.execute(f"ALTER TABLE {new_table_name} ADD 신청시작일자 DATE default null")
        cursor.execute(f"ALTER TABLE {new_table_name} ADD 신청종료일자 DATE default null")
        cursor.execute(f"ALTER TABLE {new_table_name} ADD grant_rawstring VARCHAR(64)")
        cursor.execute(f"ALTER TABLE {new_table_name} ADD grant_prepstring VARCHAR(64)")
except:
    print("DB alter error")

df_test_x.info()

params = ",".join(["%s"] * len(df_test_x.columns))
# add values to new columns
try:
    with conn.cursor() as cursor:
        for idx, value in tqdm(df_test_x.iterrows()):
            sql = f"INSERT INTO {new_table_name} VALUES " \
                  f"({params})"
            cursor.execute(sql, tuple([None if i == pd.to_timedelta(0) else str(i) for i in value.to_list()]))
except:
    print("DB insert Error")

conn.commit()
conn.close()

