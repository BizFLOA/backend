import os
import sys
sys.path.append("/Data_Engineering/DB_MGR")
sys.path.append("/Data_Engineering/Classifier_지원분야")

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

folder_path = ".\\Data_Engineering\\DB_MGR\\"

rawdata_central = pd.read_csv(folder_path + "rawdata\\test_202208030\\예산할당_2022년_중소기업지원사업_현황대상(중앙부처).csv", encoding="cp949")
rawdata_division = pd.read_csv(folder_path + "rawdata\\test_202208030\\예산할당_2022년_중소기업지원사업_현황대상(지자체).csv", encoding="cp949")

rawdata_central.columns = ["번호", "관리기관", "공고명", "예산"]
rawdata_division.columns = ["번호", "관리기관", "공고명", "예산"]

rawdata = pd.concat([rawdata_central, rawdata_division], ignore_index=True)
rawdata["번호"] = list(range(1, len(rawdata)+1))
rawdata["예산"] = rawdata["예산"].apply(lambda x: str(x) + " 백만원")

rawdata.head()

mysql_id = "ict1"
mysql_pwd = "ict1"
table_name = "raw_budget_20220830"

# idx_counter = 1
# rawdata["번호"] = list(range(idx_counter, len(rawdata)+1, 1))
# rawdata = rawdata[["번호"] + list(rawdata.columns[:-1])]

# with open(folder_path + "prepdata/inpa_idx_counter.pkl", "wb") as f:
#     pickle.dump(rawdata["번호"].max() + 1, f)

# insert initial dataset
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

# rawdata.info()

# with conn.cursor() as cursor:
#     cursor.execute("drop table raw_inpa_20220830")
# # with conn.cursor() as cursor:
# #     cursor.execute("select * from raw_inpa_20220830")
# conn.commit()

try:
    with conn.cursor() as cursor:
        sql = f"create table {table_name} ( " \
              "번호 int not null primary key," \
              "주관부처 VARCHAR(32)," \
              "공고명 VARCHAR(128)," \
              "예산 VARCHAR(64)" \
              ");"
        cursor.execute(sql)
except:
    print("already exists - recreate")
    with conn.cursor() as cursor:
        cursor.execute(f"drop table {table_name}")
        sql = f"create table {table_name} ( " \
              "번호 int not null primary key," \
              "주관부처 VARCHAR(32)," \
              "공고명 VARCHAR(128)," \
              "예산 VARCHAR(64)" \
              ");"
        cursor.execute(sql)
finally:
    conn.commit()



# with conn.cursor() as cursor:
#     cursor.execute("truncate raw_inpa_20220830")
# with conn.cursor() as cursor:
#     cursor.execute("select * from raw_inpa_20220830")
# conn.commit()

params = ",".join(["%s"] * len(rawdata.columns))
with conn.cursor() as cursor:
    for idx, value in tqdm(rawdata.iterrows()):
        try:
            sql = f"INSERT INTO {table_name} VALUES " \
                  f"({params})"
            cursor.execute(sql, tuple([str(i) for i in value.to_list()]))
        except:
            print("insert error", idx, value)

conn.commit()

conn.close()
print("done")
