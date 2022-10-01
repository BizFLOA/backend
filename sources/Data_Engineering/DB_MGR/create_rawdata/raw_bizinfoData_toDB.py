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

folder_path = "./Data_Engineering/DB_MGR/"
rawdata = pd.concat(
    [pd.read_excel(folder_path + "rawdata/20220929/bizinfo_20220929_1329_old.xlsx"),
     pd.read_excel(folder_path + "rawdata/20220929/bizinfo_20220929_1329_present.xlsx")],
    axis=0, ignore_index=True
)
# rawdata.to_csv(folder_path + "/prepdata/prep_20220830지원사업.csv", index=False, encoding="utf8")
# rawdata.info()
# print("Done")
rawdata.info()
rawdata = rawdata.sort_values(["등록일자"], ascending=False)
rawdata["번호"] = range(len(rawdata))

mysql_id = "ict1"
mysql_pwd = "ict1"
table_name = "raw_bizinfo_20220929"


# with open(folder_path + "prepdata/bizinfo_idx_counter.pkl", "wb") as f:
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

try:
    with conn.cursor() as cursor:
        sql = f"create table {table_name} (" \
              "번호 INT not null primary key," \
              "소관부처 VARCHAR(32)," \
              "수행기관 VARCHAR(32)," \
              "지원분야 VARCHAR(32)," \
              "공고명 VARCHAR(128)," \
              "신청시작일자 DATE," \
              "신청종료일자 DATE," \
              "등록일자 DATE," \
              "공고상세URL VARCHAR(1024)" \
              ");"
        cursor.execute(sql)
except:
    with conn.cursor() as cursor:
        cursor.execute(f"drop table {table_name}")
        sql = f"create table {table_name} (" \
              "번호 int not null primary key," \
              "소관부처 VARCHAR(32)," \
              "수행기관 VARCHAR(32)," \
              "지원분야 VARCHAR(32)," \
              "공고명 VARCHAR(128)," \
              "신청시작일자 DATE," \
              "신청종료일자 DATE," \
              "등록일자 DATE," \
              "공고상세URL VARCHAR(1024)" \
              ");"
        cursor.execute(sql)
    print("already exists - recreate")


# with conn.cursor() as cursor:
#     cursor.execute("truncate raw_bizinfo_20220830")
# with conn.cursor() as cursor:
#     cursor.execute("select * from raw_bizinfo_20220830")
# conn.commit()

params = ",".join(["%s"] * len(rawdata.columns))
try:
    with conn.cursor() as cursor:
        for idx, value in tqdm(rawdata.iterrows()):
            sql = f"INSERT INTO {table_name} VALUES " \
                  f"({params})"
            cursor.execute(sql, tuple([str(i) for i in value.to_list()]))
        conn.commit()
except:
    pass
finally:
    conn.close()

print("done")

