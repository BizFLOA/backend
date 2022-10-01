import pandas as pd
import numpy as np
import pymysql

import plotly
import plotly.express as px
import plotly.graph_objects as go
import json


mysql_id = "ict1"
mysql_pwd = "ict1"
db_name = "ict1"
table_name = "prepv1_budget_20220830"


def get_df():
    df = None
    # load raw dataset
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
    return df

df = get_df()
df.head()
df["budget"] = df["예산"].apply(lambda x: int(x.split()[0]))





fig = px.treemap(df, path=[px.Constant("대분류"), '지원분야', '관리기관'], values='budget',
                 color='budget', hover_data=['관리기관'],
                 color_continuous_scale='RdBu',
                 color_continuous_midpoint=np.average(df['budget'], weights=df['budget']))
# fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

fig = px.treemap(df, path=[px.Constant("대분류"), '지원분야'], values='budget')

fig.show()
# fig

go.treemap()


# graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)





