import dateutil.parser as dparser
import datefinder
import re

convert_dic = {
    "신청기간": ["신청기간", "본 접수 기간", "신청서 제출기간", "공고기간", "접수기간", "조사기간", "온라인 신청 제출기간"]
}

test_cases = [
    "신청기간 : 2022.3.4.(금) ~ 2022.4.4(월) 10:00까지",
    "신청서 제출기간 : 2022년 5월 11일(수) ~ 31일(화)",
    "조사기간 : 2022. 1. 20.(목) ~ 2022. 02. 08.(화) 16:00 시까지",
    "신청서 제출기간 : 2022년 8월 12일(금)~ 8월 24일(수)(15시까지)",
    "신청서 제출기간 : 2022년 5월 31일(화) 이후 수시 신청",
    "신청기간 : 2022년 6월 17일(금) ~ 7월 29일(금)",
    "(공고기간) 2022. 2. 28(월) ~ 2022. 4. 5(화) (접수마감) 2022. 4. 5(화) 16:00까지",
    "(공고기간) 2022. 6. 7(화) ~ 2022. 7. 6(수) (접수마감) 2022. 7. 6(수) 16:00까지",
    "(접수기간) ’22.2.22(화)~’22.3.31(목) 16:00까지",
    "신청기간 :‘22. 1. 26.(수) ~ 2. 25(금) 총 30일간"
]

folder_path = "C:\\정부사업공고\\Download_Files"

with open("C:\\Users\\take\\Desktop\\[붙임1] 2022년 인공지능 온라인 경진대회 공고_최종.txt", encoding="utf-8") as f:
    test_cases = [f.read().replace("\n", " ").replace("`", "20")]

# len(df.split("신청기간"))

# len(df.split("신청기간"))
result = []
for root_idx, df in enumerate(test_cases):
    max_len = 40

    for target_word in convert_dic.keys():
        for replaced_word in convert_dic[target_word]:
            df = df.replace(replaced_word, target_word)

    # if root_idx == 2:
    #     break

    df = df.split("신청기간")[1][:max_len]
    df_list = df.split("~")[:2]
    # if root_idx == 2:
    #     break
    tmp_result = []
    # if root_idx == 2:
    #     break
    for idx, value in enumerate(df_list):
        # 괄호 안 문자 제거 (요일 관련 정보를 제거하기 위함)
        tmp = re.sub(pattern=r'\([^)]*\)', repl='', string=value)
        # 숫자만 추출
        tmp = re.sub('[^0-9]', ' ', tmp)
        # 마침표로 다시 문자열을 합침
        tmp = ".".join(tmp.split())

        # tmp.sp

        # 월.일 형식일 경우 (앞에 년도가 없는 경우)
        if len(tmp.split(".")) < 3:
            # 앞 리스트의 년도인데도 몇 년인지 정보가 없을 경우 기본값으로 2022 설정
            if idx == 0:
                year = "2022"
            # 입력한 year와 기존 월 및 일 정보를 합하여 다시 저장
            tmp = year + "." + tmp
        # 년.월.일 형식의 경우
        else:
            # 물결 표시로 split한 리스트 중 첫번째 일 경우
            if idx == 0:
                # 뒤의 리스트에 년도가 없을 경우를 대비하여 year라는 변수에 년도를 추출하여 할당
                year = ".".join(tmp.split()).split(".")[0]

        # 검색된 datetime을 출력
        for j in datefinder.find_dates(tmp):
            # print(j)
            tmp_result.append(j)
    result.append(tuple(tmp_result))

# len(test_cases)
# len(result)

# for i, j in zip(test_cases, result):
#     print(i)
#     print(j)
#     print("\n")
#
#
#
#
#
#
#
# match = re.search(r'\d{4}.?\d{1,2}.?\d{1,2}', "2003.44.11")
# match
#
#
#
# match = re.search(r'\d{1,2}.?\d{1,2}', "2003.44.11")
# match








