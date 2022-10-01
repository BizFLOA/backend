import copy
import re
from pykospacing import Spacing

def prep1_lowerEng_removeNewline(df):
    print("prep1_lowerEng_removeNewline start")
    # cols : title, attm_text
    # 영어를 모두 소문자로 변환 및 개행문자를 공백으로 변환
    if "title" in df.columns:
        df["title"] = df["title"].apply(lambda x: x.lower())
        df["title"] = df["title"].apply(lambda x: " ".join(x.replace("\n", " ").replace("\t", " ").replace("_x000d_", " ").split()))

    if "attm_text" in df.columns:
        df["attm_text"] = df["attm_text"].apply(lambda x: x.lower())
        df["attm_text"] = df["attm_text"].apply(lambda x: " ".join(x.replace("\n", " ").replace("\t", " ").replace("_x000d_", " ").split()))
    print("prep1_lowerEng_removeNewline end")
    return df
def prep2_remove_prefix(df):
    print("prep2_remove_prefix start")
    # cols : title, attm_text
    # K-Drama, K-Bio 처럼 'K-' 접두사 제거 및 [충북] 과 같은 단어 제거
    if "title" in df.columns:
        prep_str = []
        for i in df["title"]:
            tmp = copy.deepcopy(i)
            # index = None
            while True:
                index = re.search(r'k-[a-z가-힣]', tmp)
                if index is None:
                    break
                else:
                    target_pos = index.span()
                    tmp = "".join([tmp[:(target_pos[0])], tmp[(target_pos[1] - 1):]])
            prep_str.append(tmp)
        df["title"] = prep_str

        prep_str = []
        for i in df["title"]:
            if i.lstrip()[0] == "[":
                tmp = i.split("]")
                prep_str.append(tmp[0]) if len(tmp) == 1 else prep_str.append(" ".join(tmp[1:]))
            else:
                prep_str.append(i)
        df["title"] = prep_str

    if "attm_text" in df.columns:
        prep_str = []
        for i in df["attm_text"]:
            tmp = copy.deepcopy(i)
            while True:
                index = re.search(r'k-[a-z가-힣]', tmp)
                if index is None:
                    break
                else:
                    target_pos = index.span()
                    tmp = "".join([tmp[:(target_pos[0])], tmp[(target_pos[1] - 1):]])
            prep_str.append(tmp)
        df["attm_text"] = prep_str

    print("prep2_remove_prefix end")
    return df
def prep3_correct_spacing(df):
    print("prep3_correct_spacing start")
    # cols : title
    # 훈련 및 테스트 데이터 내 공고명 띄어쓰기 교정
    if "title" in df.columns:
        spacing = Spacing(rules=['r&d', "s&m", "g&a"])
        df["title"] = df["title"].apply(lambda x: spacing(x))
    print("prep3_correct_spacing end")
    return df
def prep4_remove_special_char(df):
    print("prep4_remove_special_char start")
    # cols : title, attm_text
    # 영어 내 R&D, S&M 과 같은 약어에서 특수문자에서 &를 제거
    if "title" in df.columns:
        prep_str = []
        for i in df["title"]:
            tmp = copy.deepcopy(i)
            while True:
                index = re.search(r'[a-z]&[a-z]', tmp)
                if index is None:
                    break
                else:
                    target_pos = index.span()
                    tmp = "".join([tmp[:(target_pos[0] + 1)], tmp[(target_pos[0] + 2):]])
            prep_str.append(tmp)
        df["title"] = prep_str

    if "attm_text" in df.columns:
        prep_str = []
        for i in df["attm_text"]:
            tmp = copy.deepcopy(i)
            while True:
                index = re.search(r'[a-z]&[a-z]', tmp)
                if index is None:
                    break
                else:
                    target_pos = index.span()
                    tmp = "".join([tmp[:(target_pos[0] + 1)], tmp[(target_pos[0] + 2):]])
            prep_str.append(tmp)
        df["attm_text"] = prep_str
    print("prep4_remove_special_char end")
    return df
def prep5_remove_content_rightside(df):
    print("prep5_remove_content_rightside start")
    # cols : title
    # 훈련 및 테스트 데이터 내 맨 뒤 괄호 안 내용 제거 - 상위 지원사업 공고명이 여러 세부 공고명 뒤에 일괄적으로 붙는 경우
    # ex. 기술 - 반도체특허(4차산업), 인력 - SW인력지원(4차산업)
    if "title" in df.columns:
        prep_str = []
        for i in df["title"]:
            if i.rstrip()[-1] == ")":
                tmp = i.split("(")
                # 마지막 괄호는 있으나 시작하는 괄호는 없을 경우 그냥 원래 문자열 삽입
                prep_str.append(tmp[0]) if len(tmp) == 1 else prep_str.append(" ".join(tmp[:-1]))
            else:
                prep_str.append(i)
        df["title"] = prep_str
    print("prep5_remove_content_rightside end")
    return df
def prep5_remove_bracket_attmText(df):
    print("prep5_remove_bracket_attmText start")
    # cols : attm_text
    # 첨부 파일의 경우 <> 괄호 안 내용은 필요없는 부분이 많으므로 제거
    # ex. <표>, <서식>, <그림1>, <'URL인코딩문자열'>
    if "attm_text" in df.columns:
        re_obj = re.compile(r'<[^>]*>')
        df["attm_text"] = df["attm_text"].apply(lambda x: re.sub(re_obj, "", x))
    print("prep5_remove_bracket_attmText end")
    return df
def prep6_get_text_KorEng(df):
    print("prep6_get_text_KorEng start")
    # cols : title, attm_text
    # 한글 및 영어만 추출 & 공백 한 칸으로 제한
    if "title" in df.columns:
        prep_str = []
        for i in df["title"]:
            prep_str.append(" ".join(re.sub('[^ a-z가-힣]', ' ', i).split()))
        df["title"] = prep_str

    if "attm_text" in df.columns:
        prep_str = []
        for i in df["attm_text"]:
            prep_str.append(" ".join(re.sub('[^ a-z가-힣]', ' ', i).split()))
        df["attm_text"] = prep_str
    print("prep6_get_text_KorEng end")
    return df
def prep7_spacing_between_KorEng(df):
    print("prep7_spacing_between_KorEng start")
    # cols : title, attm_text
    # 한글과 영어 사이 공백으로 분리
    if "title" in df.columns:
        prep_str = []
        for i in df["title"]:
            tmp = copy.deepcopy(i)
            while True:
                index = re.search('[a-z][가-힣]|[가-힣][a-z]', tmp)
                if index is None:
                    break
                else:
                    target_pos = index.span()
                    tmp = " ".join([tmp[:(target_pos[0] + 1)], tmp[(target_pos[0] + 1):]])
            prep_str.append(tmp)
        df["title"] = prep_str

    if "attm_text" in df.columns:
        prep_str = []
        for i in df["attm_text"]:
            tmp = copy.deepcopy(i)
            while True:
                index = re.search('[a-z][가-힣]|[가-힣][a-z]', tmp)
                if index is None:
                    break
                else:
                    target_pos = index.span()
                    tmp = " ".join([tmp[:(target_pos[0] + 1)], tmp[(target_pos[0] + 1):]])
            prep_str.append(tmp)
        df["attm_text"] = prep_str
    print("prep7_spacing_between_KorEng end")
    return df
def remove_stopwords(df):
    print("remove_stopwords start")
    # remove stop words & only 1 word
    with open("./Data_Engineering/utilities/korean_stopwords.txt", "r",
              encoding="utf8") as f:
        stop_words_ranknl = f.read().split("\n")[:-1]
    stop_words = list(set(stop_words_ranknl + ["년", "년도", "차", "및", "모집", "추가모집", "추가", "연장", "공고", "재공고", "안내", "서식", "표", "작성가이드",
                                               "붙임", "첨부", "페이지", "page", "참여", "참가", "기업", "지원", "사업"]))
    if "content" in df.columns:
        df["content"] = df["content"].apply(lambda x: " ".join([i for i in x.split() if (i not in stop_words) & (len(i) > 1)]))
    else:
        if "title" in df.columns:
            df["title"] = df["title"].apply(lambda x: " ".join([i for i in x.split() if (i not in stop_words) & (len(i) > 1)]))
        if "attm_text" in df.columns:
            df["attm_text"] = df["attm_text"].apply(lambda x: " ".join([i for i in x.split() if (i not in stop_words) & (len(i) > 1)]))
        print("remove_stopwords end")
    return df
def get_nouns(df):
    print("get_nouns start")
    # extract nouns
    from ckonlpy.tag import Twitter as Okt
    okt = Okt()
    okt.add_dictionary(
        words=["rd", "ict", "cloud", "bigdata", "digital", "digital transformation", "digitaltransformation", "딥러닝", "deeplearning", "deep learning",
               "머신러닝", "machine learning", "machinelearning", "디지털", "디지털트랜스포메이션", "디지털 트랜스포메이션", "ai", "4차산업", "메타버스", "sw",
               "metaverse", "meta verse", "인공지능", "인공 지능", "ar", "vr", "xr", "arvr", "블록체인", "blockchain", "block chain", "빅데이터"], tag='Noun'
    )
    # concat title & attachment text
    if ("title" in df.columns) & ("attm_text" in df.columns):
        df["content"] = df["title"] + " " + df["attm_text"]
        df["content"] = df["content"].apply(lambda x: x.rstrip())
        # df["title"] = df["title"].apply(lambda x: " ".join([i for i in okt.nouns(x) if len(i) > 1]))
        # df["attm_text"] = df["attm_text"].apply(lambda x: " ".join([i for i in okt.nouns(x) if len(i) > 1]))
        df["content"] = df["content"].apply(lambda x: " ".join([i for i in okt.nouns(x)]))
    elif "title" in df.columns:
        df["title"] = df["title"].apply(lambda x: " ".join([i for i in okt.nouns(x) if len(i) > 1]))
    elif "attm_text" in df.columns:
        df["attm_text"] = df["attm_text"].apply(lambda x: " ".join([i for i in okt.nouns(x) if len(i) > 1]))
    print("get_nouns end")
    return df