from utils.Preprocess import Preprocess
from models.ner.NerModel import NerModel
from config.Dictation import WORD2INDEX_DIC, USERDIC


p = Preprocess(word2index_dic="../train_tools/dict/chatbot_dict.bin", userdic="../train_tools/dict/NIADic2Komoran/tsv")

ner = NerModel(model_name="../models/ner/ner_model.h5", proprocess=p)
query = "12월 5일 12시 30분에 시작하는 축구 경기를 보고 싶어. 내가 좋아하는 축구 선수인 손흥민도 나와."
predicts = ner.predict(query)
print(predicts)


"""
인물(PER)
학문분야(FLD)
인공물(AFW)
기관 및 단체(ORG)
지역명(LOC)
문명 및 문화(CVL)
날짜(DAT)
시간(TIM)
숫자(NUM)
사건 사고 및 행사(EVT)
동물(ANM)
식물(PLT)
금속/암석/화학물질(MAT)
의학용어/IT관련 용어(TRM)

"""