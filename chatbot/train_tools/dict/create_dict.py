"""
우밍에서 사용할 사용자 사전(USERDIC) 파일 생성

테스트 과정에서 사용한 단어가 사전에 존재하지 않는다면 사전을 업데이트해야 함.
OOV(미등록어)처리가 많아지면 품질이 떨어짐.
또한 데이터를 \t 으로 구분할 것.
"""

from utils.Preprocess import Preprocess
from keras import preprocessing
import pickle


# 말뭉치 데이터 읽어오기
def read_corpus_data(filename):
	with open(filename, 'r') as f:
		data = [line.split("\t") for line in f.read().splitlines()]
		data = data[1:]  # 헤더 제거
	return data


# 말뭉치 데이터 가져오기
corpus_data = read_corpus_data("./corpus.txt")



# 말뭉치 데이터에서 키워드만 추출해서 사전 리스트 생성
p = Preprocess()	# 전처리 객체 생성
dict = []
for c in corpus_data:
	pos = p.pos(c[1])
	for k in pos:
		dict.append(k[0])  # k[0] -> tag는 빼버리고 형태소만 가져오기

# 사전에 사용될 word2index 생성
# 사전의 첫 번째 인덱스는 OOV로 사용(OOV == out of vocab,미등록어), 또한 인덱스는 1부터 시작
tokenizer = preprocessing.text.Tokenizer(oov_token="OOV")
tokenizer.fit_on_texts(dict)
word_index = tokenizer.word_index

"""
word_index 예시) 
{"love": 1, "my": 2, "i": 3, "dog": 4, ... }
"""

# 사전 파일 생성
f = open("chatbot_dict.bin", "wb")
try:
	pickle.dump(word_index, f)
except Exception as e:
	print(e)
finally:
	f.close()
