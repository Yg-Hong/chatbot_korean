from utils.Preprocess import Preprocess
import numpy as np
import tensorflow as tf
import keras.utils
from keras import preprocessing
import keras_preprocessing.sequence
from sklearn.model_selection import train_test_split
from config.Dictation import WORD2INDEX_DIC, USERDIC

"""
# 파일 형식에 맞게 read_file 함수 수정할 것.
def read_file(file_name):
	sents = []
	with open(file_name, 'r', encoding='utf-8') as f:
		lines = f.readlines()
		for idx, l in enumerate(lines):
			if l[0] == ';' and lines[idx + 1][0] == '$':
				this_sent = []
			elif l[0] == '$' and lines[idx - 1][0] == ';':
				continue
			elif l[0] == '\n':
				sents.append(this_sent)
			else:
				this_sent.append(tuple(l.split()))
	return sents
"""


# 학습 파일 불러오기
def read_file(file_name):
	sents = []
	this_sent = []
	with open(file_name, 'r', encoding="utf-8") as f:
		lines = f.readlines()
		for l in lines:
			if l[0] == '\n':
				sents.append(this_sent)
				this_sent = []
			else:
				this_sent.append(tuple(l.split()))
	return sents


# 전처리 객체 생성
p = Preprocess(word2index_dic="../../train_tools/dict/chatbot_dict.bin", userdic="../../train_tools/dict/NIADic2Komoran.tsv")

# 학습용 말뭉치 데이터를 불러옴
corpus = read_file("NER_train_data.txt")


# 말뭉치 데이터에서 단어와 BIO 태그만 불러와 학습용 데이터셋 생성
sentences, tags = [], []
for t in corpus:
	tagged_sentence = []
	sentence, Bio_tag = [], []
	for w in t:
		tagged_sentence.append((w[1], w[2]))  # 파일 형식에 맞게 index 수정.
		sentence.append(w[1])
		Bio_tag.append(w[2])
	sentences.append(sentence)
	tags.append(Bio_tag)


# 불필요한 태그 삭제
del_tag = ["PLT_B", "PLT_I", "ANM_B", "ANM_I", "MAT_B", "MAT_I", "TRM_B", "TRM_I", "FLD_B", "FLD_I", "CVL_B", "CVL_I"]
for line in tags:
	for idx in range(len(line)):
		if line[idx] in del_tag:
			line[idx] = '-'


# tag 종류 확인
tag_list = []
for element in tags:
	for tag in element:
		if tag not in tag_list:
			tag_list.append(tag)
tag_list = set(tag_list)
print(tag_list)


print("샘플 크기 : \n", len(sentences))
print("0번째 샘플 단어 시퀀스 : \n", sentences[0])
print("0번째 샘플 BIO 태그 : \n", tags[0])
print("샘플 단어 시퀀스 최대 길이 : ", max(len(l) for l in sentences))  # 패딩 크기 조정
print("샘플 단어 시퀀스 평균 길이 : ", (sum(map(len, sentences)) / len(sentences)))  # 패딩 크기 조정


# 토크나이저 정의
tag_tokenizer = preprocessing.text.Tokenizer(lower=False)  # 태그 정보는 lower = False (소문자로 변환하지 않는다)
tag_tokenizer.fit_on_texts(tags)

# 단어 사전 및 태그 사전 크기
vocab_size = len(p.word_index) + 1
tag_size = len(tag_tokenizer.word_index) + 1

print("BIO 태그 사전 크기 : ", tag_size)
print("단어 사전 크기 : ", vocab_size)

# 학습용 사전 데이터를 시퀀스 번호 형태로 인코딩
x_train = [p.get_wordidx_sequence(sent) for sent in sentences]
y_train = tag_tokenizer.texts_to_sequences(tags)

index_to_ner = tag_tokenizer.index_word  # 시퀀스 인덱스를 NER로 변환하기 위해 사용
index_to_ner[0] = "PAD"

# 시퀀스 패딩 처리
max_len = 25
x_train = keras_preprocessing.sequence.pad_sequences(x_train, padding="post", maxlen=max_len)
y_train = keras_preprocessing.sequence.pad_sequences(y_train, padding="post", maxlen=max_len)

# 학습 데이터와 테스트 데이터를 8:2 비율로 분리
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=.2, random_state=1234)

# 출력 데이터를 원-핫 인코딩
y_train = keras.utils.to_categorical(y_train, num_classes=tag_size)
y_test = keras.utils.to_categorical(y_test, num_classes=tag_size)

print("학습 샘플 시퀀스 형상 : ", x_train.shape)
print("학습 샘플 레이블 형상 : ", y_train.shape)
print("테스트 샘플 시퀀스 형상 : ", x_test.shape)
print("테스트 샘플 레이블 형상 : ", y_test.shape)


# 모델 정의(Bi-LSTM)
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.optimizers import Adam

print("모델 정의 시작")

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=30, input_length=max_len, mask_zero=True))
model.add(Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25)))
model.add(TimeDistributed(Dense(tag_size, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=5)

print("평가 결과 : ", model.evaluate(x_test, y_test)[1])
model.save("ner_model.h5")


# 시퀀스를 NER 태그로 변환
def sequences_to_tag(sequences):  # 예측값을 index_to_ner를 사용하여 태깅 정보로 변경하는 함수
	result = []
	for sequence in sequences:  # 전체 시퀀스로부터 시퀀스를 하나씩 꺼낸다.
		temp = []
		for pred in sequence:  # 시퀀스로부터 예측값을 하나씩 꺼낸다.
			pred_index = np.argmax(pred)  # 예를 들어 [0, 0, 1, 0, 0]이라면 1의 인덱스인 2를 리턴한다.
			temp.append(index_to_ner[pred_index].replace("PAD", "0"))  # 'PAD'는 '0'으로 변경
		result.append(temp)
	return result


# F1 스코어 계산을 위해 사용
from seqeval.metrics import f1_score, classification_report

# 테스트 데이터셋의 NER 예측
y_predicted = model.predict(x_test)
pred_tags = sequences_to_tag(y_predicted)
test_tags = sequences_to_tag(y_test)

# F1 평가 결과
print(classification_report(test_tags, pred_tags))
print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))
