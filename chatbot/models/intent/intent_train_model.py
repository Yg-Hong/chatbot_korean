import keras_preprocessing.sequence
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate

from utils.Preprocess import Preprocess
from config.GlobalParams import MAX_SEQ_LEN, INTENT_SEQ, INTENT_NUM
from config.Dictation import WORD2INDEX_DIC, USERDIC

# 데이터 읽어오기
train_file = "../../train_tools/qna/train_data.csv"
data = pd.read_csv(train_file, delimiter=',')
queries = data["query"].tolist()
intents = data["intent"].tolist()


# 전처리 과정
p = Preprocess(word2index_dic=WORD2INDEX_DIC, userdic=USERDIC)

# 단어 시퀀스 생성
sequences = []
for sentence in queries:
	pos = p.pos(sentence)
	keywords = p.get_keywords(pos, without_tag=True)
	seq = p.get_wordidx_sequence(keywords)
	sequences.append(seq)

# 단어 인덱스 시퀀스 벡터 생성
# 단어 시퀀스 벡터 크기 = Config의 MAX_SEQ_LEN
padded_seqs = keras_preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

"""
# (5231, 15)
print(padded_seqs.shape)
print(len(intents)) #5231
print(padded_seqs.tolist())
"""

# 학습용, 검증용, 테스트용 데이터셋 생성
# 데이터를 랜덤으로 섞고, 학습셋:검증셋:테스트셋 = 7:2:1 비율로 나눔
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, INTENT_SEQ))
ds = ds.shuffle(len(queries))

train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)

train_ds = ds.take(train_size).batch(25)
val_ds = ds.skip(train_size).take(val_size).batch(25)
test_ds = ds.skip(train_size + val_size).take(test_size).batch(25)

# 하이퍼파라미터 설정
dropout_prob = 0.5  # 50% 확률로 dropout -> 학습 과정에서 발생하는 오버피팅(과적합)에 대비
EMB_SIZE = 128  # 임베딩 결과로 나온 밀집 벡터의 크기
EPOCH = 40
VOCAB_SIZE = len(p.word_index) + 1  # 전체 단어 수

# CNN 모델 정의
input_layer = Input(shape=(MAX_SEQ_LEN,))
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layer)
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

conv1 = Conv1D(  # kernel 크기가 3인 합성곱 필터 128개를 이용한 계층
	filters=128,
	kernel_size=3,
	padding='valid',
	activation=tf.nn.relu)(dropout_emb)
pool1 = GlobalMaxPool1D()(conv1)  # 최대 풀링 연산

conv2 = Conv1D(  # kernel 크기가 4인 합성곱 필터 128개를 이용한 계층
	filters=128,
	kernel_size=4,
	padding='valid',
	activation=tf.nn.relu)(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(  # kernel 크기가 5인 합성곱 필터 128개를 이용한 계층
	filters=128,
	kernel_size=5,
	padding='valid',
	activation=tf.nn.relu)(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)

# 합성곱 계층의 특징맵 결과(pool1, pool2, pool3)를 하나로 묶음
concat = concatenate([pool1, pool2, pool3])

hidden = Dense(128, activation=tf.nn.relu)(concat)  	# 128개의 출력 노드를 가지고, relu 함수로 평탄화된 Dense 계층(hidden) 생성
dropout_hidden = Dropout(rate=dropout_prob)(hidden)  	# 50%의 확률로 dropout 함수 사용, 오버피팅 방지
logits = Dense(INTENT_NUM, name='logits')(dropout_hidden)  # 출력 노드에서 INTENT_NUM 만큼의 점수가 출력되고 가장 큰 점수가 결과
predictions = Dense(INTENT_NUM, activation=tf.nn.softmax)(logits)


# 모델 생성
model = Model(inputs=input_layer, outputs=predictions)
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)


# 모델 평가(테스트 데이터셋 이용)
loss, accuracy = model.evaluate(test_ds, verbose=1)
print("Accuracy : %f" % (accuracy * 100))
print("loss : %f" % loss)

# 모델 저장
model.save("intent_model.h5")
