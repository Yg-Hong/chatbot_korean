import keras_preprocessing.sequence
import tensorflow as tf
from keras.models import load_model
import keras_preprocessing.sequence

from config.GlobalParams import MAX_SEQ_LEN, INTENTS_DICT


# 의도 분류 모델 모듈
class IntentModel:
	def __init__(self, model_name, proprocess):
		# 의도 클래스별 레이블
		self.labels = INTENTS_DICT

		# 의도 분류 모델 불러오기
		self.model = load_model(model_name)

		# 우밍 preprocess 객채
		self.p = proprocess

	# 의도 예측 클래스
	def predict_class(self, query):
		# 형태소 분석
		pos = self.p.pos(query)

		# 문장 내 키워드 추출(불용어 제거)
		keywords = self.p.get_keywords(pos, without_tag=True)
		sequences = [self.p.get_wordidx_sequence(keywords)]

		# 패딩 처리
		padded_seqs = keras_preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="post")

		predict = self.model.predict(padded_seqs)
		predict_class = tf.math.argmax(predict, axis=1)
		return predict_class.numpy()[0]
