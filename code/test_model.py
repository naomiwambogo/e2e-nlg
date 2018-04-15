# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle


from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, TimeDistributed, \
										RepeatVector, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from attention_decoder import AttentionDecoder

from utils import extract_feature, \
					delexicalize_tokenize_mr, \
					delexicalize_ref, \
					relexicalize_ref, \
					get_voc, \
					encode, decode, \
					one_hot_encode, \
					one_hot_decode

def main():

	path_test = "../e2e-dataset/testset.csv"
	path_model = "../models/model.h5"
	path_result = "results.txt"

	# ------------------------------
	# ---- LOAD DATA AND MODELS ----
	# ------------------------------

	print("Loading data...", end=" ")
	data = pd.read_csv(path_test, names=["mr"], skiprows=1)
	
	###
	len_seq = 25
	with open('word2idx_mr.pkl', 'rb') as handle:
		w2i_mr = pickle.load(handle)
	with open('idx2word_ref.pkl', 'rb') as handle:
		i2w_ref = pickle.load(handle)

	size_voc_mr = len(w2i_mr.values())
	size_voc_ref = len(i2w_ref.values())
	
	###
	nhid = 128
	model = Sequential()
	model.add(LSTM(nhid, return_sequences=True, input_shape=(len_seq, size_voc_mr+1)))
	model.add(AttentionDecoder(nhid, size_voc_ref+1))
	model.load_weights(path_model)

	print("ok!")
	# --------------------
	# ---- PREPROCESS ----
	# --------------------


	# -- Preprocessing MRs --
	# -----------------------
	print("Preprocessing MRs...", end=" ")
	# Extract Name, Food and Near features
	data["mr_name"] = data.mr.map(lambda mr: extract_feature(mr, "name"))
	data["mr_food"] = data.mr.map(lambda mr: extract_feature(mr, "food"))
	data["mr_near"] = data.mr.map(lambda mr: extract_feature(mr, "near"))


	# Delexicalize MRs
	data["mr_delexicalized"] = data.mr\
								.map(lambda mr: delexicalize_tokenize_mr(mr))

	print("ok!")

	# ------------------------
	# ---- CREATE DATASET ----
	# ------------------------

	# -- Create X (features)--
	# ------------------------
	print("Creating features...", end=" ")
	data["mr_encoded"] = data.mr_delexicalized\
							.map(lambda mr: encode(mr, w2i_mr))
	data["mr_padded"] = list(pad_sequences(data.mr_encoded, 
													maxlen=len_seq))

	X = []
	for i in range(len(data)):
		one_hot_encoded = one_hot_encode(data.mr_padded[i], size_voc_mr+1)
		X.append(one_hot_encoded)

	X = np.array(X)
	print("ok!")
	# -----------------
	# ---- PREDICT ----
	# -----------------
	
	print("Predicting...", end=" ")
	predictions = []
	for i in range(len(X)):
		prediction = decode(one_hot_decode(model.predict(X[i:i+1])[0]), i2w_ref)
		predictions.append(prediction)

	data["pred"] = predictions
	print("ok!")
	# ----------------------
	# ---- POST-PROCESS ----
	# ----------------------
	print("Postprocessing and saving...", end=" ")
	data["pred"] = data.apply(lambda row: relexicalize_ref(row, "mr_name", "name_tag"), axis=1)
	data["pred"] = data.apply(lambda row: relexicalize_ref(row, "mr_food", "food_tag"), axis=1)
	data["pred"] = data.apply(lambda row: relexicalize_ref(row, "mr_near", "near_tag"), axis=1)

	data["pred"] = data.pred.map(lambda pred: pred.replace("<begin>", ""))
	data["pred"] = data.pred.map(lambda pred: pred.replace("<end>", ""))
	
	np.savetxt(path_result, list(data.pred), fmt='%s', encoding="utf-8")
	print("ok!")

if __name__ == "__main__":
	main()
