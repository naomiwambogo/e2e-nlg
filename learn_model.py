import re
import numpy as np
import pandas as pd
import string
import os
import random
import pickle
import argparse

from nltk.corpus import stopwords

from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, TimeDistributed, \
                                RepeatVector, Bidirectional
from attention_decoder import AttentionDecoder

from utils import extract_feature, \
                    delexicalize_tokenize_mr, \
                    delexicalize_ref, \
                    get_voc, \
                    encode, decode, \
                    one_hot_encode, \
                    one_hot_decode



def main():

    path_train = "../e2e-dataset/trainset.csv"
    path_model = "../models/model.h5"

    # -------------------
    # ---- LOAD DATA ----
    # -------------------
    print("Loading data...", end=" ")
    data = pd.read_csv(path_train)

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
    # -- Preprocessing REFs --
    # ------------------------
    print("Preprocessing REFs...", end=" ")
    # Remove punctuation
    data["ref_punct"] = data.ref.map(lambda ref: re.sub("[{}]".format(string.punctuation), "", ref))
    
    # Delimit refs with begin and end tokens
    data["ref_delim"] = data.ref_punct\
                            .map(lambda ref: "<BEGIN> " + ref + " <END>")
    
    # Delexicalize REFs
    data["ref_delexicalized"] = data.ref_delim
    data["ref_delexicalized"] = data\
            .apply(lambda row: delexicalize_ref(row, "mr_name", "NAME_TAG"),
                     axis=1)
    data["ref_delexicalized"] = data\
            .apply(lambda row: delexicalize_ref(row, "mr_near", "NEAR_TAG"), 
                     axis=1)
    data["ref_delexicalized"] = data\
            .apply(lambda row: delexicalize_ref(row, "mr_food", "FOOD_TAG"), 
                     axis=1)
    
    # Tokenize REFs
    data["ref_tokenized"] = data.ref_delexicalized.map(lambda ref: ref.lower().split(" "))
    
    # Downsample stopwords
    sw = stopwords.words('english')
    data["ref_w_sw"] = data.ref_tokenized\
                .map(lambda ref: [word for word in ref if (word not in sw \
                                                or random.random() > 0.5)])

    print("ok!")
    # ------------------------
    # ---- CREATE DATASET ----
    # ------------------------

    len_seq = 25

    # -- Create X (features)--
    # ------------------------
    print("Creating features...", end=" ")
    voc_mr, word2idx_mr, idx2word_mr = get_voc(data.mr_delexicalized)
    size_voc_mr = len(voc_mr)

    data["mr_encoded"] = data.mr_delexicalized\
                            .map(lambda mr: encode(mr, word2idx_mr))
    data["mr_padded"] = list(pad_sequences(data.mr_encoded, 
                                                    maxlen=len_seq))

    X = []
    for i in range(len(data)):
        one_hot_encoded = one_hot_encode(data.mr_padded[i], size_voc_mr+1)
        X.append(one_hot_encoded)

    X = np.array(X)

    print("ok!")
    # -- Create y (target) --
    # -----------------------
    print("Creating target...", end=" ")
    voc_ref, word2idx_ref, idx2word_ref = get_voc(data.ref_w_sw,
                                                            min_count=100)
    size_voc_ref = len(voc_ref)
    
    data["ref_encoded"] = data.ref_tokenized\
                            .map(lambda ref: encode(ref, word2idx_ref))
    data["ref_padded"] = list(pad_sequences(data.ref_encoded, 
                                                    maxlen=len_seq))
    
    y = []
    for i in range(len(data)):
        one_hot_encoded = one_hot_encode(data.ref_padded[i], 
                                                    size_voc_ref+1)
        y.append(one_hot_encoded)
    y = np.array(y)
    print("ok!")
    # ---------------
    # ---- MODEL ----
    # ---------------
    print("Training...", end=" ")
    nhid = 128
    model = Sequential()
    model.add(LSTM(nhid, return_sequences=True, input_shape=(len_seq, size_voc_mr+1)))
    model.add(AttentionDecoder(nhid, size_voc_ref+1))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    model.fit(X, y, epochs=20)

    # --------------
    # ---- SAVE ----
    # --------------

    model.save_weights(path_model)
    with open('models/word2idx_mr.pkl', 'wb') as handle:
        pickle.dump(word2idx_mr, handle, 
                            protocol=pickle.HIGHEST_PROTOCOL)
    with open('models/idx2word_ref.pkl', 'wb') as handle:
        pickle.dump(idx2word_ref, handle, 
                            protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dataset', help='path containing test data', default="e2e-dataset/testset.csv")
    parser.add_argument('--output-model-file', help="pathname to model", default="models/model.h5")

    args = parser.parse_args()

    main(args.test_dataset, args.model, args.ouput_test_file)
    