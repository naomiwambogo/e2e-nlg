import pandas as pd
import numpy as np
import argparse
from os.path import join


def generate_baseline_sentence(data, i):
    # For each feature we generate a piece of the sentence to return
    sent_familyFriendly = ['non family friendly ' if data[' familyFriendly'][i]=='no' 
                           else ('family friendly ' if data[' familyFriendly'][i]=='yes' else '')]

    sent_food = [data[' food'][i] + ' ' if data[' food'][i] is not None else ''] 

    sent_eattype = [data[' eatType'][i] + ' ' if data[' eatType'][i] is not None else 'restaurant ']

    sent_near = ['near ' + data[' near'][i] + ' ' if data[' near'][i] is not None else '']

    sent_area = ['in the ' + data[' area'][i] + '.' if data[' area'][i] is not None else '']

    sent_customerrating = [' It has a ' + data[' customer rating'][i] + ' customer rating.' if data[' customer rating'][i] is not None else '']

    sent_pricerange = [' It has a ' + data[' priceRange'][i] + ' price range.' if data[' priceRange'][i] is not None else '']
    
    # We return the concatenation of the previous pieces of sentence
    sent = data['name'][i] + ' is a ' + sent_familyFriendly[0] + sent_food[0] + sent_eattype[0]  + sent_near[0] + sent_area[0] + sent_customerrating[0] + sent_pricerange[0]
    
    return sent

def create_feature(features_values, features, feat):
    feature = []
    for i in range(len(features_values)):
        if feat in features[i]:
            index = features[i].index(feat)
            value = features_values[i][index][1]
            feature.append(value)
        else:
            feature.append(None)

    return feature

def main(path_test, path_result):

    # -------------------
    # ---- LOAD DATA ----
    # -------------------

    print("Loading data...", end=" ")
    data = pd.read_csv(path_test, names=["mr"], skiprows=1)
    print("ok!")
    # --------------------
    # ---- PREPROCESS ----
    # --------------------
    print("Preprocessing...", end=" ")
    # Split the different fields of a sentence
    split = [data['mr'][i].split(',') for i in range(len(data['mr']))]
    
    # Get the features and its value for each split
    features_values = []
    for s in split:
        features_values.append([s[i][:-1].split('[') for i in range(len(s))])
    
    # Get the list of all the features available    
    features = []
    for fv in features_values:
        features.append([fv[i][0] for i in range(len(fv))])
        
    unique_features =  set([item for sublist in features for item in sublist])
    print("ok!")
    # -----------------
    # ---- PREDICT ----
    # -----------------
    print("Predicting and saving...", end=" ")
    # Create a column for each feature
    for feat in unique_features:
        feat_list = create_feature(features_values, features, feat)
        data[feat] = feat_list

    predictions = []
    for i in range(len(data)):
        predictions.append(generate_baseline_sentence(data, i))

    np.savetxt(path_result, predictions, fmt='%s')
    print("ok!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dataset', help='path containing test data', default="e2e-dataset/testset.csv")
    parser.add_argument('--ouput-test-file', help='pathname to results testfile', default="results/results_baseline.txt")

    args = parser.parse_args()

    main(args.test_dataset, args.ouput_test_file)


    