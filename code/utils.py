import numpy as np
import re

def extract_feature(mr, feature):
    """
    Args:
        mr (string): as provided in the dataset 
                    (ex: "name[The Vaults], food[Chinese]")
        feature (string): feature that we want to retrieve from mr
                    (ex: "name")
    
    Returns:
        string: extracted feature
                (ex: "The Vaults")
    """
    # Finder is supposed to find either 0 or 1 match in mr
    finder = re.findall(feature+"\[[\w|\s|£]+\]", mr)
    feature_found = finder[0] if len(finder)>0 else ""
    # Remove brackets
    result = re.sub(feature+"\[|\]", "", feature_found)
    return result

def tokenize_mr(mr):
    """
    Args:
        mr (string): as provided in the dataset 
                    (ex: "name[The Vaults], food[Chinese]")
    
    Returns:
        list(string): tokenized mr
                    (ex: ["name", "The Vaults", "food", "Chinese"])
    """

    # Get all features (name, food, etc.)
    features = re.findall(r"[\w|\s]+\[", mr)
    features_processed = [re.sub("\[|\s", "",
                             feature) for feature in features]
    
    # Get all values (The Vaults, Chinese, etc.)
    values = re.findall(r"\[[\w|\s|£|-]+\]", mr)
    values_processed = [re.sub("\[|\]", "", value) for value in values]

    # Returns mr_tokenized
    mr_tokens = []
    for i in range(len(valuesp)):
        mr_tokens.append(features_processed[i])
        mr_tokens.append(values_processed[i])
    return mr_tokens


def delexicalize_tokenize_mr(mr):
    
    features = ["eatType", "food", "priceRange", "customer rating", "area", "familyFriendly", "near"]
    
    delex_mr = []
    for feature in features:
        delex_mr.append(feature)
        value = extract_feature(mr, feature)
        if value != "":
            if feature == "food":
                value = "FOOD_TAG"
            if feature == "near":
                value = "NEAR_TAG"
        delex_mr.append(value)

    return delex_mr


def get_voc(list_text, min_count=-1):
    """
    Args:
        list_text (list(string)): list of tokenized MRs,
                                         or tokenized refs
        min_count (float, optional): if a word appears less 
            than min_count, it will not be added to the voc

    Returns:
        (dict, dict, set): 
                voc: set of words in the voc
                word2idx: assigns a given word of the voc to an index
                idx2word: word2idx reverse dict 
    """
    # Count the number of occurences of each word in list_text
    word_count = {}
    for text in list_text:
        for word in text:
            word_count[word] = word_count.get(word,0)+1
    
    # Initialize word2idx and voc
    word2idx = {}
    voc = set()

    # Initialize indexes at 1 ; 0 will be assigned to unknown (or
    # unfrequent words)
    idx = 1

    # Add to the voc only if greater than min_count
    for word in word_count:
        if word_count[word] >= min_count:
            voc.add(word)
            word2idx[word] = idx
            idx+=1
                
    idx2word = {v: k for k, v in word2idx.items()}

    return voc, word2idx, idx2word

def delexicalize_ref(row, feature, tag_to_put):
    # Initialize result
    result = row["ref_delexicalized"]
    # Replace only if there is we have the value for the feature
    if row[feature] != '':
        result = row["ref_delexicalized"].replace(row[feature], tag_to_put)

    return result

def relexicalize_ref(row, feature, tag):
    result = row["pred"].replace(tag, row[feature])
    return result


def encode(sent, word2idx):
    return [word2idx[word] for word in sent if word in word2idx]

def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return np.array(encoding)

def decode(seq_idx, idx2word):
    return " ".join([idx2word[idx] for idx in seq_idx if idx in idx2word])

def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]