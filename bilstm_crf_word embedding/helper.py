# encoding:utf-8
import re
import os
import csv
import time
import pickle
import numpy as np
import pandas as pd

def getEmbedding(infile_path="embedding"):
    char2id, id_char = loadMap("char2id")
    row_index = 0
    with open(infile_path, "rb") as infile:
        for row in infile:
            row = row.strip()
            row_index += 1
            if row_index == 1:
                num_chars = int(row.split()[0])
                emb_dim = int(row.split()[1])
                emb_matrix = np.zeros((len(char2id.keys()), emb_dim))
                continue
            items = row.split()
            char = items[0]
            emb_vec = [float(val) for val in items[1:]]
            if char in char2id:
                emb_matrix[char2id[char]] = emb_vec
    return emb_matrix


def nextBatch(X, y, start_index, batch_size=128):
    last_index = start_index + batch_size
    X_batch = list(X[start_index:min(last_index, len(X))])
    y_batch = list(y[start_index:min(last_index, len(X))])
    if last_index > len(X):
        left_size = last_index - (len(X))
        for i in range(left_size):
            index = np.random.randint(len(X))
            X_batch.append(X[index])
            y_batch.append(y[index])
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch


def nextRandomBatch(X, y, batch_size=128):
    X_batch = []
    y_batch = []
    for i in range(batch_size):
        index = np.random.randint(len(X))
        X_batch.append(X[index])
        y_batch.append(y[index])
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch


# use "0" to padding the sentence
def padding(sample, seq_max_len):
    for i in range(len(sample)):
        if len(sample[i]) < seq_max_len:
            sample[i] += [0 for _ in range(seq_max_len - len(sample[i]))]
    return sample


def prepare(chars, labels, seq_max_len, is_padding=True):
    X = []
    y = []
    tmp_x = []
    tmp_y = []

    for record in zip(chars, labels):
        c = record[0]
        l = record[1]
        # empty line
        if c == -1:
            if len(tmp_x) <= seq_max_len:
                X.append(tmp_x)
                y.append(tmp_y)
            tmp_x = []
            tmp_y = []
        else:
            tmp_x.append(c)
            tmp_y.append(l)
    if is_padding:
        X = np.array(padding(X, seq_max_len))
    else:
        X = np.array(X)
    y = np.array(padding(y, seq_max_len))

    return X, y


def extractEntity(sentence, labels):
    entitys = []
    re_entity = re.compile(r'BM*E')
    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        start_index = labels.find(entity_labels)
        entity = sentence[start_index:start_index + len(entity_labels)]
        labels = list(labels)
        # replace the "BM*E" with "OO*O"
        labels[start_index: start_index + len(entity_labels)] = ['O' for i in range(len(entity_labels))]
        entitys.append(entity)
        labels = ''.join(labels)
        m = re_entity.search(labels)
    return entitys


def loadMap(token2id_filepath):
    if not os.path.isfile(token2id_filepath):
        print "file not exist, building map"
        buildMap()

    token2id = {}
    id2token = {}
    with open(token2id_filepath) as infile:
        for row in infile:
            row = row.rstrip().decode("utf-8")
            token = row.split('\t')[0]
            token_id = int(row.split('\t')[1])
            token2id[token] = token_id
            id2token[token_id] = token
    return token2id, id2token


def saveMap(id2char, id2label):
    with open("char2id", "wb") as outfile:
        for idx in id2char:
            outfile.write(id2char[idx] + "\t" + str(idx) + "\r\n")
    with open("label2id", "wb") as outfile:
        for idx in id2label:
            outfile.write(id2label[idx] + "\t" + str(idx) + "\r\n")
    print "saved map between token and id"


def buildMap(train_path="train.in"):
    df_train = pd.read_csv(train_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=["char", "label"])
    chars = list(set(df_train["char"][df_train["char"].notnull()]))
    labels = list(set(df_train["label"][df_train["label"].notnull()]))
    char2id = dict(zip(chars, range(1, len(chars) + 1)))
    label2id = dict(zip(labels, range(1, len(labels) + 1)))
    id2char = dict(zip(range(1, len(chars) + 1), chars))
    id2label = dict(zip(range(1, len(labels) + 1), labels))
    id2char[0] = "<PAD>"
    id2label[0] = "<PAD>"
    char2id["<PAD>"] = 0
    label2id["<PAD>"] = 0
    id2char[len(chars) + 1] = "<NEW>"
    char2id["<NEW>"] = len(chars) + 1

    saveMap(id2char, id2label)

    return char2id, id2char, label2id, id2label


def getTrain(train_path, val_path, train_val_ratio=0.99, use_custom_val=False, seq_max_len=200):
    char2id, id2char, label2id, id2label = buildMap(train_path)
    df_train = pd.read_csv(train_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=["char", "label"])

    # map the char and label into id
    df_train["char_id"] = df_train.char.map(lambda x: -1 if str(x) == str(np.nan) else char2id[x])
    df_train["label_id"] = df_train.label.map(lambda x: -1 if str(x) == str(np.nan) else label2id[x])

    # convert the data in maxtrix
    X, y = prepare(df_train["char_id"], df_train["label_id"], seq_max_len)

    # shuffle the samples
    num_samples = len(X)
    indexs = np.arange(num_samples)
    np.random.shuffle(indexs)
    X = X[indexs]
    y = y[indexs]

    if val_path != None:
        X_train = X
        y_train = y
        X_val, y_val = getTest(val_path, is_validation=True, seq_max_len=seq_max_len)
    else:
        # split the data into train and validation set
        X_train = X[:int(num_samples * train_val_ratio)]
        y_train = y[:int(num_samples * train_val_ratio)]
        X_val = X[int(num_samples * train_val_ratio):]
        y_val = y[int(num_samples * train_val_ratio):]

    print "train size: %d, validation size: %d" % (len(X_train), len(y_val))

    return X_train, y_train, X_val, y_val


def getTest(test_path="test.in", is_validation=False, seq_max_len=200):
    char2id, id2char = loadMap("char2id")
    label2id, id2label = loadMap("label2id")

    df_test = pd.read_csv(test_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=["char", "label"])

    def mapFunc(x, char2id):
        if str(x) == str(np.nan):
            return -1
        elif x.decode("utf-8") not in char2id:
            return char2id["<NEW>"]
        else:
            return char2id[x.decode("utf-8")]

    df_test["char_id"] = df_test.char.map(lambda x: mapFunc(x, char2id))
    df_test["label_id"] = df_test.label.map(lambda x: -1 if str(x) == str(np.nan) else label2id[x])

    if is_validation:
        X_test, y_test = prepare(df_test["char_id"], df_test["label_id"], seq_max_len)
        return X_test, y_test
    else:
        df_test["char"] = df_test.char.map(lambda x: -1 if str(x) == str(np.nan) else x)
        X_test, _ = prepare(df_test["char_id"], df_test["char_id"], seq_max_len)
        X_test_str, _ = prepare(df_test["char"], df_test["char_id"], seq_max_len, is_padding=False)
        print "test size: %d" % (len(X_test))
        return X_test, X_test_str


def getTransition(y_train_batch):
    transition_batch = []
    for m in range(len(y_train_batch)):
        y = [5] + list(y_train_batch[m]) + [0]
        for t in range(len(y)):
            if t + 1 == len(y):
                continue
            i = y[t]
            j = y[t + 1]
            if i == 0:
                break
            transition_batch.append(i * 6 + j)
    transition_batch = np.array(transition_batch)
    return transition_batch
