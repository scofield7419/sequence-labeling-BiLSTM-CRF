# -*- coding: utf-8 -*-
# @Time : 2019/6/2 上午10:55
# @Author : Scofield Phil
# @FileName: utils.py
# @Project: sequence-lableing-vex

import re, logging, datetime, csv
import pandas as pd


def get_logger(log_dir):
    log_file = log_dir + "/" + (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.log'))
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(message)s')

    # log into file
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # log into terminal
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(datetime.datetime.now().strftime('%Y-%m-%d: %H %M %S'))

    return logger


def extractEntity_(sentence, labels_, reg_str, label_level):
    entitys = []
    labled_labels = []
    labled_indexs = []
    labels__ = [('%03d' % (ind)) + lb for lb, ind in zip(labels_, range(len(labels_)))]
    labels = " ".join(labels__)

    re_entity = re.compile(reg_str)

    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        if label_level == 1:
            labled_labels.append("_")
        elif label_level == 2:
            labled_labels.append(entity_labels.split()[0][5:])

        start_index = int(entity_labels.split()[0][:3])
        if len(entity_labels.split()) != 1:
            end_index = int(entity_labels.split()[-1][:3]) + 1
        else:
            end_index = start_index + 1
        entity = ' '.join(sentence[start_index:end_index])
        labels = labels__[end_index:]
        labels = " ".join(labels)
        entitys.append(entity)
        labled_indexs.append((start_index, end_index))
        m = re_entity.search(labels)

    return entitys, labled_labels, labled_indexs


def extractEntity(x, y, dataManager):
    label_scheme = dataManager.label_scheme
    label_level = dataManager.label_level
    label_hyphen = dataManager.hyphen

    if label_scheme == "BIO":
        if label_level == 1:
            reg_str = r'([0-9][0-9][0-9]B' + r' )([0-9][0-9][0-9]I' + r' )*'

        elif label_level == 2:
            tag_bodys = ["(" + tag + ")" for tag in dataManager.suffix]
            tag_str = "(" + ('|'.join(tag_bodys)) + ")"
            reg_str = r'([0-9][0-9][0-9]B'+label_hyphen + tag_str + r' )([0-9][0-9][0-9]I'+label_hyphen + tag_str + r' )*'

    elif label_scheme == "BIESO":
        if label_level == 1:
            reg_str = r'([0-9][0-9][0-9]B' + r' )([0-9][0-9][0-9]I' + r' )*([0-9][0-9][0-9]E' + r' )|([0-9][0-9][0-9]S' + r' )'

        elif label_level == 2:
            tag_bodys = ["(" + tag + ")" for tag in dataManager.suffix]
            tag_str = "(" + ('|'.join(tag_bodys)) + ")"
            reg_str = r'([0-9][0-9][0-9]B'+label_hyphen + tag_str + r' )([0-9][0-9][0-9]I'+label_hyphen + tag_str + r' )*([0-9][0-9][0-9]E'+label_hyphen + tag_str + r' )|([0-9][0-9][0-9]S'+label_hyphen + tag_str + r' )'

    return extractEntity_(x, y, reg_str, label_level)


def metrics(X, y_true, y_pred, measuring_metrics, dataManager):
    precision = -1.0
    recall = -1.0
    f1 = -1.0

    hit_num = 0
    pred_num = 0
    true_num = 0

    correct_label_num = 0
    total_label_num = 0
    for i in range(len(y_true)):
        x = [str(dataManager.id2token[val]) for val in X[i] if val != dataManager.token2id[dataManager.PADDING]]
        y = [str(dataManager.id2label[val]) for val in y_true[i] if val != dataManager.label2id[dataManager.PADDING]]
        y_hat = [str(dataManager.id2label[val]) for val in y_pred[i] if
                 val != dataManager.label2id[dataManager.PADDING]]  # if val != 5

        correct_label_num += len([1 for a, b in zip(y, y_hat) if a == b])
        total_label_num += len(y)

        true_labels, labled_labels, _ = extractEntity(x, y, dataManager)
        pred_labels, labled_labels, _ = extractEntity(x, y_hat, dataManager)

        hit_num += len(set(true_labels) & set(pred_labels))
        pred_num += len(set(pred_labels))
        true_num += len(set(true_labels))

    if total_label_num != 0:
        accuracy = 1.0 * correct_label_num / total_label_num

    if pred_num != 0:
        precision = 1.0 * hit_num / pred_num
    if true_num != 0:
        recall = 1.0 * hit_num / true_num
    if precision > 0 and recall > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    results = {}
    for measu in measuring_metrics:
        results[measu] = vars()[measu]
    return results


def read_csv_(file_name, names, delimiter='t'):
    if delimiter=='t':
        sep = "\t"
    elif delimiter=='b':
        sep = " "
    else:
        sep = delimiter

    return pd.read_csv(file_name, sep=sep, quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None,
                       names=names)


def save_csv_(df_, file_name, names, delimiter='t'):
    if delimiter == 't':
        sep = "\t"
    elif delimiter == 'b':
        sep = " "
    else:
        sep = delimiter

    df_.to_csv(file_name, quoting=csv.QUOTE_NONE,
               columns=names, sep=sep, header=False,
               index=False)
