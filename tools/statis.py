# -*- coding: utf-8 -*-
import csv
import numpy as np
import pandas as pd


def statis(data_dir='', dataset_files=[], delimeter='\t'):
    '''
    make statistics for train, dev, test sets.
    :return:
    '''
    print(data_dir)
    names = ["token", "label"]

    # df_all = pd.DataFrame(columns=names)
    df_all_token = []
    df_all_label = []
    for dataset in dataset_files:
        df_set = pd.read_csv(data_dir + "/" + dataset, sep=delimeter, quoting=csv.QUOTE_NONE, skip_blank_lines=False,
                             header=None)
        columns = list(df_set.columns)
        tmp_x = []
        ind = 0
        max_lengs = []

        if len(columns) == 2:
            df_set.columns = names
            tmp_y = []

            ### max sequence length
            df_set["token"] = df_set.token.map(lambda x: -1111 if str(x) == str(np.nan)else x)
            df_set["label"] = df_set.label.map(lambda x: -1111 if str(x) == str(np.nan)else x)
            for record in zip(df_set["token"], df_set["label"]):
                c = record[0]
                l = record[1]
                if c == -1111:
                    ind += 1
                    # if len(tmp_x) > max_leng:
                    #     max_leng = len(tmp_x)
                    max_lengs.append(len(tmp_x))
                    tmp_x = []
                    tmp_y = []
                else:
                    tmp_x.append(c)
                    tmp_y.append(l)
            # df_all.append(df_set, ignore_index=True)
            df_set["label"] = df_set.label.map(lambda x: "" if x == -1111 else x)
            df_all_label.extend(list(df_set["label"]))

        else:
            df_set["token"] = df_set.token.map(lambda x: -1111 if str(x) == str(np.nan)else x)

            for c in df_set["token"]:
                if c == -1111:
                    ind += 1
                    # if len(tmp_x) > max_leng:
                    #     max_leng = len(tmp_x)
                    max_lengs.append(len(tmp_x))
                    tmp_x = []
                else:
                    tmp_x.append(c)

        df_all_token.extend(list(df_set["token"]))

        print("total sentence number in %s : %d" % (dataset, ind))
        print("max sequence length in %s : %d" % (dataset, max(max_lengs)))
        print("avg sequence length in %s : %.3f" % (dataset, np.array(max_lengs).mean()))
        print("median sequence length in %s : %.3f" % (dataset, np.median(np.array(max_lengs))) )
        print("95 percentile sequence length in %s : %.3f" % (dataset, np.percentile(np.array(max_lengs), 95)) )

        print("length of token line in %s is %d" % (dataset, len(df_set)))
        print()

    tokens_dict = list(set(df_all_token))
    print("length of token vocab is %d" % (len(tokens_dict)))

    labels_dict = list(set(df_all_label))
    labels_dict.remove("")
    print("length of labels set is %d" % (len(labels_dict)))
    print("labels set: [%s]" % (",".join(labels_dict)))


statis(data_dir='../data/example_datasets4',
       dataset_files=['train.csv', 'dev.csv', 'test.csv'], delimeter='\t')
