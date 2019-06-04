# HandBook for BiLSTM+CRF repo

---

## 1. wiki for learning system

1.1 machine learning systems work following two step orders:  
    
    A. train the model based on labeled dataset.
    B. make inference based on the above trained model.

Thus, whatever we do with the model, we first should train the model.
Afterwards, we can make prediction on `testset`, or any text you want to predict.

1.2 why BiLSTM+CRF?

For this question, please refer to the following materials.

- [_Neural Architectures for Named Entity Recognition_](https://www.aclweb.org/anthology/N16-1030)
- [如何理解LSTM后接CRF？](https://www.zhihu.com/question/62399257/answer/241969722)


## 2. `system.config`

`Don't` change the item key name at will.

Use # to comment out the configure item.


#### A.  Status ################

```
mode=api_service
# string: train/test/interactive_predict/api_service

```

#### B.  Datasets(Input/Output) ################

```
datasets_fold=data/example_datasets3
train_file=train.csv
dev_file=dev.csv
test_file=test.csv
```

```
delimiter=t
# string: (t: "\t";"table")|(b: "backspace";" ")|(other, e.g., '|||', ...)
```

```
use_pretrained_embedding=False
token_emb_dir=data/example_datasets3/word.emb
```
Be aware of the following three path.
```
vocabs_dir=data/example_datasets3/vocabs

log_dir=data/example_datasets3/logs

checkpoints_dir=checkpoints/BILSTM-CRFs-datasets3
```

#### C.  Labeling Scheme ################
Be very careful of the following settings.

```
label_scheme=BIO
# string: BIO/BIESO
```

The system support at max 2 level of the label scheme.
You need to make some modification on the source to adapt to more complicated labeling schemes.
```
label_level=2
# int, 1:BIO/BIESO; 2:BIO/BIESO + suffix
# max to 2
```

```
hyphen=_
# string: -|_, for connecting the prefix and suffix: `B_PER', `I_LOC'
```

The suffix for the second level labels.
```
suffix=[NR,NS,NT]
# unnecessary if label_level=1
```

labeling_level:
- for English: （word: hello），（char: h）
- for Chinese: （word: 你好），（char: 你）

```
labeling_level=word
# string: word/char
```

To measure the performance of the model, you have to specify the metrics.
Following are the most used indicators.

Note that the `f1` is compulsory.
You can define any other metrics, in the codes.
```
measuring_metrics=[precision,recall,f1,accuracy]
# string: accuracy|precision|recall|f1
```


#### D.  Model Configuration ################
```
use_crf=True
```
```
cell_type=LSTM
# LSTM, GRU
biderectional=True
encoder_layers=1
```

`embedding_dim` must be consistent with the one in `token_emb_dir` file.

```
embedding_dim=100
```
```
hidden_dim=100
```

*cautions*! set a LARGE number as possible as u can.

The `max_sequence_length` will be fix after training,
 and during inferring, those texts having length larger than this will be truncated.


```
max_sequence_length=300
```

We implement the self attention in `multi-step` RNN.

```
use_self_attention=False
attention_dim=500
```

for reproduction.

```
seed=42
```

#### E. Training Settings ###
```
epoch=300
batch_size=100

dropout=0.5
learning_rate=0.005

optimizer=Adam
#string: GD/Adagrad/AdaDelta/RMSprop/Adam
```
```
checkpoints_max_to_keep=3
print_per_batch=20
```

`early_stop`: if the model did'nt progress within `patient` times of iterations of training,
the training processing will be terminated.

```
is_early_stop=True
patient=5
# unnecessary if is_early_stop=False

checkpoint_name=model-CRFs
```

#### F.  Testing Settings ###
```
output_test_file=test.out
```


```
is_output_sentence_entity=True
output_sentence_entity_file=test.entity.out
# unnecessary if is_output_sentence_entity=False
```

#### G. Api service Settings ###

unnecessary to change the default setting if you operate at the local host.

if you make display the web around the Intranet, you may change the 
`ip`=`0.0.0.0`. 

```
ip=127.0.0.1
port=8000
```

## 3. more tips:

- in `tools` fold:
    - `statis.py` can calculate the statistics for your dataset.
    - `calcu_measure_testout.py` can compute the metrics based on `test.out` and `test.csv`
- ...



