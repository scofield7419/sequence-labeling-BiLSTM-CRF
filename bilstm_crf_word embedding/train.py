import time
import helper
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from BILSTM_CRF import BILSTM_CRF

# python train.py train.in model -v validation.in -c char_emb -e 10 -g 2

# python train.py train.in model -v validation.in -e 10


parser = argparse.ArgumentParser()
parser.add_argument("train_path", help="the path of the train file")
parser.add_argument("save_path", help="the path of the saved model")
parser.add_argument("-v","--val_path", help="the path of the validation file", default=None)
parser.add_argument("-e","--epoch", help="the number of epoch", default=100, type=int)
parser.add_argument("-c","--char_emb", help="the char embedding file", default=None)
parser.add_argument("-g","--gpu", help="the id of gpu, the default is 0", default=0, type=int)

args = parser.parse_args()

train_path = args.train_path
save_path = args.save_path
val_path = args.val_path
num_epochs = args.epoch
emb_path = args.char_emb
# gpu_config = "/gpu:"+str(args.gpu)
gpu_config = "/cpu:0"
num_steps = 200 # it must consist with the test

start_time = time.time()
print "preparing train and validation data"
X_train, y_train, X_val, y_val = helper.getTrain(train_path=train_path, val_path=val_path, seq_max_len=num_steps)
char2id, id2char = helper.loadMap("char2id")
label2id, id2label = helper.loadMap("label2id")
num_chars = len(id2char.keys())
num_classes = len(id2label.keys())
if emb_path != None:
	embedding_matrix = helper.getEmbedding(emb_path)
else:
	embedding_matrix = None

print "building model"
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
	with tf.device(gpu_config):
		initializer = tf.random_uniform_initializer(-0.1, 0.1)
		with tf.variable_scope("model", reuse=None, initializer=initializer):
			model = BILSTM_CRF(num_chars=num_chars, num_classes=num_classes, num_steps=num_steps, num_epochs=num_epochs, embedding_matrix=embedding_matrix, is_training=True)

		print "training model"
		tf.initialize_all_variables().run()
		model.train(sess, save_path, X_train, y_train, X_val, y_val)

		print "final best f1 is: %f" % (model.max_f1)

		end_time = time.time()
		print "time used %f(hour)" % ((end_time - start_time) / 3600)
