# -*- coding: utf-8 -*-
# @Time : 2019/6/2 上午10:55
# @Author : Scofield Phil
# @FileName: BiLSTM_CRFs.py
# @Project: sequence-lableing-vex

import math, os
from engines.utils import metrics, save_csv_, extractEntity
import numpy as np
import tensorflow as tf
import pandas as pd
import time
from pathlib import PurePosixPath
import tensorflow_addons as tfa
from pathlib import PurePosixPath

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# disable v2 behavior here
tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.enable_eager_execution()
tf.compat.v1.enable_resource_variables()
# tf.compat.v1.enable_v2_tensorshape()



class BiLSTM_CRFs(object):
    def __init__(self, configs, logger, dataManager):
        


        os.environ['CUDA_VISIBLE_DEVICES'] = configs.CUDA_VISIBLE_DEVICES

        self.configs = configs
        self.logger = logger
        self.logdir = configs.log_dir
        self.measuring_metrics = configs.measuring_metrics
        self.dataManager = dataManager

        if configs.mode == "train":
            self.is_training = True
        else:
            self.is_training = False

        self.checkpoint_name = configs.checkpoint_name
        self.checkpoints_dir = configs.checkpoints_dir
        self.output_test_file = PurePosixPath(configs.datasets_fold/configs.output_test_file)
        self.is_output_sentence_entity = configs.is_output_sentence_entity
        self.output_sentence_entity_file = PurePosixPath(configs.datasets_fold/configs.output_sentence_entity_file)

        self.biderectional = configs.biderectional
        self.cell_type = configs.cell_type
        self.num_layers = configs.encoder_layers

        self.is_crf = configs.use_crf

        self.learning_rate = configs.learning_rate
        self.dropout_rate = configs.dropout
        self.batch_size = configs.batch_size

        self.emb_dim = configs.embedding_dim
        self.hidden_dim = configs.hidden_dim

# ====================== model part =========================================

        # check the chosen RNN model
        if configs.cell_type == 'LSTM':
            if self.biderectional:
                self.cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.hidden_dim)
            else:
                self.cell = tf.compat.v1.nn.rnn_cell.LSTMCell(2 * self.hidden_dim)
        else:
            # choose GRU model
            if self.biderectional:
                self.cell = tf.compat.v1.nn.rnn_cell.GRUCell(self.hidden_dim)
            else:
                self.cell = tf.compat.v1.nn.rnn_cell.GRUCell(2 * self.hidden_dim)

# ====================== model part ========================================

        self.is_attention = configs.use_self_attention
        self.attention_dim = configs.attention_dim

        self.num_epochs = configs.epoch
        self.max_time_steps = configs.max_sequence_length

        self.num_tokens = dataManager.max_token_number
        self.num_classes = dataManager.max_label_number

        self.is_early_stop = configs.is_early_stop
        self.patient = configs.patient

        self.max_to_keep = configs.checkpoints_max_to_keep
        self.print_per_batch = configs.print_per_batch
        self.best_f1_val = 0

        # define the set of optimizers
        if configs.optimizer == 'Adagrad':
            # self.optimizer = tf.keras.optimizers.Adagrad(self.learning_rate)
            self.optimizer = tf.compat.v1.train.AdagradOptimizer(self.learning_rate)
        elif configs.optimizer == 'Adadelta':
            self.optimizer = tf.compat.v1.train.AdadeltaOptimizer(self.learning_rate)
            # self.optimizer = tf.keras.optimizers.Adadelta(self.learning_rate)
        elif configs.optimizer == 'RMSprop':
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate)
            # self.optimizer = tf.keras.optimizers.RMSProp(self.learning_rate)
        elif configs.optimizer == 'GD':
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.learning_rate)
            # self.optimizer = tf.keras.optimizers.SGD(self.learning_rate)
        else:
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
            # self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

# ================== Variable Definitions ====================================

        # adapting its scale to the shape of weights tensors
        self.initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        self.global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int32)

        # whether using the pretrained embedding
        if configs.use_pretrained_embedding:
            embedding_matrix = dataManager.getEmbedding(configs.token_emb_dir)
            self.embedding = tf.Variable(embedding_matrix, trainable=False, use_resource=False, name="emb", dtype=tf.float32)
        else:
            self.embedding = tf.compat.v1.get_variable("emb", [self.num_tokens, self.emb_dim], trainable=True,
                                             initializer=self.initializer)

# ================== Variable Definitions ====================================

# ================= Session run ===============

        # build the model based on the parameters provided above
        self.build()
        self.logger.info("model initialed...\n")

        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
# ================= Session run ===============

    def build(self):

# ========== eager execution problem in tf2 =============
        
        # create the variable inputs and targets
        self.inputs = tf.compat.v1.placeholder(tf.int32, [None, self.max_time_steps])
        self.targets = tf.compat.v1.placeholder(tf.int32, [None, self.max_time_steps])
        # self.inputs = tf.keras.Input(shape=[None, self.max_time_steps],dtype = tf.dtypes.int32)
        # self.targets = tf.keras.Input(shape=[None, self.max_time_steps],dtype = tf.dtypes.int32)


        # look up embeddings for the given ids from a list of tensors
        # self.inputs_emb = tf.nn.embedding_lookup(params=self.embedding, ids=self.inputs)
        self.inputs_emb = tf.nn.embedding_lookup(params=self.embedding, ids=self.inputs)

        self.inputs_emb = tf.transpose(a=self.inputs_emb, perm=[1, 0, 2])
        self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.emb_dim])
        self.inputs_emb = tf.split(self.inputs_emb, self.max_time_steps, 0)

        # ================ lstm cell =================
        if self.biderectional:
            lstm_cell_fw = self.cell
            lstm_cell_bw = self.cell

            # dropout
            if self.is_training:
                lstm_cell_fw = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
                lstm_cell_bw = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))

            lstm_cell_fw = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
            lstm_cell_bw = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)

            # get the length of each sample
            self.length = tf.reduce_sum(input_tensor=tf.sign(self.inputs), axis=1)
            self.length = tf.cast(self.length, tf.int32)

            # forward and backward
            # outputs, _, _ = tf.compat.v1.nn.rnn_cell.static_bidirectional_rnn(
            outputs, _, _ = tf.compat.v1.nn.static_bidirectional_rnn(    
                lstm_cell_fw,
                lstm_cell_bw,
                self.inputs_emb,
                dtype=tf.float32,
                sequence_length=self.length
            )

        else:
            lstm_cell = self.cell
            if self.is_training:
                lstm_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=(1 - self.dropout_rate))
            lstm_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
            self.length = tf.reduce_sum(input_tensor=tf.sign(self.inputs), axis=1)
            self.length = tf.cast(self.length, tf.int32)

            outputs, _ = tf.compat.v1.nn.rnn_cell.static_rnn(
                lstm_cell,
                self.inputs_emb,
                dtype=tf.float32,
                sequence_length=self.length
            )

        # outputs: list_steps[batch, 2*dim]
        outputs = tf.concat(outputs, 1)
        outputs = tf.reshape(outputs, [self.batch_size, self.max_time_steps, self.hidden_dim * 2])

        # self attention module
        if self.is_attention:
            H1 = tf.reshape(outputs, [-1, self.hidden_dim * 2])
            W_a1 = tf.compat.v1.get_variable("W_a1", shape=[self.hidden_dim * 2, self.attention_dim],
                                   initializer=self.initializer, trainable=True)
            u1 = tf.matmul(H1, W_a1)

            H2 = tf.reshape(tf.identity(outputs), [-1, self.hidden_dim * 2])
            W_a2 = tf.compat.v1.get_variable("W_a2", shape=[self.hidden_dim * 2, self.attention_dim],
                                   initializer=self.initializer, trainable=True)
            u2 = tf.matmul(H2, W_a2)

            u1 = tf.reshape(u1, [self.batch_size, self.max_time_steps, self.hidden_dim * 2])
            u2 = tf.reshape(u2, [self.batch_size, self.max_time_steps, self.hidden_dim * 2])
            u = tf.matmul(u1, u2, transpose_b=True)

            # Array of weights for each time step
            A = tf.nn.softmax(u, name="attention")
            outputs = tf.matmul(A, tf.reshape(tf.identity(outputs),
                                              [self.batch_size, self.max_time_steps, self.hidden_dim * 2]))

        # linear
        self.outputs = tf.reshape(outputs, [-1, self.hidden_dim * 2])
        self.softmax_w = tf.compat.v1.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes],
                                         initializer=self.initializer)
        self.softmax_b = tf.compat.v1.get_variable("softmax_b", [self.num_classes], initializer=self.initializer)
        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b

        self.logits = tf.reshape(self.logits, [self.batch_size, self.max_time_steps, self.num_classes])
        
        # print(self.logits.get_shape().as_list())
        if not self.is_crf:
            # softmax
            softmax_out = tf.nn.softmax(self.logits, axis=-1)

            self.batch_pred_sequence = tf.cast(tf.argmax(input=softmax_out, axis=-1), tf.int32)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
            mask = tf.sequence_mask(self.length)

            self.losses = tf.boolean_mask(tensor=losses, mask=mask)

            self.loss = tf.reduce_mean(input_tensor=losses)
        else:
            # crf
            # self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            #     self.logits, self.targets, self.length)
            self.log_likelihood, self.transition_params = tfa.text.crf.crf_log_likelihood(
                self.logits, self.targets, self.length)
            self.batch_pred_sequence, self.batch_viterbi_score = tfa.text.crf.crf_decode(self.logits,
                                                                                           self.transition_params,
                                                                                           self.length)

            self.loss = tf.reduce_mean(input_tensor=-self.log_likelihood)

        self.train_summary = tf.compat.v1.summary.scalar("loss", self.loss)
        self.dev_summary = tf.compat.v1.summary.scalar("loss", self.loss)

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def train(self):
        X_train, y_train, X_val, y_val = self.dataManager.getTrainingSet()
        tf.compat.v1.initialize_all_variables().run(session=self.sess)

        saver = tf.compat.v1.train.Saver(max_to_keep=self.max_to_keep)
        tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(PurePosixPath(self.logdir/"training_loss"),self.sess.graph)
        dev_writer = tf.compat.v1.summary.FileWriter(PurePosixPath(self.logdir/"validating_loss"),self.sess.graph)

        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))
        num_val_iterations = int(math.ceil(1.0 * len(X_val) / self.batch_size))

        cnt = 0
        cnt_dev = 0
        unprogressed = 0
        very_start_time = time.time()
        best_at_epoch = 0
        self.logger.info("\ntraining starting" + ("+" * 20))
        for epoch in range(self.num_epochs):
            start_time = time.time()
            # shuffle train at each epoch
            sh_index = np.arange(len(X_train))
            np.random.shuffle(sh_index)
            X_train = X_train[sh_index]
            y_train = y_train[sh_index]

            self.logger.info("\ncurrent epoch: %d" % (epoch))
            for iteration in range(num_iterations):
                X_train_batch, y_train_batch = self.dataManager.nextBatch(X_train, y_train,
                                                                          start_index=iteration * self.batch_size)
                _, loss_train, train_batch_viterbi_sequence, train_summary = \
                    self.sess.run([
                        self.opt_op,
                        self.loss,
                        self.batch_pred_sequence,
                        self.train_summary
                    ],
                        feed_dict={
                            self.inputs: X_train_batch,
                            self.targets: y_train_batch,
                        })

                if iteration % self.print_per_batch == 0:
                    cnt += 1
                    train_writer.add_summary(train_summary, cnt)

                    measures = metrics(X_train_batch, y_train_batch,
                                       train_batch_viterbi_sequence,
                                       self.measuring_metrics, self.dataManager)

                    res_str = ''
                    for k, v in measures.items():
                        res_str += (k + ": %.3f " % v)
                    self.logger.info("training batch: %5d, loss: %.5f, %s" % (iteration, loss_train, res_str))

            # validation
            loss_vals = list()
            val_results = dict()
            for measu in self.measuring_metrics:
                val_results[measu] = 0

            for iterr in range(num_val_iterations):
                cnt_dev += 1
                X_val_batch, y_val_batch = self.dataManager.nextBatch(X_val, y_val, start_index=iterr * self.batch_size)

                loss_val, val_batch_viterbi_sequence, dev_summary = \
                    self.sess.run([
                        self.loss,
                        self.batch_pred_sequence,
                        self.dev_summary
                    ],
                        feed_dict={
                            self.inputs: X_val_batch,
                            self.targets: y_val_batch,
                        })

                measures = metrics(X_val_batch, y_val_batch, val_batch_viterbi_sequence,
                                   self.measuring_metrics, self.dataManager)
                dev_writer.add_summary(dev_summary, cnt_dev)

                for k, v in measures.items():
                    val_results[k] += v
                loss_vals.append(loss_val)

            time_span = (time.time() - start_time) / 60
            val_res_str = ''
            dev_f1_avg = 0
            for k, v in val_results.items():
                val_results[k] /= num_val_iterations
                val_res_str += (k + ": %.3f " % val_results[k])
                if k == 'f1': dev_f1_avg = val_results[k]

            self.logger.info("time consumption:%.2f(min),  validation loss: %.5f, %s" %
                             (time_span, np.array(loss_vals).mean(), val_res_str))
            if np.array(dev_f1_avg).mean() > self.best_f1_val:
                unprogressed = 0
                self.best_f1_val = np.array(dev_f1_avg).mean()
                best_at_epoch = epoch
                saver.save(self.sess, PurePosixPath(self.checkpoints_dir/self.checkpoint_name).as_posix(), global_step=self.global_step)
                self.logger.info("saved the new best model with f1: %.3f" % (self.best_f1_val))
            else:
                unprogressed += 1

            if self.is_early_stop:
                if unprogressed >= self.patient:
                    self.logger.info("early stopped, no progress obtained within %d epochs" % self.patient)
                    self.logger.info("overall best f1 is %f at %d epoch" % (self.best_f1_val, best_at_epoch))
                    self.logger.info(
                        "total training time consumption: %.3f(min)" % ((time.time() - very_start_time) / 60))
                    self.sess.close()
                    return
        self.logger.info("overall best f1 is %f at %d epoch" % (self.best_f1_val, best_at_epoch))
        self.logger.info("total training time consumption: %.3f(min)" % ((time.time() - very_start_time) / 60))
        self.sess.close()

    def test(self):
        X_test, y_test_psyduo_label, X_test_str = self.dataManager.getTestingSet()

        num_iterations = int(math.ceil(1.0 * len(X_test) / self.batch_size))
        self.logger.info("total number of testing iterations: " + str(num_iterations))

        self.logger.info("loading model parameter\n")
        tf.compat.v1.initialize_all_variables().run(session=self.sess)
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoints_dir))

        tokens = []
        labels = []
        entities = []
        entities_types = []
        self.logger.info("\ntesting starting" + ("+" * 20))
        for i in range(num_iterations):
            self.logger.info("batch: " + str(i + 1))
            X_test_batch = X_test[i * self.batch_size: (i + 1) * self.batch_size]
            X_test_str_batch = X_test_str[i * self.batch_size: (i + 1) * self.batch_size]
            y_test_psyduo_label_batch = y_test_psyduo_label[i * self.batch_size: (i + 1) * self.batch_size]

            if i == num_iterations - 1 and len(X_test_batch) < self.batch_size:
                X_test_batch = list(X_test_batch)
                X_test_str_batch = list(X_test_str_batch)
                y_test_psyduo_label_batch = list(y_test_psyduo_label_batch)
                gap = self.batch_size - len(X_test_batch)

                X_test_batch += [[0 for j in range(self.max_time_steps)] for i in range(gap)]
                X_test_str_batch += [['x' for j in range(self.max_time_steps)] for i in
                                     range(gap)]
                y_test_psyduo_label_batch += [[self.dataManager.label2id['O'] for j in range(self.max_time_steps)] for i
                                              in range(gap)]
                X_test_batch = np.array(X_test_batch)
                X_test_str_batch = np.array(X_test_str_batch)
                y_test_psyduo_label_batch = np.array(y_test_psyduo_label_batch)
                results, token, entity, entities_type, _ = self.predictBatch(self.sess, X_test_batch,
                                                                             y_test_psyduo_label_batch,
                                                                             X_test_str_batch)
                results = results[:len(X_test_batch)]
                token = token[:len(X_test_batch)]
                entity = entity[:len(X_test_batch)]
                entities_type = entities_type[:len(X_test_batch)]
            else:
                results, token, entity, entities_type, _ = self.predictBatch(self.sess, X_test_batch,
                                                                             y_test_psyduo_label_batch,
                                                                             X_test_str_batch)

            labels.extend(results)
            tokens.extend(token)
            entities.extend(entity)
            entities_types.extend(entities_type)

        def save_test_out(tokens, labels):
            # transform format
            newtokens, newlabels = [], []
            for to, la in zip(tokens, labels):
                newtokens.extend(to)
                newtokens.append("")
                newlabels.extend(la)
                newlabels.append("")
            # save
            save_csv_(pd.DataFrame({"token": newtokens, "label": newlabels}), self.output_test_file, ["token", "label"],
                      delimiter=self.configs.delimiter)

        save_test_out(tokens, labels)
        self.logger.info("testing results saved.\n")

        if self.is_output_sentence_entity:
            with open(self.output_sentence_entity_file, "w", encoding='utf-8') as outfile:
                for i in range(len(entities)):
                    if self.configs.label_level == 1:
                        outfile.write(' '.join(tokens[i]) + "\n" + "\n".join(entities[i]) + "\n\n")
                    elif self.configs.label_level == 2:
                        outfile.write(' '.join(tokens[i]) + "\n" + "\n".join(
                            [a + "\t(%s)" % b for a, b in zip(entities[i], entities_types[i])]) + "\n\n")

            self.logger.info("testing results with sentences&entities saved.\n")

        self.sess.close()


    def predict_single(self, sentence):
        X, Sentence, Y = self.dataManager.prepare_single_sentence(sentence)
        _, tokens, entitys, predicts_labels_entitylevel, indexs = self.predictBatch(self.sess, X, Y, Sentence)
        return tokens[0], entitys[0], predicts_labels_entitylevel[0], indexs[0]


    def predictBatch(self, sess, X, y_psydo_label, X_test_str_batch):
        entity_list = []
        tokens = []
        predicts_labels_entitylevel = []
        indexs = []
        predicts_labels_tokenlevel = []

        predicts_label_id, lengths = \
            sess.run([
                self.batch_pred_sequence,
                self.length
            ],
                feed_dict={
                    self.inputs: X,
                    self.targets: y_psydo_label,
                })

        for i in range(len(lengths)):
            x_ = [val for val in X_test_str_batch[i, 0:lengths[i]]]
            tokens.append(x_)

            y_pred = [str(self.dataManager.id2label[val]) for val in predicts_label_id[i, 0:lengths[i]]]
            predicts_labels_tokenlevel.append(y_pred)

            entitys, entity_labels, labled_indexs = extractEntity(x_, y_pred, self.dataManager)
            entity_list.append(entitys)
            predicts_labels_entitylevel.append(entity_labels)
            indexs.append(labled_indexs)

        return predicts_labels_tokenlevel, tokens, entity_list, predicts_labels_entitylevel, indexs

    def soft_load(self):
        self.logger.info("loading model parameter")
        tf.compat.v1.initialize_all_variables().run(session=self.sess)
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoints_dir))
        self.logger.info("loading model successfully")
