# -*- coding: utf-8 -*-
# @Time : 2019/6/2 上午10:52
# @Author : Scofield Phil
# @FileName: Parser.py
# @Project: sequence-lableing-vex
import sys


class Configer:
    def __init__(self, config_file='system.config'):
        config = self.config_file_to_dict(config_file)

        ## Status:
        the_item = 'mode'
        if the_item in config:
            self.mode = config[the_item]

        ## Datasets(Input/Output):
        the_item = 'datasets_fold'
        if the_item in config:
            self.datasets_fold = config[the_item]
        the_item = 'train_file'
        if the_item in config:
            self.train_file = config[the_item]
            the_item = 'dev_file'
        if the_item in config:
            self.dev_file = config[the_item]
        the_item = 'test_file'
        if the_item in config:
            self.test_file = config[the_item]
        the_item = 'delimiter'
        if the_item in config:
            self.delimiter = config[the_item]

        the_item = 'use_pretrained_embedding'
        if the_item in config:
            self.use_pretrained_embedding = self.str2bool(config[the_item])
        the_item = 'token_emb_dir'
        if the_item in config:
            self.token_emb_dir = config[the_item]

        the_item = 'vocabs_dir'
        if the_item in config:
            self.vocabs_dir = config[the_item]

        the_item = 'checkpoints_dir'
        if the_item in config:
            self.checkpoints_dir = config[the_item]

        the_item = 'log_dir'
        if the_item in config:
            self.log_dir = config[the_item]

        ## Labeling Scheme
        the_item = 'label_scheme'
        if the_item in config:
            self.label_scheme = config[the_item]

        the_item = 'label_level'
        if the_item in config:
            self.label_level = int(config[the_item])

        the_item = 'hyphen'
        if the_item in config:
            self.hyphen = config[the_item]

        the_item = 'suffix'
        if the_item in config:
            self.suffix = config[the_item]

        the_item = 'labeling_level'
        if the_item in config:
            self.labeling_level = config[the_item]

        the_item = 'measuring_metrics'
        if the_item in config:
            self.measuring_metrics = config[the_item]

        ## ModelConfiguration
        the_item = 'use_crf'
        if the_item in config:
            self.use_crf = self.str2bool(config[the_item])
        the_item = 'use_self_attention'
        if the_item in config:
            self.use_self_attention = self.str2bool(config[the_item])

        the_item = 'cell_type'
        if the_item in config:
            self.cell_type = config[the_item]
        the_item = 'biderectional'
        if the_item in config:
            self.biderectional = self.str2bool(config[the_item])
        the_item = 'encoder_layers'
        if the_item in config:
            self.encoder_layers = int(config[the_item])

        the_item = 'embedding_dim'
        if the_item in config:
            self.embedding_dim = int(config[the_item])

        the_item = 'max_sequence_length'
        if the_item in config:
            self.max_sequence_length = int(config[the_item])

        the_item = 'attention_dim'
        if the_item in config:
            self.attention_dim = int(config[the_item])

        the_item = 'hidden_dim'
        if the_item in config:
            self.hidden_dim = int(config[the_item])

        the_item = 'CUDA_VISIBLE_DEVICES'
        if the_item in config:
            self.CUDA_VISIBLE_DEVICES = config[the_item]

        the_item = 'seed'
        if the_item in config:
            self.seed = int(config[the_item])

        ## Training Settings:
        the_item = 'is_early_stop'
        if the_item in config:
            self.is_early_stop = self.str2bool(config[the_item])
        the_item = 'patient'
        if the_item in config:
            self.patient = int(config[the_item])

        the_item = 'epoch'
        if the_item in config:
            self.epoch = int(config[the_item])
        the_item = 'batch_size'
        if the_item in config:
            self.batch_size = int(config[the_item])

        the_item = 'dropout'
        if the_item in config:
            self.dropout = float(config[the_item])
        the_item = 'learning_rate'
        if the_item in config:
            self.learning_rate = float(config[the_item])

        the_item = 'optimizer'
        if the_item in config:
            self.optimizer = config[the_item]

        the_item = 'checkpoint_name'
        if the_item in config:
            self.checkpoint_name = config[the_item]

        the_item = 'checkpoints_max_to_keep'
        if the_item in config:
            self.checkpoints_max_to_keep = int(config[the_item])
        the_item = 'print_per_batch'
        if the_item in config:
            self.print_per_batch = int(config[the_item])

        ## Testing Settings

        the_item = 'output_test_file'
        if the_item in config:
            self.output_test_file = config[the_item]
        the_item = 'is_output_sentence_entity'
        if the_item in config:
            self.is_output_sentence_entity = self.str2bool(config[the_item])
        the_item = 'output_sentence_entity_file'
        if the_item in config:
            self.output_sentence_entity_file = config[the_item]

        ## Api service Settings
        the_item = 'ip'
        if the_item in config:
            self.ip = config[the_item]
        the_item = 'port'
        if the_item in config:
            self.port = config[the_item]

    def config_file_to_dict(self, input_file):
        config = {}
        fins = open(input_file, 'r').readlines()
        for line in fins:
            if len(line) > 0 and line[0] == "#":
                continue
            if "=" in line:
                pair = line.strip().split('#', 1)[0].split('=', 1)
                item = pair[0]
                value = pair[1]
                try:
                    if item in config:
                        print("Warning: duplicated config item found: %s, updated." % (pair[0]))
                    if value[0] == '[' and value[-1] == ']':
                        value_itmes = list(value[1:-1].split(','))
                        config[item] = value_itmes
                    else:
                        config[item] = value
                except:
                    print("configuration parsing error, please check correctness of the config file.")
                    exit(1)
        return config

    def str2bool(self, string):
        if string == "True" or string == "true" or string == "TRUE":
            return True
        else:
            return False

    def show_data_summary(self, logger):
        logger.info("\n")
        logger.info("++" * 20 + "CONFIGURATION SUMMARY" + "++" * 20)
        logger.info(" Status:")
        logger.info("     mode               : %s" % (self.mode))
        logger.info(" " + "++" * 20)
        logger.info(" Datasets:")
        logger.info("     datasets       fold: %s" % (self.datasets_fold))
        logger.info("     train          file: %s" % (self.train_file))
        logger.info("     developing     file: %s" % (self.dev_file))
        logger.info("     test           file: %s" % (self.test_file))
        logger.info("     pre trained embedin: %s" % (self.use_pretrained_embedding))
        logger.info("     embedding      file: %s" % (self.token_emb_dir))
        logger.info("     vocab           dir: %s" % (self.vocabs_dir))
        logger.info("     delimiter          : %s" % (self.delimiter))
        logger.info("     checkpoints     dir: %s" % (self.checkpoints_dir))
        logger.info("     log             dir: %s" % (self.log_dir))
        logger.info(" " + "++" * 20)
        logger.info("Labeling Scheme:")
        logger.info("     label        scheme: %s" % (self.label_scheme))
        logger.info("     label         level: %s" % (self.label_level))
        logger.info("     suffixs           : %s" % (self.suffix))
        logger.info("     labeling_level     : %s" % (self.labeling_level))
        logger.info("     measuring   metrics: %s" % (self.measuring_metrics))
        logger.info(" " + "++" * 20)
        logger.info("Model Configuration:")
        logger.info("     use             crf: %s" % (self.use_crf))
        logger.info("     use  self attention: %s" % (self.use_self_attention))
        logger.info("     cell           type: %s" % (self.cell_type))
        logger.info("     biderectional      : %s" % (self.biderectional))
        logger.info("     encoder      layers: %s" % (self.encoder_layers))
        logger.info("     embedding       dim: %s" % (self.embedding_dim))

        logger.info("     max sequence length: %s" % (self.max_sequence_length))
        logger.info("     attention       dim: %s" % (self.attention_dim))
        logger.info("     hidden          dim: %s" % (self.hidden_dim))
        logger.info("     CUDA VISIBLE DEVICE: %s" % (self.CUDA_VISIBLE_DEVICES))
        logger.info("     seed               : %s" % (self.seed))
        logger.info(" " + "++" * 20)
        logger.info(" Training Settings:")
        logger.info("     epoch              : %s" % (self.epoch))
        logger.info("     batch          size: %s" % (self.batch_size))
        logger.info("     dropout            : %s" % (self.dropout))
        logger.info("     learning       rate: %s" % (self.learning_rate))

        logger.info("     optimizer          : %s" % (self.optimizer))
        logger.info("     checkpoint     name: %s" % (self.checkpoint_name))

        logger.info("     max     checkpoints: %s" % (self.checkpoints_max_to_keep))
        logger.info("     print   per   batch: %s" % (self.print_per_batch))

        logger.info("     is   early     stop: %s" % (self.is_early_stop))
        logger.info("     patient            : %s" % (self.patient))
        logger.info(" " + "++" * 20)
        logger.info(" Training Settings:")
        logger.info("     output   test  file: %s" % (self.output_test_file))
        logger.info("     output sent and ent: %s" % (self.is_output_sentence_entity))
        logger.info("     output sen&ent file: %s" % (self.output_sentence_entity_file))
        logger.info(" " + "++" * 20)
        logger.info(" Api service Settings:")
        logger.info("     ip                 : %s" % (self.ip))
        logger.info("     port               : %s" % (self.port))

        logger.info("++" * 20 + "CONFIGURATION SUMMARY END" + "++" * 20)
        logger.info('\n\n')
        sys.stdout.flush()
