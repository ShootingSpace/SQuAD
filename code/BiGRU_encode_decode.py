from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from utils.util import *
from utils.evaluate import exact_match_score, f1_score
from model import Model, Encoder, Decoder
from utils.result_saver import ResultSaver

logging.basicConfig(level=logging.INFO)

class QASystem(Model):
    def __init__(self, embeddings, config):
        """ Initializes System
        """
        #self.model = config.model
        self.embeddings = embeddings
        self.config = config

        self.result_saver = ResultSaver(self.config.output_dir)



        self.encoder = Encoder(config.encoder_state_size, self.config)
        self.decoder = Decoder(output_size=config.output_size, state_size = config.decoder_state_size)

        # ==== set up placeholder tokens ========
        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.context_mask_placeholder = tf.placeholder(tf.bool, shape=(None, None))
        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.question_mask_placeholder = tf.placeholder(tf.bool, shape=(None, None))

        self.answer_start_placeholder = tf.placeholder(tf.int32)
        self.answer_end_placeholder = tf.placeholder(tf.int32)

        self.max_context_length_placeholder = tf.placeholder(tf.int32)
        self.max_question_length_placeholder = tf.placeholder(tf.int32)
        self.dropout_placeholder = tf.placeholder(tf.float32)

        # ==== assemble pieces ====
        with tf.variable_scope(self.config.which_model, initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.question_embeddings, self.context_embeddings = self.setup_embeddings()
            self.preds = self.setup_system()
            self.loss = self.setup_loss(self.preds)
            self.f1_train = tf.Variable(0., tf.float64)
            self.EM_train = tf.Variable(0., tf.float64)
            self.f1_val = tf.Variable(0., tf.float64)
            self.EM_val = tf.Variable(0., tf.float64)
            tf.summary.scalar('f1_train', self.f1_train)
            tf.summary.scalar('EM_train', self.EM_train)
            tf.summary.scalar('f1_val', self.f1_val)
            tf.summary.scalar('EM_val', self.EM_val)

        # ==== set up training/updating procedure ====
        ''' With gradient clipping'''
        opt_op = get_optimizer(self.config.optimizer, self.loss, config.max_gradient_norm, config.learning_rate)

        if config.exdma_weight_decay is not None:
            self.train_op = self.build_exdma(opt_op)
        else:
            self.train_op = opt_op
        self.merged = tf.summary.merge_all()

    def build_exdma(self, opt_op):
        ''' Implement Exponential Moving Average'''
        self.exdma = tf.train.ExponentialMovingAverage(self.config.exdma_weight_decay)
        exdma_op = self.exdma.apply(tf.trainable_variables())
        with tf.control_dependencies([opt_op]):
            train_op = tf.group(exdma_op)
        return train_op

    def setup_system(self):
        """
        Connect all parts of your system here:
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        context: [None, max_context_length, d]
        question: [None, max_question_length, d]
        :return:
        """
        d = self.context_embeddings.get_shape().as_list()[-1] # self.config.embedding_size
        assert self.context_embeddings.get_shape().ndims == 3
        assert self.question_embeddings.get_shape().ndims == 3

        '''Step 1: encode context and question, respectively, with independent weights
        e.g. hq = encode_question(question)  # get U (d*J) as representation of q
        e.g. hc = encode_context(context, q_state)   # get H (d*T) as representation of x
        '''

        with tf.variable_scope('question'):
            hq, question_repr, question_state = \
                self.encoder.BiGRU_encode(self.question_embeddings, self.question_mask_placeholder,
                                    dropout = self.dropout_placeholder)
            if self.config.QA_ENCODER_SHARE:
                tf.get_variable_scope().reuse_variables()
                hc, context_state =\
                     self.encoder.BiGRU_encode(self.context_embeddings, self.context_mask_placeholder,
                                         encoder_state_input = question_state,
                                         dropout = self.dropout_placeholder)

        if not self.config.QA_ENCODER_SHARE:
            with tf.variable_scope('context'):
                hc, context_repr, context_state =\
                     self.encoder.BiGRU_encode(self.context_embeddings, self.context_mask_placeholder,
                                         encoder_state_input = question_state,
                                         dropout=self.dropout_placeholder)

        d_Bi = self.config.encoder_state_size*2
        assert hc.get_shape().as_list() == [None, None, d_Bi], (
                "Expected {}, got {}".format([None, self.max_context_length_placeholder,
                self.config.encoder_state_size], hc.get_shape().as_list()))
        assert hq.get_shape().as_list() == [None, None, d_Bi], (
                "Expected {}, got {}".format([None, self.max_question_length_placeholder,
                self.config.encoder_state_size], hq.get_shape().as_list()))

        '''Step 2: decoding   '''
        with tf.variable_scope("decoding"):
            start, end = self.decoder.BiGRU_decode(hc, self.context_mask_placeholder,
                                    self.max_context_length_placeholder, self.dropout_placeholder)
        return start, end

    def setup_loss(self, preds):
        """ Set up loss computation
        :return:
        """
        with vs.variable_scope("loss"):
            s, e = preds # [None, max length]
            assert s.get_shape().ndims == 2
            assert e.get_shape().ndims == 2
            loss1 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=s, labels=self.answer_start_placeholder),)
            loss2 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=e, labels=self.answer_end_placeholder),)
            # loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=s, labels=self.answer_start_placeholder),)
            # loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=e, labels=self.answer_end_placeholder),)
        loss = loss1 + loss2
        tf.summary.scalar('loss', loss)

        return loss

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return: embeddings representaion of question and context.
        """
        with tf.variable_scope("embeddings"):
            if self.config.RE_TRAIN_EMBED:
                embeddings = tf.get_variable("embeddings", initializer=self.embeddings)
            else:
                embeddings = tf.cast(self.embeddings, dtype=tf.float32)

            question_embeddings = tf.nn.embedding_lookup(embeddings, self.question_placeholder)
            question_embeddings = tf.reshape(question_embeddings,
                        shape = [-1, self.max_question_length_placeholder, self.config.embedding_size])

            context_embeddings = tf.nn.embedding_lookup(embeddings, self.context_placeholder)
            context_embeddings = tf.reshape(context_embeddings,
                        shape = [-1, self.max_context_length_placeholder, self.config.embedding_size])

        return question_embeddings, context_embeddings
