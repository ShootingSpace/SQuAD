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
from model import Model, Encoder, Decoder, Attention
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



        self.encoder = Encoder(config.encoder_state_size)
        self.decoder = Decoder(output_size=config.output_size, state_size = config.decoder_state_size)
        self.attention = Attention()

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
            hq, question_state_fw, question_state_bw = \
                self.encoder.BiGRU_encode(self.question_embeddings, self.question_mask_placeholder,
                                    keep_prob = self.dropout_placeholder)
            if self.config.QA_ENCODER_SHARE:
                #tf.get_variable_scope().reuse_variables()
                hc, context_state_fw, context_state_bw =\
                     self.encoder.BiGRU_encode(self.context_embeddings, self.context_mask_placeholder,
                             initial_state_fw = question_state_fw, initial_state_bw = question_state_bw,
                             reuse = True, keep_prob = self.dropout_placeholder)

        if not self.config.QA_ENCODER_SHARE:
            with tf.variable_scope('context'):
                hc, context_state_fw, context_state_bw =\
                     self.encoder.BiGRU_encode(self.context_embeddings, self.context_mask_placeholder,
                             initial_state_fw = question_state_fw, initial_state_bw = question_state_bw,
                                         keep_prob=self.dropout_placeholder)

        d_Bi = self.config.encoder_state_size*2
        assert hc.get_shape().as_list() == [None, None, d_Bi], (
                "Expected {}, got {}".format([None, self.max_context_length_placeholder,
                self.config.encoder_state_size], hc.get_shape().as_list()))
        assert hq.get_shape().as_list() == [None, None, d_Bi], (
                "Expected {}, got {}".format([None, self.max_question_length_placeholder,
                self.config.encoder_state_size], hq.get_shape().as_list()))

        '''Step 2: combine context hidden state(hc) and question hidden state(hq) with attention
             measured similarity = hc.T * hq

             Context-to-query (C2Q) attention signifies which query words are most relevant to each P context word.
                attention_c2q = softmax(similarity)
                hq_hat = sum(attention_c2q*hq)

             Query-to-context (Q2C) attention signifies which context words have the closest similarity
                to one of the query words and are hence critical for answering the query.
                attention_q2c = softmax(similarity.T)
                hc_hat = sum(attention_q2c*hc)

             combine with β activation: β function can be an arbitrary trainable neural network
             g = β(hc, hq, hc_hat, hq_hat)
        '''
        # concat[h, u_a, h*u_a, h*h_a]
        attention = self.attention.forwards_bilinear(hc, hq, self.context_mask_placeholder, self.question_mask_placeholder,
                                    max_context_length_placeholder = self.max_context_length_placeholder,
                                    max_question_length_placeholder = self.max_question_length_placeholder,
                                    is_train=(self.dropout_placeholder < 1.0), keep_prob=self.dropout_placeholder) 
        d_com = d_Bi*4



        '''Step 3: decoding   '''
        with tf.variable_scope("decoding"):
            start, end = self.decoder.BiGRU_decode(attention, self.context_mask_placeholder,
                                    self.max_context_length_placeholder, self.dropout_placeholder)
        return start, end
