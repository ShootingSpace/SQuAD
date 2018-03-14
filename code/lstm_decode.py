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
from model import Model
from utils.result_saver import ResultSaver

logging.basicConfig(level=logging.INFO)

class Encoder(object):
    def __init__(self, size, config):
        self.size = size
        self.config = config

    def encode(self, inputs, masks, encoder_state_input = None, reuse = False, dropout = 1.0):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return:
                outputs: The RNN output Tensor
                          an encoded representation of your input.
                          It can be context-level representation,
                          word-level representation, or both.
                state: The final state.
        """
        logging.debug('-'*5 + 'encode by BiLSTM' + '-'*5)

        # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
        cell_fw = tf.contrib.rnn.BasicLSTMCell(self.size, reuse = reuse)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = dropout)

        cell_bw = tf.contrib.rnn.BasicLSTMCell(self.size, reuse = reuse)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob = dropout)

        # defining initial state
        if encoder_state_input is not None:
            initial_state_fw, initial_state_bw = encoder_state_input
        else:
            #initial_state = cell.zero_state(self.config.batch_size, dtype = tf.float32)
            initial_state_fw = None
            initial_state_bw = None

        sequence_length = tf.reduce_sum(tf.cast(masks, 'int32'), axis=1)
        sequence_length = tf.reshape(sequence_length, [-1,])

        # Outputs Tensor shaped: [batch_size, max_time, cell.output_size]
        (outputs_fw, outputs_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                                            cell_fw = cell_fw,\
                                            cell_bw = cell_bw,\
                                            inputs = inputs,\
                                            sequence_length = sequence_length,
                                            initial_state_fw = initial_state_fw,\
                                            initial_state_bw = initial_state_bw,
                                            dtype = tf.float32)

        outputs = tf.concat([outputs_fw, outputs_bw], 2)
        # final_state_fw and final_state_bw are the final states of the forwards/backwards LSTM
        final_state = tf.concat([final_state_fw[1], final_state_bw[1]], 1)
        return (outputs, final_state, (final_state_fw, final_state_bw))

class Decoder(object):
    """
    takes in a knowledge representation
    and output a probability estimation over
    all paragraph tokens on which token should be
    the start of the answer span, and which should be
    the end of the answer span.

    :param knowledge_rep: it is a representation of the paragraph and question,
                          decided by how you choose to implement the encoder
    :return: (start, end)
    """
    def __init__(self, output_size, state_size):
        self.output_size = output_size
        self.state_size = state_size

    def decode(self, knowledge_rep, mask, max_input_length, dropout = 1.0):
        '''Decode with 1 layer BiLSTM '''
        with tf.variable_scope('Modeling'):
            outputs, final_state, m_state = \
                 BiLSTM_layer(inputs=knowledge_rep, masks=mask, dropout = dropout,
                  state_size=self.state_size, encoder_state_input=None)

        with tf.variable_scope("start"):
            start = self.get_logit(outputs, max_input_length)
            start = softmax_mask_prepro(start, mask)

        with tf.variable_scope("end"):
            end = self.get_logit(outputs, max_input_length)
            end = softmax_mask_prepro(end, mask)

        return (start, end)


    def get_logit(self, inputs, max_inputs_length):
        ''' Get the logit (-inf, inf). '''
        d = inputs.get_shape().as_list()[-1]
        assert inputs.get_shape().ndims == 3, ("Got {}".format(inputs.get_shape().ndims))
        # -1 is used to infer the shape
        inputs = tf.reshape(inputs, shape = [-1, d])
        W = tf.get_variable('W', initializer=tf.contrib.layers.xavier_initializer(),
                             shape=(d, 1), dtype=tf.float32)
        pred = tf.matmul(inputs, W)
        pred = tf.reshape(pred, shape = [-1, max_inputs_length])
        tf.summary.histogram('logit', pred)
        return pred

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

        with tf.variable_scope('q'):
            hq, question_repr, question_state = \
                self.encoder.encode(self.question_embeddings,
                                    self.question_mask_placeholder)
            if self.config.QA_ENCODER_SHARE:
                #tf.get_variable_scope().reuse_variables()
                hc, context_state =\
                     self.encoder.encode(self.context_embeddings,
                                         self.context_mask_placeholder,
                                         encoder_state_input = question_state,
                                         reuse = True)

        if not self.config.QA_ENCODER_SHARE:
            with tf.variable_scope('c'):
                hc, context_repr, context_state =\
                     self.encoder.encode(self.context_embeddings,
                                         self.context_mask_placeholder,
                                         encoder_state_input = question_state)

        d_Bi = self.config.encoder_state_size*2
        assert hc.get_shape().as_list() == [None, None, d_Bi], (
                "Expected {}, got {}".format([None, self.max_context_length_placeholder,
                self.config.encoder_state_size], hc.get_shape().as_list()))
        assert hq.get_shape().as_list() == [None, None, d_Bi], (
                "Expected {}, got {}".format([None, self.max_question_length_placeholder,
                self.config.encoder_state_size], hq.get_shape().as_list()))

        '''Step 2: decoding   '''
        with tf.variable_scope("decoding"):
            start, end = self.decoder.decode(hc, self.context_mask_placeholder,
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

    def create_feed_dict(self, question_batch, question_len_batch, context_batch,
                        context_len_batch, max_context_length=10, max_question_length=10,
                        answer_batch=None, is_train = True):
        ''' Fill in this feed_dictionary like: feed_dict['train_x'] = train_x
        '''
        feed_dict = {}
        max_question_length = np.max(question_len_batch)
        max_context_length = np.max(context_len_batch)
        def add_paddings(sentence, max_length):
            mask = [True] * len(sentence)
            pad_len = max_length - len(sentence)
            if pad_len > 0:
                padded_sentence = sentence + [0] * pad_len
                mask += [False] * pad_len
            else:
                padded_sentence = sentence[:max_length]
                mask = mask[:max_length]
            return padded_sentence, mask

        def padding_batch(data, max_len):
            padded_data = []
            padded_mask = []
            for sentence in data:
                d, m = add_paddings(sentence, max_len)
                padded_data.append(d)
                padded_mask.append(m)
            return (padded_data, padded_mask)

        question, question_mask = padding_batch(question_batch, max_question_length)
        context, context_mask = padding_batch(context_batch, max_context_length)

        feed_dict[self.question_placeholder] = question
        feed_dict[self.question_mask_placeholder] = question_mask
        feed_dict[self.context_placeholder] = context
        feed_dict[self.context_mask_placeholder] = context_mask
        feed_dict[self.max_question_length_placeholder] = max_question_length
        feed_dict[self.max_context_length_placeholder] = max_context_length

        if answer_batch is not None:
            start = answer_batch[:,0]
            end = answer_batch[:,1]
            feed_dict[self.answer_start_placeholder] = start
            feed_dict[self.answer_end_placeholder] = end
        if is_train:
            feed_dict[self.dropout_placeholder] = 0.6
        else:
            feed_dict[self.dropout_placeholder] = 1.0

        return feed_dict
