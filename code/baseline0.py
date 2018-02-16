from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from utils import prepro_for_softmax
from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, encoder_state_input):
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
        logging.debug('-'*5 + 'encode' + '-'*5)

        # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
        cell = tf.contrib.rnn.BasicRNNCell(size, reuse=reuse)

        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout)


        # defining initial state
        if encoder_state_input is not None:
            initial_state = encoder_state_input
        else:
            initial_state = cell.zero_state(batch_size, dtype=tf.float32)

        logging.debug('Inputs: %s' % (inputs.shape))
        logging.debug('Masks: %s' % (masks.shape))

        sequence_length = length(masks)

        # sequence_length = tf.reduce_sum(tf.cast(mask, 'int32'), axis=1)
        # Outputs Tensor shaped: [batch_size, max_time, cell.output_size]
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length,
                                           initial_state = initial_state,
                                           dtype = tf.float32)

        logging.debug("output shape: {}".format(output.get_shape()))

        return (output, state)


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep, mask, max_input_length, dropout = 1.0):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        with tf.variable_scope("start"):
            start = get_logit(knowledge_rep, max_input_length)
            start = prepro_for_softmax(start, mask)

        with tf.variable_scope("end"):
            end = get_logit(knowledge_rep, max_input_length)
            end = prepro_for_softmax(end, mask)

        return (start, end)

    def get_logit(self, inputs, max_inputs_length):
        ''' Get the logit (-inf, inf). '''
        d = inputs.get_shape().as_list()[-1]
        assert inputs.get_shape().ndims == 3
        # -1 is used to infer the shape
        inputs = tf.reshape(inputs, shape = [-1, d])
        W = tf.get_variable('W', initializer=tf.contrib.layers.xavier_initializer(), shape=(d, 1), dtype=tf.float32)
        pred = tf.matmul(inputs, W)
        pred = tf.reshape(pred, shape = [-1, max_inputs_length])
        tf.summary.histogram('logit', pred)
        return pred


class QASystem(object):
    def __init__(self,  result_saver, embeddings, config):
        """ Initializes System
        """
        self.embeddings = embeddings
        self.config = config
        self.encoder = Encoder(config.hidden_size)
        self.decoder = Decoder(config.hidden_size)

        # ==== set up placeholder tokens ========
        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.context_mask_placeholder = tf.placeholder(tf.bool, shape=(None, None))
        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.question_mask_placeholder = tf.placeholder(tf.bool, shape=(None, None))

        self.answer_span_start_placeholder = tf.placeholder(tf.int32)
        self.answer_span_end_placeholder = tf.placeholder(tf.int32)

        self.max_context_length_placeholder = tf.placeholder(tf.int32)
        self.max_question_length_placeholder = tf.placeholder(tf.int32)
        self.dropout_placeholder = tf.placeholder(tf.float32)

        # ==== assemble pieces ====
        with tf.variable_scope("baseline", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.question_embeddings, self.context_embeddings = self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        raise NotImplementedError("Connect all parts of your system here!")


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            pass

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return: embeddings representaion of question and context.
        """
        with tf.variable_scope("embeddings"):
            if self.config.retrain_embeddings:
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

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
