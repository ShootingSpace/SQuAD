from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, datetime
import logging
from tqdm import tqdm
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from operator import mul
from tensorflow.python.ops import variable_scope as vs
from os.path import join as pjoin
from abc import ABCMeta, abstractmethod
from utils.util import *
# from utils.util import save_graphs, variable_summaries, get_optimizer, softmax_mask_prepro, ConfusionMatrix, Progbar, minibatches, one_hot, minibatch, get_best_span
from utils.evaluate import exact_match_score, f1_score
from utils.result_saver import ResultSaver

logging.basicConfig(level=logging.INFO)

class Encoder(object):
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
    def __init__(self, state_size):
        self.state_size = state_size

    def encode(self, inputs, masks, encoder_state_input = None, keep_prob = 1.0):
        # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
        cell = tf.contrib.rnn.BasicRNNCell(self.state_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = keep_prob)
        # defining initial state
        if encoder_state_input is not None:
            initial_state = encoder_state_input
        else:
            initial_state = None

        sequence_length = tf.reduce_sum(tf.cast(masks, 'int32'), axis=1)
        sequence_length = tf.reshape(sequence_length, [-1,])

        # Outputs Tensor shaped: [batch_size, max_time, cell.output_size]
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, sequence_length,
                                           initial_state = initial_state, dtype = tf.float32)

        return (outputs, final_state)

    def LSTM_encode(self, inputs, masks, encoder_state_input = None, keep_prob = 1.0):
        # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
        cell = tf.contrib.rnn.BasicLSTMCell(self.state_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = keep_prob)
        # defining initial state
        if encoder_state_input is not None:
            initial_state = encoder_state_input
        else:
            #initial_state = cell.zero_state(self.config.batch_size, dtype = tf.float32)
            initial_state = None

        sequence_length = tf.reduce_sum(tf.cast(masks, 'int32'), axis=1)
        sequence_length = tf.reshape(sequence_length, [-1,])

        # Outputs Tensor shaped: [batch_size, max_time, cell.output_size]
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, sequence_length,
                                           initial_state = initial_state, dtype = tf.float32)

        return (outputs, final_state)

    def BiLSTM_encode(self, inputs, masks, initial_state_fw=None, initial_state_bw=None, reuse=False, keep_prob = 1.0):
        return BiLSTM_layer(inputs, masks, self.state_size, initial_state_fw,
                                                                 initial_state_bw, reuse, keep_prob)

    def BiGRU_encode(self, inputs, masks, initial_state_fw=None, initial_state_bw=None, reuse=False, keep_prob = 1.0):
        return BiGRU_layer(inputs, masks, self.state_size, initial_state_fw,
                                                                 initial_state_bw, reuse, keep_prob)


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
    def __init__(self, output_size, state_size=None):
        self.output_size = output_size
        self.state_size = state_size

    def decode(self, knowledge_rep, mask, max_input_length, keep_prob = 1.0):
        with tf.variable_scope("start"):
            start = self.get_logit(knowledge_rep, max_input_length)
            start = softmax_mask_prepro(start, mask)

        with tf.variable_scope("end"):
            end = self.get_logit(knowledge_rep, max_input_length)
            end = softmax_mask_prepro(end, mask)

        return (start, end)

    def BiLSTM_decode(self, knowledge_rep, mask, max_input_length, keep_prob = 1.0):
        '''Decode with BiLSTM '''
        with tf.variable_scope('Modeling'):
            outputs, _, _ = BiLSTM_layer(knowledge_rep, mask, self.state_size, keep_prob=keep_prob)

        with tf.variable_scope("start"):
            start = self.get_logit(outputs, max_input_length)
            start = softmax_mask_prepro(start, mask)

        with tf.variable_scope("end"):
            end = self.get_logit(outputs, max_input_length)
            end = softmax_mask_prepro(end, mask)

        return (start, end)

    def BiGRU_decode(self, knowledge_rep, mask, max_input_length, keep_prob = 1.0):
        '''Decode with BiGRU'''
        with tf.variable_scope('Modeling'):
            outputs, _, _ = BiGRU_layer(knowledge_rep, mask, self.state_size, keep_prob=keep_prob)

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


class Attention(object):
    def __init__(self):
        pass

    def aoa(self, hc, hq, hc_mask, hq_mask, max_context_length_placeholder, max_question_length_placeholder):
        '''combine context hidden state(hc) and question hidden state(hq) with attention
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
        """
        :param hc: [None, max_context_length_placeholder, d_Bi]
        :param hq: [None, max_question_length_placeholder, d_Bi]
        :param hc_mask:  [None, max_context_length_placeholder]
        :param hq_mask:  [None, max_question_length_placeholder]

        :return: [N, max_context_length_placeholder, d_com]
        """
        pass

    def forwards_complex(self, hc, hq, hc_mask, hq_mask, max_context_length_placeholder, max_question_length_placeholder):
        '''combine context hidden state(hc) and question hidden state(hq) with attention
             measured similarity = hc : hq : hc.T * hq
        '''
        pass

    def forwards_bilinear(self, hc, hq, hc_mask, hq_mask, max_context_length_placeholder,
                                max_question_length_placeholder, is_train, keep_prob):
        '''combine context hidden state(hc) and question hidden state(hq) with global attention
            bilinear score = hc.T *W *hq
        '''
        d_en = hc.get_shape().as_list()[-1]
        # (BS, MPL, MQL)
        interaction_weights = tf.get_variable("W_interaction", shape=[d_en, d_en])
        hc_W = tf.reshape(tf.reshape(hc, shape=[-1, d_en]) @ interaction_weights,
                          shape=[-1, max_context_length_placeholder, d_en])

        # (BS, MPL, HS * 2) @ (BS, HS * 2, MCL) -> (BS ,MCL, MQL)
        score = hc_W @ tf.transpose(hq, [0, 2, 1])
        # Create mask (BS, MPL) -> (BS, MPL, 1) -> (BS, MPL, MQL)
        hc_mask_aug = tf.tile(tf.expand_dims(hc_mask, -1), [1, 1, max_question_length_placeholder])
        hq_mask_aug = tf.tile(tf.expand_dims(hq_mask, -2), [1, max_context_length_placeholder, 1])
        hq_mask_aug = hc_mask_aug & hq_mask_aug
        score = softmax_mask_prepro(score, hq_mask_aug)

        # (BS, MPL, MQL)
        alignment_weights = tf.nn.softmax(score)

        # (BS, MPL, MQL) @ (BS, MQL, HS * 2) -> (BS, MPL, HS * 2)
        context_aware = tf.matmul(alignment_weights, hq)

        concat_hidden = tf.concat([context_aware, hc], axis=2)
        concat_hidden = tf.cond(is_train, lambda: tf.nn.dropout(concat_hidden, keep_prob), lambda: concat_hidden)

        # (HS * 4, HS * 2)
        Ws = tf.get_variable("Ws", shape=[d_en * 2, d_en])
        attention = tf.nn.tanh(tf.reshape(tf.reshape(concat_hidden, [-1, d_en * 2]) @ Ws,
                                          [-1, max_context_length_placeholder, d_en]))
        return (attention)
        
    def forwards(self, hc, hq, hc_mask, hq_mask, max_context_length_placeholder, max_question_length_placeholder):
        '''combine context hidden state(hc) and question hidden state(hq) with: bidirectional attention flow
             simple score = hc.T * hq

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
        """
        :param hc: [None, max_context_length_placeholder, d_Bi]
        :param hq: [None, max_question_length_placeholder, d_Bi]
        :param hc_mask:  [None, max_context_length_placeholder]
        :param hq_mask:  [None, max_question_length_placeholder]

        :return: [N, max_context_length_placeholder, d_com]
        """
        logging.info('-'*5 + 'attention' + '-'*5)
        logging.debug('Context representation: %s' % str(hc))
        logging.debug('Question representation: %s' % str(hq))
        d_en = hc.get_shape().as_list()[-1]

        # get similarity
        hc_aug = tf.reshape(hc, shape = [-1, max_context_length_placeholder, 1, d_en])
        hq_aug = tf.reshape(hq, shape = [-1, 1, max_question_length_placeholder, d_en])
        hc_mask_aug = tf.tile(tf.expand_dims(hc_mask, -1), [1, 1, max_question_length_placeholder]) # [N, JX] -(expend)-> [N, JX, 1] -(tile)-> [N, JX, JQ]
        hq_mask_aug = tf.tile(tf.expand_dims(hq_mask, -2), [1, max_context_length_placeholder, 1]) # [N, JQ] -(expend)-> [N, 1, JQ] -(tile)-> [N, JX, JQ]

        similarity = tf.reduce_sum(tf.multiply(hc_aug, hq_aug), axis = -1) # h * u: [N, JX, d_en] * [N, JQ, d_en] -> [N, JX, JQ]
        hq_mask_aug = hc_mask_aug & hq_mask_aug

        similarity = softmax_mask_prepro(similarity, hq_mask_aug)

        # get a_x
        attention_c2q = tf.nn.softmax(similarity, dim=-1) # softmax -> [N, JX, softmax(JQ)]

        #     use a_x to get u_a
        attention_c2q = tf.reshape(attention_c2q,
                            shape = [-1, max_context_length_placeholder, max_question_length_placeholder, 1])
        hq_aug = tf.reshape(hq_aug, shape = [-1, 1, max_question_length_placeholder, d_en])
        hq_hat = tf.reduce_sum(tf.multiply(attention_c2q, hq_aug), axis = -2)# a_x * u: [N, JX, JQ](weight) * [N, JQ, d_en] -> [N, JX, d_en]
        logging.debug('Context with attention: %s' % str(hq_hat))

        # get a_q
        attention_q2c = tf.reduce_max(similarity, axis=-1) # max -> [N, JX]
        attention_q2c = tf.nn.softmax(attention_q2c, dim=-1) # softmax -> [N, softmax(JX)]
        #     use a_q to get h_a
        attention_q2c = tf.reshape(attention_q2c, shape = [-1, max_context_length_placeholder, 1])
        hc_aug = tf.reshape(hc, shape = [-1, max_context_length_placeholder, d_en])

        hc_hat = tf.reduce_sum(tf.multiply(attention_q2c, hc_aug), axis = -2)# a_q * h: [N, JX](weight) * [N, JX, d_en] -> [N, d_en]
        assert hc_hat.get_shape().as_list() == [None, d_en]
        hc_hat = tf.tile(tf.expand_dims(hc_hat, -2), [1, max_context_length_placeholder, 1]) # [None, JX, d_en]

        return tf.concat([hc, hq_hat, hc*hq_hat, hc*hc_hat], 2)

class Model(metaclass=ABCMeta):

    @abstractmethod
    def setup_system(self):
        pass

    @abstractmethod
    def setup_loss(self, preds):
        pass

    @abstractmethod
    def create_feed_dict(self, context, question, answer_span_start_batch = None,
                            answer_span_end_batch=None, is_train=True):
        pass

    @abstractmethod
    def setup_embeddings(self):
        pass

    def __init__(self, config):
        self.config = config
        self.result_saver = ResultSaver(self.config.output_dir)

    def setup_loss(self, preds):
        """ Set up loss computation
        :return:
        """
        with vs.variable_scope("loss"):
            s, e = preds # [None, max length]
            assert s.get_shape().ndims == 2
            assert e.get_shape().ndims == 2
            # loss1 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=s, labels=self.answer_start_placeholder),)
            # loss2 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=e, labels=self.answer_end_placeholder),)
            loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=s, labels=self.answer_start_placeholder),)
            loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=e, labels=self.answer_end_placeholder),)
        loss = loss1 + loss2
        tf.summary.scalar('loss', loss)

        return loss

    def build_exdma(self, opt_op):
        ''' Implement Exponential Moving Average'''
        self.exdma = tf.train.ExponentialMovingAverage(self.config.exdma_weight_decay)
        exdma_op = self.exdma.apply(tf.trainable_variables())
        with tf.control_dependencies([opt_op]):
            train_op = tf.group(exdma_op)
        return train_op

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

    def train(self, session, dataset, train_dir, vocab, checkpoint_prefix):
        ''' Implement main training loop
        TIPS:
        Implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        you will also want to save your model parameters in train_dir
        so that you can use your trained model to make predictions, or
        even continue training

        :param session: it should be passed in from train.py
        :param dataset: a dict of representation of our data,
            {"training": train, "validation": val,
            "question_maxlen": max_q_len,
            "context_maxlen": max_c_len}
        dataset['training']: one sample of the batch,
                a list [question, len(question), context, len(context), answer]
        answer: a list of index [as,ae]
        :param train_dir: path to the directory where you should save the model checkpoint

        '''
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        training_set = dataset['training']
        validation_set = dataset['validation']
        f1_best = 0
        if self.config.tensorboard:
            # + datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
            #train_writer_dir = self.config.log_dir + '/train/'
            print('tensorboard dir {}'.format(train_dir))
            self.train_writer = tf.summary.FileWriter(train_dir, session.graph)

        for epoch in range(self.config.epochs):
            logging.info("="* 20 + " Epoch %d out of %d " + "="* 20,
                                                epoch + 1, self.config.epochs)

            score = self.run_epoch(session, epoch, training_set, vocab, validation_set,
                                sample_size = self.config.evaluate_sample_size)
            logging.info("-- validation --")
            self.validate(session, validation_set)

            f1, em = self.evaluate_answer(session, validation_set, vocab,
                        sample=self.config.model_selection_sample_size, log=True)


            # Saving the model
            if f1>f1_best:
                f1_best = f1
                saver = tf.train.Saver()
                # The last one is the prefix of checkpoint files
                # saver.save(session, train_dir+self.config.which_model)
                saver.save(session, pjoin(train_dir, checkpoint_prefix))
                logging.info('New best f1 in val set')
            logging.info('')

    def run_epoch(self, session, epoch_num, training_set, vocab, validation_set,
                    sample_size = 400):
        tic = time.time()
        set_num = len(training_set)
        batch_size = self.config.batch_size
        batch_num = int(np.ceil(set_num * 1.0 / batch_size))

        self.result_saver.save("batch_size", self.config.batch_size)

        prog = Progbar(target=batch_num)
        avg_loss = 0

        for i, batch in enumerate(minibatches(training_set, self.config.batch_size,
                                        window_batch = self.config.window_batch)):
            global_batch_num = batch_num * epoch_num + i
            _, summary, loss = self.optimize(session, batch)
            prog.update(i + 1, [("training loss", loss)])

            self.result_saver.save("losses", loss)

            if self.config.tensorboard and global_batch_num % self.config.log_batch_num == 0:
                self.train_writer.add_summary(summary, global_batch_num)

            if (i+1) % self.config.log_batch_num == 0:
                f1_train, EM_train = self.evaluate_answer(session, training_set, vocab,
                            sample=sample_size, log=True, indicaiton = 'train')
                f1_val, EM_val = self.evaluate_answer(session, validation_set, vocab,
                            sample=sample_size, log=True, indicaiton = 'validation')

                # run the assign operation
                session.run([tf.assign(self.f1_train, f1_train), tf.assign(self.EM_train, EM_train),
                                      tf.assign(self.f1_val, f1_val), tf.assign(self.EM_val, EM_val)])

                self.result_saver.save("f1_train", f1_train)
                self.result_saver.save("EM_train", EM_train)
                self.result_saver.save("f1_val", f1_val)
                self.result_saver.save("EM_val", EM_val)


                batches_trained = 0 if self.result_saver.is_empty("batch_indices") \
                    else self.result_saver.get("batch_indices")[-1] + min(i + 1, self.config.log_batch_num)

                self.result_saver.save("batch_indices", batches_trained)
                save_graphs(self.result_saver.data,
                            path = self.config.output_dir)

            avg_loss += loss

        avg_loss /= batch_num
        toc = time.time()
        logging.info("Took {} secs for one epoch training, average training loss: {}".format(
                                                        toc - tic, avg_loss))
        return avg_loss

    def evaluate_answer(self, session, dataset, vocab, sample = 100, log = True, indicaiton = None):
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

        N = len(dataset)
        sampleIndices = np.random.choice(N, sample, replace = False)
        evaluate_set = [dataset[i] for i in sampleIndices]
        predicts = self.predict_on_batch(session, evaluate_set)

        for example, (start, end) in zip(evaluate_set, predicts):
            q, _, c, _, (true_s, true_e) = example
            # print (start, end, true_s, true_e)
            context_words = [vocab[w] for w in c]

            true_answer = ' '.join(context_words[true_s : true_e + 1])
            if start <= end:
                predict_answer = ' '.join(context_words[start : end + 1])
            else:
                predict_answer = ''

            f1 += f1_score(predict_answer, true_answer)
            em += exact_match_score(predict_answer, true_answer)

        f1 = 100 * f1 / sample
        em = 100 * em / sample

        if log:
            logging.info("Evaluate on {} {} samples - F1: {}, EM: {}, ".format(
                                    sample, indicaiton, f1, em ))

        return f1, em

    def optimize(self, session, training_set):
        """ Takes in actual data to optimize your model
        :return:
        """
        question_batch, question_len_batch, context_batch, context_len_batch, answer_batch = training_set
        input_feed = self.create_feed_dict(question_batch, question_len_batch,
                                            context_batch, context_len_batch,
                                            answer_batch=answer_batch,
                                            is_train = True)
        output_feed = [self.train_op, self.merged, self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, validation_set):
        """ compute a cost for validation set
        and tune hyperparameters according to the validation set performance
        :return:
        """
        question_batch, question_len_batch, context_batch, context_len_batch, answer_batch = validation_set
        input_feed = self.create_feed_dict(question_batch, question_len_batch,
                                            context_batch, context_len_batch,
                                            answer_batch = answer_batch,
                                            is_train = False)

        output_feed = [self.loss]
        outputs = session.run(output_feed, input_feed)

        return outputs


    def predict_on_batch(self, session, dataset):
        batch_num = int(np.ceil(len(dataset) * 1.0 / self.config.batch_size))
        # prog = Progbar(target=batch_num)
        predicts = []
        for i, batch in tqdm(enumerate(minibatches(dataset, self.config.batch_size, shuffle=False))):
            pred = self.answer(session, batch)
            # prog.update(i + 1)
            predicts.extend(pred)
        return predicts

    def answer(self, session, test_batch, use_best_span = False):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        # decode
        question_batch, question_len_batch, context_batch, context_len_batch, answer_batch = test_batch
        input_feed = self.create_feed_dict(question_batch, question_len_batch,
                                            context_batch, context_len_batch,
                                            answer_batch = None,
                                            is_train = False)

        output_feed = [self.preds[0], self.preds[1]]

        s, e = session.run(output_feed, input_feed)

        if use_best_span:
            spans, scores = zip(*[get_best_span(si, ei, ci)
                                    for si, ei, ci in zip(s, e, context_batch)])
        else:
            start_index = np.argmax(s, axis=1)
            end_index = np.argmax(e, axis=1)
            spans = zip(start_index, end_index)

        return spans

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.
        :return:
        """
        batch_num = int(np.ceil(len(valid_dataset) * 1.0 / self.config.batch_size))
        prog = Progbar(target = batch_num)

        valid_cost = 0

        for i, batch in enumerate(minibatches(valid_dataset, self.config.batch_size)):
            loss = self.test(sess, batch)[0]
            prog.update(i + 1, [("validation loss", loss)])
            valid_cost += loss
        valid_cost /= batch_num
        logging.info("Average validation loss: {}".format(valid_cost))

        return valid_cost

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
            feed_dict[self.dropout_placeholder] = self.config.keep_prob
        else:
            feed_dict[self.dropout_placeholder] = 1.0

        return feed_dict
