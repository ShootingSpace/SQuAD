import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from os.path import join as pjoin
from abc import ABCMeta, abstractmethod
from utils.util import variable_summaries, get_optimizer, softmax_mask_prepro, ConfusionMatrix, Progbar, minibatches, one_hot, minibatch, get_best_span

logging.basicConfig(level=logging.INFO)

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


    def train(self, session, dataset, train_dir, vocab, which_model):
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
        if self.config.tensorboard and global_batch_num % self.config.log_batch_num == 0:
            # + datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
            train_writer_dir = self.config.log_dir + '/train/'

            self.train_writer = tf.summary.FileWriter(train_writer_dir, session.graph)

        for epoch in range(self.config.epochs):
            logging.info("="* 10 + " Epoch %d out of %d " + "="* 10,
                                                epoch + 1, self.config.epochs)

            score = self.run_epoch(session, epoch, training_set, vocab, validation_set,
                                sample_size=self.config.evaluate_sample_size)
            logging.info("-- validation --")
            self.validate(session, validation_set)

            f1, em = self.evaluate_answer(session, validation_set, vocab,
                        sample=self.config.model_selection_sample_size, log=True)
            # Saving the model
            if f1>f1_best:
                f1_best = f1
                saver = tf.train.Saver()
                saver.save(session, train_dir + which_model)
                logging.info('New best f1 in val set')
            logging.info('')

    def run_epoch(self, session, epoch_num, training_set, vocab, validation_set,
                    sample_size=400):
        set_num = len(training_set)
        batch_size = self.config.batch_size
        batch_num = int(np.ceil(set_num * 1.0 / batch_size))
        sample_size = 400

        prog = Progbar(target=batch_num)
        avg_loss = 0

        for i, batch in enumerate(minibatches(training_set, self.config.batch_size,
                                        window_batch = self.config.window_batch)):
            global_batch_num = batch_num * epoch_num + i
            _, summary, loss = self.optimize(session, batch)
            prog.update(i + 1, [("training loss", loss)])

            if self.config.tensorboard and global_batch_num % self.config.log_batch_num == 0:
                self.train_writer.add_summary(summary, global_batch_num)

            if (i+1) % self.config.log_batch_num == 0:
                logging.info('')
                self.evaluate_answer(session, training_set, vocab, sample=sample_size, log=True)
                self.evaluate_answer(session, validation_set, vocab, sample=sample_size, log=True)

            avg_loss += loss

        avg_loss /= batch_num
        logging.info("Average training loss: {}".format(avg_loss))
        return avg_loss

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

    # def decode(self, session, test_batch):
    #
    #     question_batch, question_len_batch, context_batch, context_len_batch, answer_batch = test_batch
    #     input_feed = self.create_feed_dict(question_batch, question_len_batch,
    #                                         context_batch, context_len_batch,
    #                                         answer_batch = None,
    #                                         is_train = False)
    #
    #     output_feed = [self.preds[0], self.preds[1]]
    #
    #     start, end = session.run(output_feed, input_feed)
    #
    #     return start, end

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
            spans = (start_index, end_index)

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
            avg_loss += loss
        valid_cost /= batch_num
        logging.info("Average validation loss: {}".format(avg_loss))

        return valid_cost

    def evaluate_answer(self, session, dataset, vocab, sample = 100, log = False):
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
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def predict_on_batch(self, session, dataset):
        batch_num = int(np.ceil(len(dataset) * 1.0 / self.config.batch_size))
        # prog = Progbar(target=batch_num)
        predicts = []
        for i, batch in tqdm(enumerate(minibatches(dataset, self.config.batch_size, shuffle=False))):
            pred = self.answer(session, batch)
            # prog.update(i + 1)
            predicts.extend(pred)
        return predicts
