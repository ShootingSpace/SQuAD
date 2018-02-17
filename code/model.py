import logging
#from utils.general import batches, Progbar, get_random_samples, find_best_span, save_graphs
#from utils.eval import evaluate
import numpy as np
import tensorflow as tf
from os.path import join as pjoin
from abc import ABCMeta, abstractmethod

logging.basicConfig(level=logging.INFO)

class Model(metaclass=ABCMeta):
    @abstractmethod
    def add_placeholders(self):
        pass

    @abstractmethod
    def add_preds_op(self):
        pass

    @abstractmethod
    def add_loss_op(self, preds):
        pass

    @abstractmethod
    def add_training_op(self, loss):
        pass

    @abstractmethod
    def create_feed_dict(self, context, question, answer_span_start_batch=None, answer_span_end_batch=None,
                         is_train=True):
        pass

    @abstractmethod
    def setup_embeddings(self):
        pass

    def variable_summaries(var):
    """ Attach summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def get_optimizer(opt, loss, max_grad_norm, learning_rate):
        '''With gradient clipping'''
        if opt == "adam":
            optfn = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif opt == "sgd":
            optfn = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            assert (False)

        grads_and_vars = optfn.compute_gradients(loss)
        variables = [output[1] for output in grads_and_vars]
        gradients = [output[0] for output in grads_and_vars]

        gradients = tf.clip_by_global_norm(gradients, clip_norm=max_grad_norm)[0]
        grads_and_vars = [(gradients[i], variables[i]) for i in range(len(gradients))]
        train_op = optfn.apply_gradients(grads_and_vars)

        return train_op


    def build(self, config, result_saver):
        self.config = config
        #self.result_saver = result_saver
        self.preds = self.add_preds_op()
        self.loss = self.add_loss_op(self.preds)
        self.train_op = self.add_training_op(self.loss)

    def train(self, session, dataset, train_dir, vocab, which_model):
        ''' Implement main training loop
        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
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
        dataset['training']: a list
            [question, len(question), context, len(context), answer]
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
            train_writer_dir = self.config.log_dir + '/train/' # + datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
            self.train_writer = tf.summary.FileWriter(train_writer_dir, session.graph)
        for epoch in range(self.config.epochs):
            logging.info("="* 10 + " Epoch %d out of %d " + "="* 10, epoch + 1, self.config.epochs)

            score = self.run_epoch(session, epoch, training_set, vocab, validation_set, sample_size=self.config.evaluate_sample_size)
            logging.info("-- validation --")
            self.validate(session, validation_set)

            f1, em = self.evaluate_answer(session, validation_set, vocab, sample=self.config.model_selection_sample_size, log=True)
            # Saving the model
            if f1>f1_best:
                f1_best = f1
                saver = tf.train.Saver()
                saver.save(session, train_dir + which_model)
                logging.info('New best f1 in val set')
            logging.info('')

    def run_epoch(self, session, epoch_num, training_set, vocab, validation_set, sample_size=400):
        set_num = len(training_set)
        batch_size = self.config.batch_size
        batch_num = int(np.ceil(set_num * 1.0 / batch_size))
        sample_size = 400

        prog = Progbar(target=batch_num)
        avg_loss = 0
        for i, batch in enumerate(minibatches(training_set, self.config.batch_size, window_batch = self.config.window_batch)):
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
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        question_batch, question_len_batch, context_batch, context_len_batch, answer_batch = training_set
        input_feed = self.create_feed_dict(question_batch, question_len_batch,
                                            context_batch, context_len_batch,
                                            answer_batch=answer_batch,
                                            is_train = True)
        output_feed = [self.train_op, self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs
