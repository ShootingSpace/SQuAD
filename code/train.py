from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin
from utils.data_reader import read_data, load_glove_embeddings

import logging
import baseline0
import baseline1
import baseline2
import lstm_decode

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.25, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 30, "Number of epochs to train.")

#tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("encoder_state_size", 100, "Size of each encoder model layer.")
tf.app.flags.DEFINE_integer("decoder_state_size", 100, "Size of each decoder model layer.")

tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
#tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "train", "Training directory to load model parameters from to resume training (default: '/train').")
#tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

tf.app.flags.DEFINE_string("which_model", "Baseline", "Which model to run")
tf.app.flags.DEFINE_string("question_maxlen", None, "Max length of question (default: 30")
tf.app.flags.DEFINE_string("context_maxlen", None, "Max length of the context (default: 400)")
tf.app.flags.DEFINE_integer("log_batch_num", 250, "Number of batches to evaluate answer and write logs on tensorboard.")
tf.app.flags.DEFINE_string("RE_TRAIN_EMBED", False, "Max length of the context (default: 400)")
tf.app.flags.DEFINE_float("exdma_weight_decay", 0.999, "exponential decay for moving averages ")
tf.app.flags.DEFINE_string("QA_ENCODER_SHARE", False, "Share the encoder weights")
tf.app.flags.DEFINE_string("tensorboard", True, "Write tensorboard log or not.")
tf.app.flags.DEFINE_integer("evaluate_sample_size", 400, "number of samples for evaluation (default: 100)")
tf.app.flags.DEFINE_integer("model_selection_sample_size", 1000, "# samples for making model update decision (default: 1000)")
tf.app.flags.DEFINE_integer("window_batch", 3, "window size / batch size")
tf.app.flags.DEFINE_string("output_dir", "output", "directory contains output graph(default: ./output).")


FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    checkpoint = tf.train.get_checkpoint_state(train_dir)
    if checkpoint:
        print("Find checkpoint from {}".format(train_dir))
    else: print("No checkpoint found in {}".format(train_dir))
    v2_path = checkpoint.model_checkpoint_path + ".index" if checkpoint else ""
    # print("checkpoint.model_checkpoint_path ", checkpoint.model_checkpoint_path)
    if checkpoint and (tf.gfile.Exists(checkpoint.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(session, checkpoint.model_checkpoint_path)
        save_path = saver.save(session, checkpoint.model_checkpoint_path)
        print("Model saved in path: %s" % save_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))

    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.decode().strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to train_dir from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def make_dirs(*args):
    '''check and make directory'''
    for _dir in args:
        if not os.path.exists(_dir):
            os.makedirs(_dir)


def main(_):
    # load datasets from FLAGS.data_dir
    dataset = read_data(FLAGS.data_dir)
    if FLAGS.context_maxlen is None:
        FLAGS.context_maxlen = dataset['context_maxlen']
    if FLAGS.question_maxlen is None:
        FLAGS.question_maxlen = dataset['question_maxlen']

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    embeddings = load_glove_embeddings(embed_path)

    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    FLAGS.output_dir = '{}/{}'.format(FLAGS.output_dir, FLAGS.which_model)

    # '''check and make directory'''
    # if not os.path.exists(FLAGS.output_dir):
    #     os.makedirs(FLAGS.output_dir, exist_ok = True)

    FLAGS.load_train_dir = '{}/{}/'.format(FLAGS.load_train_dir, FLAGS.which_model)
    #FLAGS.train_dir = '{}/{}/'.format(FLAGS.log_dir, 'train')
    make_dirs(FLAGS.output_dir, FLAGS.load_train_dir) #, FLAGS.train_dir)

    file_handler = logging.FileHandler(pjoin(FLAGS.output_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    logging.info('output directory: {}'.format(FLAGS.output_dir))
    #logging.info('train directory: {}'.format(FLAGS.train_dir))
    logging.info('load_train directory: {}'.format(FLAGS.load_train_dir))

    # if not os.path.exists(FLAGS.log_dir):
    #     os.makedirs(FLAGS.log_dir)



    logging.info("="* 10 + 'Running {} model'.format(FLAGS.which_model) + "="* 20)
    if FLAGS.which_model == "Baseline":
        qa = baseline0.QASystem(embeddings, FLAGS)
    elif FLAGS.which_model == "Baseline-LSTM":
        qa = baseline1.QASystem(embeddings, FLAGS)
    elif FLAGS.which_model == "Baseline-BiLSTM":
        qa = baseline2.QASystem(embeddings, FLAGS)
    elif FLAGS.which_model == "LSTM_decode":
        qa = lstm_decode.QASystem(embeddings, FLAGS)
    else:
        logging.info("No such specified model, use default baseline model")
        qa = baseline0.QASystem(embeddings, FLAGS)
    # elif FLAGS.which_model == "BiDAF":
    #         model = BiDAF(embeddings, FLAGS)
    # elif FLAGS.which_model == "LuongAttention":
    #     model = LuongAttention(embeddings, FLAGS)

    #print(vars(FLAGS))

    with open(os.path.join(FLAGS.output_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        #load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)

        initialize_model(sess, qa, FLAGS.load_train_dir)

        #save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, dataset, FLAGS.load_train_dir, rev_vocab, FLAGS.which_model)

        #qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
