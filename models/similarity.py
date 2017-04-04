import tensorflow as tf
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from preprocess import preprocessor

class Config:
    embed_size = 100
    vocab_size = 137077
    trained_embedding_file_path = '../data/trained_model/word2vec/trained_embeddings.ckpt'
    trained_rnn_model_file_path = '../data/trained_model/rnn/trained_rnn_model.ckpt'
    hidden_state_size = 100
    embedding_variable_name = 'embedding'
    batch_size = 100
    question1_index = 3
    question2_index = 4
    label_index = 5
    maximum_length = 240
    padding_embedding = [0] * embed_size
    padding = [vocab_size]
    scope_name = 'rnn'
    starter_learning_rate = 0.1
    dropout = 0.95
    vocab_file_path = '../data/vocab.txt'

class SentenceEncoding(object):
    def __init__(self):
        self.add_placeholder()
        self.build()
        init_op = tf.global_variables_initializer()
        self.sess =tf.Session()
        self.sess.run(init_op)
        self.restore_variables(self.sess)
        self.add_padding_embedding(self.sess)
        self.reverse_vocab = preprocessor.load_reverse_vocab(Config.vocab_file_path)

    def add_placeholder(self):
        self.question1_placeholder = tf.placeholder(shape=(None, None), dtype=tf.int32)
        self.question2_placeholder = tf.placeholder(shape=(None, None), dtype=tf.int32)
        self.question1_length_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.question2_length_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)

    def create_feed_dict(self, question1, question1_length, question2, question2_length):
        feed_dict = {}
        feed_dict[self.question1_placeholder]

    def build(self):
        self.pretrained_embeddings = tf.Variable(tf.zeros([Config.vocab_size, Config.embed_size]),
                                                 name=Config.embedding_variable_name, trainable=False)
        self.extended_embeddings = tf.Variable(tf.zeros([Config.vocab_size + 1, Config.embed_size]),
                                               name='extended_embeddings', trainable=False)
        with tf.variable_scope(Config.scope_name) as scope:

            sm_w = tf.get_variable(shape=(Config.hidden_state_size, Config.hidden_state_size),
                                   initializer=tf.contrib.layers.xavier_initializer(), name='sm_w')
            sm_w_output = tf.get_variable(shape=(Config.hidden_state_size, 1),
                                          initializer=tf.contrib.layers.xavier_initializer(), name='sm_w_1')
            sm_b = tf.get_variable(name='sm_b', shape=(Config.hidden_state_size,),
                                   initializer=tf.contrib.layers.xavier_initializer())
            sm_b_output = tf.get_variable(name='sm_b_1', shape=(1,),
                                          initializer=tf.contrib.layers.xavier_initializer())

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(Config.hidden_state_size)
            #[batch_size, sentence_size, embed_size]
            question1_embeddings = tf.nn.embedding_lookup(self.extended_embeddings, self.question1_placeholder)
            question2_embeddings = tf.nn.embedding_lookup(self.extended_embeddings, self.question2_placeholder)


            #[batch_size, hidden_size]
            question1_outputs, _= tf.nn.dynamic_rnn(cell=lstm_cell, dtype=tf.float32, inputs=question1_embeddings, sequence_length=self.question1_length_placeholder)
            scope.reuse_variables()
            question2_outputs, _= tf.nn.dynamic_rnn(cell=lstm_cell, dtype=tf.float32, inputs=question2_embeddings, sequence_length=self.question2_length_placeholder)

            index1 = tf.range(0, Config.batch_size) * Config.maximum_length + (self.question1_length_placeholder - 1)
            self.question1_outputs = tf.gather(tf.reshape(question1_outputs, [-1, Config.hidden_state_size]), index1)

            index2 = tf.range(0, Config.batch_size) * Config.maximum_length + (self.question2_length_placeholder - 1)
            self.question2_outputs = tf.gather(tf.reshape(question2_outputs, [-1, Config.hidden_state_size]), index2)

    def add_padding_embedding(self, sess):
        embeddings = self.pretrained_embeddings.eval(session=sess)
        embeddings = np.append(embeddings, [Config.padding_embedding], axis=0)
        tf.assign(self.extended_embeddings, embeddings)
        self.extended_embeddings.eval(sess)

    def restore_variables(self, sess):
        self.embedding_saver = tf.train.Saver({Config.embedding_variable_name: self.pretrained_embeddings})
        self.embedding_saver.restore(sess, Config.trained_embedding_file_path)
        self.saver = tf.train.Saver()
        self.saver.restore(sess, Config.trained_rnn_model_file_path)

    def detect(self, question1, question2):
        question1 = self.text_to_sequence(question1)
        question2 = self.text_to_sequence(question2)
        question1

    def text_to_sequence(self, text):
        preprocessor.sentence_to_numeric(text, self.reverse_vocab)


    def close(self):
        self.sess.close()

if __name__ == '__main__':
    model = SentenceEncoding()
    model.close()
    print 'test'