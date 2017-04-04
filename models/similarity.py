import tensorflow as tf
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from rnn import RNN
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
        self.rnn = RNN()
        self.rnn.build()
        init_op = tf.global_variables_initializer()
        self.sess =tf.Session()
        self.sess.run(init_op)
        self.reverse_vocab = preprocessor.load_reverse_vocab(Config.vocab_file_path)
        self.restore_variables(self.sess)

    def close(self):
        self.sess.close()

    def detect(self, question1, question2):
        question1 = preprocessor.sentence_to_numeric(question1, self.reverse_vocab)
        question2 = preprocessor.sentence_to_numeric(question2, self.reverse_vocab)
        predict = self.rnn.predict(self.sess, [question1], [question2])[0]
        return predict

    def restore_variables(self, sess):
        self.embedding_saver = tf.train.Saver({Config.embedding_variable_name: self.rnn.pretrained_embeddings})
        self.embedding_saver.restore(sess, Config.trained_embedding_file_path)
        self.saver = tf.train.Saver()
        self.saver.restore(sess, Config.trained_rnn_model_file_path)



if __name__ == '__main__':
    model = SentenceEncoding()
    question1 = 'what can make Physics easy to learn?'
    question2 = 'how can you make physics easy to learn?'
    predict = model.detect(question1, question2)
    print predict
    model.close()