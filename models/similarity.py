import tensorflow as tf
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from seq2seq import Seq2seq
from preprocess import preprocessor
from numpy.linalg import norm

class Config:
    embed_size = 100
    vocab_size = 137077
    epoch = 50
    trained_embedding_file_path = '../data/trained_model/word2vec/trained_embeddings.ckpt'
    trained_seq2seq_model_file_path = '../data/trained_model/seq2seq/trained_seq2seq_model.ckpt'
    hidden_state_size = 50
    embedding_variable_name = 'embedding'
    batch_size = 50
    question1_index = 3
    question2_index = 4
    label_index = 5
    maximum_length = 240
    padding_embedding = [0] * embed_size
    padding = [vocab_size]
    scope_name = 'seq2seq'
    starter_learning_rate = 1.0
    dropout = 0.95
    vocab_file_path = '../data/vocab.txt'

class SentenceEncoding(object):
    def __init__(self):
        self.seq2seq = Seq2seq()
        self.seq2seq.build()
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
        encodings = self.seq2seq.encode_batch(self.sess, [question1, question2])
        encodings = np.array(encodings)
        return np.matmul(encodings[0], encodings[1])/(norm(encodings[0]) * norm(encodings[1]))



    def restore_variables(self, sess):
        self.embedding_saver = tf.train.Saver({Config.embedding_variable_name: self.seq2seq.pretrained_embeddings})
        self.embedding_saver.restore(sess, Config.trained_embedding_file_path)
        self.saver = tf.train.Saver()
        self.saver.restore(sess, Config.trained_seq2seq_model_file_path)



if __name__ == '__main__':
    model = SentenceEncoding()
    cases = []
    cases.append(('How can I be a good geologist?', 'What should I do to be a great geologist?'))
    cases.append(('How do I read and find my YouTube comments', 'How can I see all my Youtube comments'))
    cases.append(('Why do rockets look white','why are rockets and boosters painted white'))
    cases.append(('What is web application', 'what is the web application framework'))
    cases.append(('How can one increase concentration', 'How can I improve my concentration'))
    cases.append(('How do we start a business', 'How do I start business from nothing'))
    cases.append(('how racist is too racist', 'how racist are you'))
    cases.append(('Should prostitution be legalized','how is prostitution legal'))
    cases.append(('Is it important to be creative to be a writer','where can I find creative writer'))
    for question1, question2 in cases:
        probability = model.detect(question1, question2)
        print question1 + ' ' + question2 + ' probability:' + str(probability)
    model.close()