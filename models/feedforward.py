import tensorflow as tf

class Config:
    embed_size = 100
    encoding_size = 100
    vocab_size = 137077
    epoch = 10
    trained_embedding_file_path = '../data/trained_model/word2vec/trained_embeddings.ckpt'
    trained_seq2seq_model_file_path = '../data/trained_model/seq2seq/trained_seq2seq_model.ckpt'
    trained_feedforward_model_file_path = '../data/trained_model/feedforward/trained_feedforward_model.ckpt'
    embedding_variable_name = 'embedding'
    batch_size = 50
    question1_index = 3
    question2_index = 4
    label_index = 5
    maximum_length = 240
    padding_embedding = [0] * embed_size
    padding = [vocab_size]
    scope_name = 'seq2seq'
    starter_learning_rate = 0.5
    dropout = 0.85
    num_sampled = 20000
    max_grad_norm = 5.0
    clip_gradients = True

class FeedForwardClassifier(object):

    def add_placeholder(self):
        self.input_placeholder = tf.placeholder(shape=(None, Config.maximum_length), dtype=tf.int32)
        self.target_placeholder = tf.placeholder(shape=(None, Config.maximum_length), dtype=tf.int32)
        self.input_length_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.target_mask_placeholder = tf.placeholder(shape=(None, Config.maximum_length), dtype=tf.bool)
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32)
