import tensorflow as tf
import numpy as np

class Config:
    embed_size = 100
    vocab_size = 137077
    epoch = 100
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
    scope_name = 'seq2seq'
    starter_learning_rate = 0.1
    dropout = 0.95

class Seq2seq(object):

    def add_placeholder(self):
        self.input_placeholder = tf.placeholder(shape=(None, None), dtype=tf.int32)
        self.target_placeholder = tf.placeholder(shape=(None, None), dtype=tf.int32)
        self.input_length_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.target_length_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32)

    def create_feed_dict(self, input, input_length, target=None, target_length=None, dropout=1):
        feed_dict = {}
        feed_dict[self.input_placeholder] = input
        feed_dict[self.input_length_placeholder] = input_length
        if target is not None:
            feed_dict[self.target_placeholder] = target
            feed_dict[self.target_length_placeholder] = target_length
        feed_dict[self.dropout_placeholder] = dropout
        return feed_dict

    def add_predict_op(self):
        self.pretrained_embeddings = tf.Variable(tf.zeros([Config.vocab_size, Config.embed_size]), name=Config.embedding_variable_name, trainable=False)
        self.extended_embeddings = tf.Variable(tf.zeros([Config.vocab_size + 1, Config.embed_size]),
                                                 name='extended_embeddings', trainable=False)

        with tf.variable_scope(Config.scope_name):
            with tf.variable_scope('encode'):
                lstm_encode_cell = tf.contrib.rnn.BasicLSTMCell(Config.hidden_state_size)
                #[batch_size, sentence_size, embed_size]
                inputs = tf.nn.embedding_lookup(self.extended_embeddings, self.input_placeholder)
                #Dropout embeddings
                input_embeddings = tf.nn.dropout(inputs, keep_prob=self.dropout_placeholder)

                #[input encoding batch_size, hidden_size]
                input_encodings, _= tf.nn.dynamic_rnn(cell=lstm_encode_cell, dtype=tf.float32, inputs=input_embeddings, sequence_length=self.input_length_placeholder)
                batch_size = tf.shape(input_encodings)[0]
                maximum_length = tf.shape(input_encodings)[1]

                indices = tf.range(0, batch_size) * maximum_length + (self.input_length_placeholder - 1)
                input_encodings = tf.gather(tf.reshape(input_encodings, [-1, Config.hidden_state_size]), indices)
            with tf.variable_scope('decode'):
                sm_w = tf.get_variable('sm_w', [Config.vocab_size, Config.hidden_state_size], initializer=tf.contrib.layers.xavier_initializer())
                sm_b = tf.get_variable('sm_b', [Config.vocab_size], initializer=tf.contrib.layers.xavier_initializer())
                lstm_decode_cell = tf.contrib.rnn.BasicLSTMCell(Config.hidden_state_size)
                def loop(prev, _):
                    prev = tf.matmul(sm_w, prev) + sm_b
                    prev_symbol = tf.stop_gradient(tf.argmax(prev,1))
                    return tf.nn.embedding_lookup(self.extended_embeddings, prev_symbol)
                outputs, _ =tf.contrib.legacy_seq2seq.rnn_decoder(self.target_placeholder, input_encodings, lstm_decode_cell, loop_function = loop)
                sm_w_true = tf.nn.embedding_lookup(sm_w, self.target_placeholder)
                sm_b_true = tf.nn.embedding_lookup(sm_b, self.target_placeholder)
                maximum_length = tf.shape(outputs)[1]
                logits = tf.reshape(tf.reduce_mean(tf.multiply(outputs, sm_b_true),axis=2), [-1, maximum_length]) + sm_b_true
                loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example()
        return logits

    def add_loss(self, logits):
        predicts = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(self.label_placeholder), logits=logits)
        loss = tf.reduce_sum(predicts) / Config.batch_size
        return loss

    def add_optimizer(self, loss):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(Config.starter_learning_rate, global_step, 10, 0.9, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(loss, global_step=global_step)

    def create_example_batch(self, file_path, batch_size):
        examples = self.read_training_file(file_path)
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size
        return tf.train.shuffle_batch([examples], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

    def read_training_file(self, file_path):
        """
        read line of file and output array of indexes
        :param file_path:
        :return: array[index, index,...]
        """
        filename_queue = tf.train.string_input_producer([file_path])
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        return value

    def create_batch(self, example_batch):
        question1s = []
        question2s = []
        labels = []
        for idx, example in enumerate(example_batch):
            example = example.split(',')
            question1 = example[Config.question1_index].split()
            question2 = example[Config.question2_index].split()
            label = int(example[Config.label_index])
            question1 = map(int, question1)
            question2 = map(int, question2)
            question1s.append(question1)
            question2s.append(question2)
            labels.append(label)
        question1s, question1_length, question2s, question2_length = self.padding(question1s, question2s)
        return question1s, question1_length, question2s, question2_length, labels


    def build(self):
        self.add_placeholder()
        self.logits = self.add_predict_op()
        self.loss = self.add_loss(self.logits)
        self.train_op = self.add_optimizer(self.loss)
        self.embedding_saver = tf.train.Saver({Config.embedding_variable_name: self.pretrained_embeddings})
        self.rnn_saver = tf.train.Saver()


    def restore_embedding(self, sess, embedding_file_path):
        self.embedding_saver.restore(sess, embedding_file_path)

    def train_batch(self, sess, question1s, question1_length, question2s, question2_length, labels):
        feed_dict = self.create_feed_dict(question1s, question1_length, question2s, question2_length, labels, Config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def predict(self, sess, question1s, question2s):
        question1s_length = map(len, question1s)
        question2s_length = map(len, question2s)
        feed_dict = self.create_feed_dict(question1=question1s, question1_length=question1s_length, question2=question2s, question2_length=question2s_length)
        predicts = sess.run(self.predicts, feed_dict=feed_dict)
        return predicts

    def save(self, sess, save_path):
        self.rnn_saver.save(sess, save_path)

    def padding(self, question1s, question2s):
        question1_length = []
        question2_length = []
        for question in question1s:
            question1_length.append(len(question))
            if len(question) < Config.maximum_length:
                question.extend(Config.padding * (Config.maximum_length - len(question)))
        for question in question2s:
            question2_length.append(len(question))
            if len(question) < Config.maximum_length:
                question.extend(Config.padding * (Config.maximum_length - len(question)))
        return question1s, question1_length, question2s, question2_length

    def add_padding_embedding(self, sess):
        embeddings = self.pretrained_embeddings.eval(session=sess)
        embeddings = np.append(embeddings, [Config.padding_embedding], axis=0)
        self.extended_embeddings = self.extended_embeddings.assign(embeddings)
        self.extended_embeddings.eval()



if __name__ == "__main__":
    model = RNN()
    example_reader = model.create_example_batch('../data/numerical_train.csv', Config.batch_size)
    model.build()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(init_op)
        model.restore_embedding(sess, Config.trained_embedding_file_path)
        model.add_padding_embedding(sess)
        for step in range(Config.epoch):
            examples = sess.run(example_reader)
            question1s, question1_length, question2s, question2_length, labels = model.create_batch(examples)
            loss = model.train_batch(sess, question1s, question1_length, question2s, question2_length, labels)
            print "step:{}, loss:{}".format(step, loss)
        model.save(sess, Config.trained_rnn_model_file_path)
        coord.request_stop()
        coord.join(threads)