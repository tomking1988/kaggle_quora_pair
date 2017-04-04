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
    scope_name = 'rnn'
    starter_learning_rate = 0.1
    dropout = 0.95

class RNN(object):

    def add_placeholder(self):
        self.question1_placeholder = tf.placeholder(shape=(None, None), dtype=tf.int32)
        self.question2_placeholder = tf.placeholder(shape=(None, None), dtype=tf.int32)
        self.label_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.question1_length_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.question2_length_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32)

    def create_feed_dict(self, question1, question1_length, question2, question2_length, label=None, dropout=1):
        feed_dict = {}
        feed_dict[self.question1_placeholder] = question1
        feed_dict[self.question1_length_placeholder] = question1_length
        feed_dict[self.question2_placeholder] = question2
        feed_dict[self.question2_length_placeholder] = question2_length
        if label is not None:
            feed_dict[self.label_placeholder] = label
        feed_dict[self.dropout_placeholder] = dropout
        return feed_dict

    def add_predict_op(self):
        self.pretrained_embeddings = tf.Variable(tf.zeros([Config.vocab_size, Config.embed_size]), name=Config.embedding_variable_name, trainable=False)
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

            #Dropout embeddings
            question1_embeddings = tf.nn.dropout(question1_embeddings, keep_prob=self.dropout_placeholder)
            question2_embeddings = tf.nn.dropout(question2_embeddings, keep_prob=self.dropout_placeholder)

            #[batch_size, hidden_size]
            question1_outputs, _= tf.nn.dynamic_rnn(cell=lstm_cell, dtype=tf.float32, inputs=question1_embeddings, sequence_length=self.question1_length_placeholder)
            scope.reuse_variables()
            question2_outputs, _= tf.nn.dynamic_rnn(cell=lstm_cell, dtype=tf.float32, inputs=question2_embeddings, sequence_length=self.question2_length_placeholder)
        #question1_states = tf.matmul(question1_states, sm_w)

        batch_size = tf.shape(question1_outputs)[0]
        maximum_length_1 = tf.shape(question1_outputs)[1]
        maximum_length_2 = tf.shape(question2_outputs)[1]

        index1 = tf.range(0, batch_size) * maximum_length_1 + (self.question1_length_placeholder - 1)
        question1_outputs = tf.gather(tf.reshape(question1_outputs, [-1, Config.hidden_state_size]), index1)
        question1_outputs = tf.nn.dropout(question1_outputs, keep_prob=self.dropout_placeholder)

        index2 = tf.range(0, batch_size) * maximum_length_2 + (self.question2_length_placeholder - 1)
        question2_outputs = tf.gather(tf.reshape(question2_outputs, [-1, Config.hidden_state_size]), index2)
        question2_outputs = tf.nn.dropout(question2_outputs, keep_prob=self.dropout_placeholder)

        question1_outputs = tf.matmul(question1_outputs, sm_w) + sm_b
        #logits = tf.matmul(question_encoding, sm_w_1) + sm_b_1

        merge_output = tf.multiply(question1_outputs, question2_outputs)
        logits = tf.matmul(merge_output, sm_w_output) + sm_b_output
        logits = tf.reshape(logits, shape=(-1,))
        self.predicts = tf.sigmoid(logits)
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