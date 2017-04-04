import tensorflow as tf
import numpy as np

class Config:
    embed_size = 100
    vocab_size = 137077
    epoch = 100
    trained_embedding_file_path = '../data/trained_model/trained_embeddings.ckpt'
    trained_rnn_model_file_path = '../data/trained_model/trained_rnn_model.ckpt'
    hidden_state_size = 50
    embedding_variable_name = 'embedding'
    batch_size = 500
    question1_index = 3
    question2_index = 4
    label_index = 5
    maximum_length = 250
    padding_embedding = [0] * embed_size
    padding = [vocab_size]
    scope_name = 'rnn'

class RNN(object):

    def add_placeholder(self):
        self.question1_placeholder = tf.placeholder(shape=(None, None), dtype=tf.int32)
        self.question2_placeholder = tf.placeholder(shape=(None, None), dtype=tf.int32)
        self.label_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.question1_length_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.question2_length_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)

    def create_feed_dict(self, question1, question1_length, question2, question2_length, label):
        feed_dict = {}
        feed_dict[self.question1_placeholder] = question1
        feed_dict[self.question1_length_placeholder] = question1_length
        feed_dict[self.question2_placeholder] = question2
        feed_dict[self.question2_length_placeholder] = question2_length
        feed_dict[self.label_placeholder] = label
        return feed_dict

    def add_predict_op(self):
        self.pretrained_embeddings = tf.Variable(tf.zeros([Config.vocab_size, Config.embed_size]), name=Config.embedding_variable_name, trainable=False)
        self.extended_embeddings = tf.Variable(tf.zeros([Config.vocab_size + 1, Config.embed_size]),
                                                 name='extended_embeddings', trainable=False)
        sm_w = tf.get_variable(shape=(Config.hidden_state_size, Config.hidden_state_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), name='sm_w_t')
        sm_b = tf.get_variable(name='sm_b', shape=(1,),
                               initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope(Config.scope_name) as scope:
            lstm_cell = tf.contrib.rnn.GRUCell(Config.hidden_state_size)
            #[batch_size, sentence_size, embed_size]
            question1_embeddings = tf.nn.embedding_lookup(self.extended_embeddings, self.question1_placeholder)
            question2_embeddings = tf.nn.embedding_lookup(self.extended_embeddings, self.question2_placeholder)
            #[batch_size, hidden_size]
            _, question1_states= tf.nn.dynamic_rnn(cell=lstm_cell, dtype=tf.float32, inputs=question1_embeddings, sequence_length=self.question1_length_placeholder)
            scope.reuse_variables()
            _, question2_states= tf.nn.dynamic_rnn(cell=lstm_cell, dtype=tf.float32, inputs=question2_embeddings, sequence_length=self.question2_length_placeholder)
        question1_encodings = tf.matmul(question1_states, sm_w)
        logits = tf.reduce_sum(tf.multiply(question1_encodings, question2_states), 1, keep_dims=True) + sm_b
        return tf.reshape(logits, shape=(-1, ))

    def add_loss(self, logits):
        predicts = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(self.label_placeholder), logits=logits)
        loss = tf.reduce_sum(predicts) / Config.batch_size
        return loss

    def add_optimizer(self, loss):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10, 0.6, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
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
        self.saver = tf.train.Saver({Config.embedding_variable_name: self.pretrained_embeddings})


    def restore_embedding(self, sess, embedding_file_path):
        self.saver.restore(sess, embedding_file_path)

    def train_batch(self, sess, question1s, question1_length, question2s, question2_length, labels):
        feed_dict = self.create_feed_dict(question1s, question1_length, question2s, question2_length, labels)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def save(self, sess, save_path):
        self.saver.save(sess, save_path)

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
        self.extended_embeddings.assign(embeddings)

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