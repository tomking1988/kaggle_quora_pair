import tensorflow as tf
import numpy as np
from seq2seq import Seq2seq

class Config:
    embed_size = 100
    encoding_size = 50
    vocab_size = 50773
    epoch = 20
    trained_embedding_file_path = '../data/trained_model/word2vec/trained_embeddings.ckpt'
    trained_seq2seq_model_file_path = '../data/trained_model/seq2seq/trained_seq2seq_model.ckpt'
    trained_feedforward_model_file_path = '../data/trained_model/feedforward/trained_feedforward_model.ckpt'
    embedding_variable_name = 'embedding'
    batch_size = 1000
    question1_index = 3
    question2_index = 4
    label_index = 5
    scope_name = 'feedforward'
    starter_learning_rate = 0.5
    dropout = 0.95
    layer1_output_size = 200

class FeedForwardClassifier(object):

    def __init__(self, trainable=True):
        self.trainable = trainable
        self.seq2seq = Seq2seq(trainable=False)
        self.seq2seq.build()
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)
        self.seq2seq.init(self.sess)

    def add_placeholder(self):
        #self.input_placeholder = tf.placeholder(shape=(None, Config.encoding_size * 2), dtype=tf.float32)
        self.question1_placeholder = tf.placeholder(shape=(None, Config.encoding_size), dtype=tf.float32)
        self.question2_placeholder = tf.placeholder(shape=(None, Config.encoding_size), dtype=tf.float32)
        self.target_placeholder = tf.placeholder(shape=(None, ), dtype=tf.float32)
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32)

    def create_feed_dict(self, question1, question2, target=None, dropout=1):
        feed_dict = {}
        feed_dict[self.question1_placeholder] = question1
        feed_dict[self.question2_placeholder] = question2

        if target is not None:
            feed_dict[self.target_placeholder] = target
            
        feed_dict[self.dropout_placeholder] = dropout
        return feed_dict

    def add_predict_op(self):
        with tf.variable_scope(Config.scope_name):
            self.w_1 = tf.get_variable('w_1', shape=(Config.encoding_size, Config.encoding_size),
                                       dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=self.trainable)

            self.b_1 = tf.get_variable('b_1', shape=(1, ),
                                       dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=self.trainable)

            # self.w_2 = tf.get_variable('w_2', shape=(Config.layer1_output_size, 1),
            #                            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=self.trainable)
            #
            # self.b_2 = tf.get_variable('b_2', shape=(1,),
            #                            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=self.trainable)
            self.question1_placeholder = tf.nn.dropout(self.question1_placeholder, self.dropout_placeholder)
            self.question2_placeholder = tf.nn.dropout(self.question2_placeholder, self.dropout_placeholder)
            #transformed_1 = tf.matmul(self.question1_placeholder, self.w_1)
            predict = tf.reduce_mean(tf.multiply(self.question1_placeholder, self.question2_placeholder), axis=1) + self.b_1
            predict = tf.squeeze(predict)
            self.predict = tf.sigmoid(predict)
        return predict

    def add_loss(self, predict):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_placeholder, logits=predict))

    def add_optimize(self, loss):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(Config.starter_learning_rate, global_step, 10, 0.9,staircase=True)
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
        targets = []
        question2s = []
        for idx, example in enumerate(example_batch):
            example = example.split(',')
            question1 = map(int, example[Config.question1_index].split())
            question2 = map(int, example[Config.question2_index].split())
            try:
                encodings = self.seq2seq.encode_batch(self.sess, [question1, question2])
            except:
                print question1
                print question2
            label = float(example[Config.label_index])
            targets.append(label)
            question1s.append(encodings[0])
            question2s.append(encodings[1])
        return self.balance_batch(question1s, question2s, targets)

    def balance_batch(self, question1s, question2s, targets):
        positive = 0
        negative = 0
        for label in targets:
            if label == 1.0:
                positive += 1
            else:
                negative += 1
        minimum = positive if positive < negative else negative
        balanced_question1s = []
        balanced_question2s = []
        balanced_targets = []
        positive = 0
        negative = 0
        for idx, label in enumerate(targets):
            if label == 1.0 and positive < minimum:
                balanced_question1s.append(question1s[idx])
                balanced_targets.append(targets[idx])
                balanced_question2s.append(question2s[idx])
            if label == 0.0 and negative < minimum:
                balanced_question1s.append(question1s[idx])
                balanced_targets.append(targets[idx])
                balanced_question2s.append(question2s[idx])
        return balanced_question1s, balanced_question2s, balanced_targets


    def train_batch(self, sess, question1s, question2s, targets):
        feed_dict = self.create_feed_dict(question1s, question2s, targets, Config.dropout)
        _, predict, loss = sess.run([self.train_op, self.predict, self.loss], feed_dict=feed_dict)
        print targets
        print predict
        return loss

    def build(self):
        self.add_placeholder()
        self.loss = self.add_loss(self.add_predict_op())
        self.train_op = self.add_optimize(self.loss)
        self.feedforward_saver = tf.train.Saver()

    def save(self, sess, save_path):
        self.feedforward_saver.save(sess, save_path)


if __name__ == '__main__':
    model = FeedForwardClassifier()
    example_reader = model.create_example_batch('../data/numerical_train.csv', Config.batch_size)
    model.build()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(init_op)
        print 'start training'
        for step in range(Config.epoch):
            examples = sess.run(example_reader)
            question1, question2s, targets = model.create_batch(examples)
            loss = model.train_batch(sess, question1, question2s, targets)
            print "step:{}, loss:{}".format(step, loss)
        model.save(sess, Config.trained_feedforward_model_file_path)
        coord.request_stop()
        coord.join(threads)