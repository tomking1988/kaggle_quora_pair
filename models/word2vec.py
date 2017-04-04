import tensorflow as tf
import io
import numpy as np

class Config(object):
    question_batch_size = 5000
    batch_size = 50000
    window_size = 3
    embed_size = 100
    vocab_size = 137077
    epoch = 100
    trained_embedding_file_path = '../data/trained_model/word2vec/trained_embeddings.ckpt'
    dropout = 0.9

class Word2vec(object):

    def add_placeholder(self):
        self.batch_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.label_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.sample_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32)

    def create_feed_dict(self, batch, label, dropout=1):
        feed_dict = {}
        feed_dict[self.batch_placeholder] = batch
        feed_dict[self.label_placeholder] = label
        #feed_dict[self.sample_placeholder] = sample
        feed_dict[self.dropout_placeholder] = dropout
        return feed_dict

    def add_predict_op(self):
        sm_w_t = tf.get_variable(shape=(Config.vocab_size, Config.embed_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), name='sm_w_t')
        sm_b = tf.get_variable(name='sm_b', shape=(Config.vocab_size,),
                               initializer=tf.contrib.layers.xavier_initializer())
        batch_embeddings = tf.nn.embedding_lookup(self.pretrained_embeddings, self.batch_placeholder)
        batch_embeddings = tf.nn.dropout(batch_embeddings, keep_prob=self.dropout_placeholder)
        true_w = tf.nn.embedding_lookup(sm_w_t, self.label_placeholder)
        true_b = tf.nn.embedding_lookup(sm_b, self.label_placeholder)
        true_logits = tf.reduce_sum(tf.multiply(batch_embeddings, true_w), 1) + true_b
        return true_logits

    def add_loss(self, true_logits):
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(true_logits),
                                                            logits=true_logits)
        loss = tf.reduce_sum(true_xent) / Config.batch_size
        return loss

    def add_optimize(self, loss):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10, 0.9,staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(loss, global_step=global_step)

    def build(self):
        self.add_placeholder()
        self.true_logits = self.add_predict_op()
        self.loss = self.add_loss(self.true_logits)
        self.train_op = self.add_optimize(self.loss)
        self.saver = tf.train.Saver()

    def train_batch(self, sess, batch, label):
        feed_dict = self.create_feed_dict(batch, label, Config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def load_embeddings(self, embedding_file_path):
        embeddings = []
        with open(embedding_file_path) as vocab_file:
            for line in vocab_file:
                line = line.split()[1:]
                line = map(float, line)
                embeddings.append(line)
        self.pretrained_embeddings = tf.Variable(embeddings, name='embedding')


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

    def create_question_batch(self, file_path, batch_size):
        examples = self.read_training_file(file_path)
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size
        return tf.train.shuffle_batch([examples], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

    def create_batch(self, question_batch, batch_size):
        batchs = []
        labels = []
        for idx, question in enumerate(question_batch):
            if question == '':
                continue
            question = question.split(',')
            question = map(int, question)
            batch, label = self.skip_gram(question)
            batchs.extend(batch)
            labels.extend(label)
        batchs = np.array(batchs)
        labels = np.array(labels)
        batch_size = batch_size if batch_size < len(batchs) else len(batchs)
        idx = np.random.choice(len(batchs), size=batch_size, replace=False)
        return batchs[idx], labels[idx]

    def skip_gram(self, question):
        batch = []
        labels = []
        length = len(question)
        for idx, word in enumerate(question):
            for j in range(1, Config.window_size + 1):
                if idx - j >= 0:
                    batch.append(question[idx - j])
                    labels.append(word)
                if idx + j < length:
                    batch.append(question[idx + j])
                    labels.append(word)
        return batch, labels

    def save(self, sess, save_path):
        self.saver.save(sess, save_path)

if __name__ == "__main__":
    model = Word2vec()
    question_batch = model.create_question_batch('../data/numerical_questions.txt', Config.question_batch_size)
    model.load_embeddings(embedding_file_path='../data/embedding_init.txt')
    model.build()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(init_op)
        for step in range(Config.epoch):
            questions = sess.run(question_batch)
            batch, label = model.create_batch(questions, Config.batch_size)
            loss = model.train_batch(sess, batch, label)
            print "step:{}, loss:{}".format(step, loss)
        model.save(sess, Config.trained_embedding_file_path)
        coord.request_stop()
        coord.join(threads)