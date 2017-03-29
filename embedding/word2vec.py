import tensorflow as tf
import io
import numpy as np

class Config(object):
    question_batch_size = 100
    batch_size = 1000
    window_size = 2
    embed_size = 100
    vocab_size = 101343

class word2vec(object):

    def add_placeholder(self):
        self.batch_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.label_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.sample_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)

    def create_feed_dict(self, batch, label, sample):
        feed_dict = {}
        feed_dict[self.batch_placeholder] = batch
        feed_dict[self.label_placeholder] = label
        feed_dict[self.sample_placeholder] = sample
        return feed_dict

    def add_predict_op(self):
        sm_w_t = tf.get_variable(shape=)

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
        #value = tf.string_split([value], delimiter=',').values
        #value = tf.string_to_number(value, out_type=tf.int32)

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

    def build_graph(self):
        pass

if __name__ == "__main__":
    #example = read_training_file('../data/numerical_questions.txt')
    model = word2vec()
    question_batch = model.create_question_batch('../data/numerical_questions.txt', Config.question_batch_size)
    model.load_embeddings(embedding_file_path='../data/embedding_init.txt')
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(init_op)
        questions = sess.run(question_batch)
        batch, label = model.create_batch(questions, Config.batch_size)
        coord.request_stop()
        coord.join(threads)