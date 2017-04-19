import tensorflow as tf
import numpy as np

class Config:
    embed_size = 100
    vocab_size = 137077
    epoch = 10
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
    starter_learning_rate = 0.5
    dropout = 0.85
    num_sampled = 20000
    max_grad_norm = 5.0
    clip_gradients = True


class Seq2seq(object):

    def add_placeholder(self):
        self.input_placeholder = tf.placeholder(shape=(None, Config.maximum_length), dtype=tf.int32)
        self.target_placeholder = tf.placeholder(shape=(None, Config.maximum_length), dtype=tf.int32)
        self.input_length_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.target_mask_placeholder = tf.placeholder(shape=(None, Config.maximum_length), dtype=tf.bool)
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32)

    def create_feed_dict(self, input, input_length, target=None, target_mask=None, dropout=1):
        feed_dict = {}
        feed_dict[self.input_placeholder] = input
        feed_dict[self.input_length_placeholder] = input_length
        if target is not None:
            feed_dict[self.target_placeholder] = target
            feed_dict[self.target_mask_placeholder] = target_mask
        feed_dict[self.dropout_placeholder] = dropout
        return feed_dict

    def add_loss(self):
        encodings = self.add_encode()
        outputs = self.add_decode(encodings)
        outputs = tf.reshape(outputs, [-1, Config.hidden_state_size])
        targets = tf.reshape(self.target_placeholder, [-1, 1])
        loss = tf.nn.sampled_softmax_loss(weights=tf.transpose(self.sm_w), biases=self.sm_b, labels=targets, inputs=outputs,
                                   num_sampled=Config.num_sampled, num_classes=Config.vocab_size + 1)
        target_masks = tf.reshape(self.target_mask_placeholder, [-1])
        loss = tf.boolean_mask(loss, target_masks)
        # sm_w_true = tf.nn.embedding_lookup(tf.transpose(self.sm_w), self.target_placeholder)
        # sm_b_true = tf.nn.embedding_lookup(self.sm_b, self.target_placeholder)
        # maximum_length = tf.shape(outputs)[1]
        # logits = tf.reshape(tf.reduce_mean(tf.multiply(outputs, sm_w_true),axis=2), [-1, maximum_length]) + sm_b_true
        loss = tf.reduce_mean(loss)
        return loss

    def add_encode(self):
        self.pretrained_embeddings = tf.Variable(tf.zeros([Config.vocab_size, Config.embed_size]),
                                                 name=Config.embedding_variable_name, trainable=False)
        self.extended_embeddings = tf.Variable(tf.zeros([Config.vocab_size + 1, Config.embed_size]),
                                               name='extended_embeddings', trainable=False)

        with tf.variable_scope(Config.scope_name):
            with tf.variable_scope('encode'):
                self.gru_encode_cell = tf.contrib.rnn.GRUCell(Config.hidden_state_size)
                # [batch_size, sentence_size, embed_size]
                inputs = tf.nn.embedding_lookup(self.extended_embeddings, self.input_placeholder)
                # Dropout embeddings
                input_embeddings = tf.nn.dropout(inputs, keep_prob=self.dropout_placeholder)

                # [input encoding batch_size, hidden_size]
                input_encodings, states = tf.nn.dynamic_rnn(cell=self.gru_encode_cell, dtype=tf.float32,
                                                            inputs=input_embeddings,
                                                            sequence_length=self.input_length_placeholder)
                batch_size = tf.shape(input_encodings)[0]
                maximum_length = tf.shape(input_encodings)[1]

                indices = tf.range(0, batch_size) * maximum_length + (self.input_length_placeholder - 1)
                input_encodings = tf.gather(tf.reshape(input_encodings, [-1, Config.hidden_state_size]), indices)
            self.encodings = input_encodings
            self.indices = indices
        return input_encodings

    def add_decode(self, input_encodings):
        with tf.variable_scope(Config.scope_name):
            with tf.variable_scope('decode'):
                self.sm_w = tf.get_variable('sm_w', [Config.hidden_state_size, Config.vocab_size + 1],
                                       initializer=tf.contrib.layers.xavier_initializer())
                self.sm_b = tf.get_variable('sm_b', [Config.vocab_size + 1], initializer=tf.contrib.layers.xavier_initializer())
                gru_decode_cell = tf.contrib.rnn.GRUCell(Config.hidden_state_size)

                def loop(prev, _):
                    prev = tf.matmul(prev, self.sm_w) + self.sm_b
                    prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
                    return tf.nn.dropout(tf.nn.embedding_lookup(self.extended_embeddings, prev_symbol),
                                         keep_prob=self.dropout_placeholder)

                targets = tf.nn.embedding_lookup(self.extended_embeddings, self.target_placeholder)
                targets = tf.split(targets, Config.maximum_length, 1)
                targets = [tf.squeeze(target, [1]) for target in targets]
                outputs, _ = tf.contrib.legacy_seq2seq.rnn_decoder(targets, input_encodings, gru_decode_cell,
                                                                   loop_function=loop)
                outputs = tf.stack(outputs, axis=1)
        self.outputs = outputs
        return outputs

    # def add_loss(self, logits):
    #     loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(tf.shape(logits)), logits=logits)
    #     loss = tf.boolean_mask(loss, self.target_mask_placeholder)
    #     loss = tf.reduce_sum(loss) / Config.batch_size
    #     return loss

    def add_optimizer(self, loss):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(Config.starter_learning_rate, global_step, 10, 0.9, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        variables = tf.trainable_variables()
        gvs = optimizer.compute_gradients(loss, variables)
        gradients = [gradient for gradient, var in gvs]
        if Config.clip_gradients:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=Config.max_grad_norm)
        self.grad_norm = tf.global_norm(gradients)
        train_op = optimizer.apply_gradients(grads_and_vars=zip(gradients, variables), global_step=global_step)
        return train_op

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
        inputs = []
        targets = []
        for idx, example in enumerate(example_batch):
            example = example.split(',')
            input = example[0].split()
            target = example[1].split()
            input = map(int, input)
            target = map(int, target)
            inputs.append(input)
            targets.append(target)
        return self.preprocess(inputs, targets)


    def build(self):
        self.add_placeholder()
        self.loss = self.add_loss()
        self.train_op = self.add_optimizer(self.loss)
        self.embedding_saver = tf.train.Saver({Config.embedding_variable_name: self.pretrained_embeddings})
        self.seq2seq_saver = tf.train.Saver()


    def restore_embedding(self, sess, embedding_file_path):
        self.embedding_saver.restore(sess, embedding_file_path)

    def train_batch(self, sess, inputs, inputs_length, targets, target_masks):
        feed_dict = self.create_feed_dict(inputs, inputs_length, targets, target_masks, Config.dropout)
        _,encodings, outputs, loss = sess.run([self.train_op, self.encodings, self.outputs, self.loss], feed_dict=feed_dict)
        print encodings
        return loss

    def encode_batch(self, sess, inputs):
        inputs, inputs_length = self.padding(inputs)
        feed_dict = self.create_feed_dict(input=inputs, input_length=inputs_length)
        encodings, indices = sess.run([self.encodings,self.indices], feed_dict=feed_dict)
        return encodings

    def save(self, sess, save_path):
        self.seq2seq_saver.save(sess, save_path)

    def preprocess(self, inputs, targets):
        target_mask = []
        inputs, inputs_length = self.padding(inputs)
        for target in targets:
            mask = [True] * len(target) + [False] * (Config.maximum_length - len(target))
            target_mask.append(mask)
            if len(target) < Config.maximum_length:
                target.extend(Config.padding * (Config.maximum_length - len(target)))
        return inputs, inputs_length, targets, target_mask

    def padding(self, sequence):
        sequence_length = []
        for element in sequence:
            sequence_length.append(len(element))
            if len(element) < Config.maximum_length:
                element.extend(Config.padding * (Config.maximum_length - len(element)))
        return sequence, sequence_length


    def add_padding_embedding(self, sess):
        embeddings = self.pretrained_embeddings.eval(session=sess)
        embeddings = np.append(embeddings, [Config.padding_embedding], axis=0)
        self.extended_embeddings = self.extended_embeddings.assign(embeddings)
        self.extended_embeddings.eval()



if __name__ == "__main__":
    model = Seq2seq()
    example_reader = model.create_example_batch('../data/seq2seq_train.csv', Config.batch_size)
    model.build()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(init_op)
        model.restore_embedding(sess, Config.trained_embedding_file_path)
        model.add_padding_embedding(sess)
        print 'start training'
        for step in range(Config.epoch):
            examples = sess.run(example_reader)
            inputs, inputs_length, targets, target_masks = model.create_batch(examples)
            loss = model.train_batch(sess, inputs, inputs_length, targets, target_masks)
            print "step:{}, loss:{}".format(step, loss)
        model.save(sess, Config.trained_seq2seq_model_file_path)
        coord.request_stop()
        coord.join(threads)