import pandas as pd
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
import io
import re

class Config:
    test_csv_path = '../data/test.csv'
    train_csv_path = '../data/train.csv'
    seq2seq_training_file_path = '../data/seq2seq_train.csv'
    vocab_file_path = '../data/vocab.txt'

def extract_qeustions(input_file_path, output_file_path, mode='w+'):
    with io.open(input_file_path, encoding='utf-8') as input_file:
        with io.open(output_file_path, mode=mode, encoding='utf-8') as output_file:
            csv = pd.read_csv(input_file)
            questions1 = csv['question1'].tolist()
            questions2 = csv['question2'].tolist()
            questions = questions1 + questions2
            for idx, question in enumerate(questions):
                words = text_to_word_sequence(str(question))
                words = space_concatenator(words) + '\n'
                words = words.decode('utf-8')
                output_file.write(words)
                print 'processed: ' + str(idx)


def space_concatenator(words):
    return " ".join(map(str, words))

def dot_concatenator(words):
    return ",".join(map(str, words))

def create_vocab(input_file_path, output_file_path):
    vocab = set()
    with io.open(input_file_path, encoding='utf-8') as input_file:
        for idx, line in enumerate(input_file):
            words = text_to_word_sequence(line.encode(encoding='utf-8'))
            for word in words:
                vocab.add(word)
            print 'processed: ' + str(idx)
        with io.open(output_file_path, mode='w+', encoding='utf-8') as output_file:
            for word in vocab:
                output_file.write(word.add_decode('utf-8'))
                output_file.write('\n'.decode('utf-8'))

def load_reverse_vocab(vocab_file_path):
    reverse_idx_vocab = dict()
    with io.open(vocab_file_path, encoding='utf-8') as vocab_file:
        for idx, word in enumerate(vocab_file):
            reverse_idx_vocab[word.strip()] = idx
    return reverse_idx_vocab


def numerical_encoding(input_file_path, vocab_file_path, output_file_path):
    reverse_idx_vocab = dict()
    with io.open(vocab_file_path, encoding='utf-8') as vocab_file:
        for idx, word in enumerate(vocab_file):
            reverse_idx_vocab[word.strip()] = idx
        with io.open(input_file_path, encoding='utf-8') as input_file:
            with open(output_file_path, mode='w+') as output_file:
                for line in input_file:
                    words = line.split()
                    for i, word in enumerate(words):
                        words[i] = str(reverse_idx_vocab[word])
                    output_file.write(dot_concatenator(words) + '\n')

def numerical_encode_train(train_file_path, vocab_file_path, output_file_path):
    reverse_idx_vocab = dict()
    with io.open(vocab_file_path, encoding='utf-8') as vocab_file:
        for idx, word in enumerate(vocab_file):
            reverse_idx_vocab[word.strip()] = idx
        with io.open(train_file_path, encoding='utf-8') as train_file:
            csv = pd.read_csv(train_file)
            csv = csv[pd.notnull(csv['question1'])]
            csv = csv[pd.notnull(csv['question2'])]
            for idx, row in csv.iterrows():
                try:
                    csv.set_value(idx, 'question1', encoding_string(row['question1'], reverse_idx_vocab))
                    csv.set_value(idx, 'question2', encoding_string(row['question2'], reverse_idx_vocab))
                except:
                    print 'index: {}'.format(idx)
                    print row['question1']
                    print row['question2']
                    raise
                print 'processed:{}'.format(idx)
            csv.to_csv(output_file_path, encoding='utf-8', index=False, header=False)


def encoding_string(sentence, reverse_vocab):
    words = sentence_to_numeric(sentence, reverse_vocab)
    return space_concatenator(words)

def sentence_to_numeric(sentence, reverse_vocab):
    words = text_to_word_sequence(sentence)
    words = map(lambda word: word.decode('utf-8'), words)
    return map(lambda word: reverse_vocab[word], words)


def load_glove(glove_file_path, dimension=100):
    embeddings = {}
    with open(glove_file_path) as glove_file:
        for line in glove_file:
            tokens = line.split()
            word = tokens[0]
            embedding = [float(val) for val in tokens[1:]]
            embeddings[word] = embedding
    return embeddings

def load_vocab(vocab_file_path):
    vocab = list()
    with open(vocab_file_path) as vocab_file:
        for line in vocab_file:
            vocab.append(line.strip())
    return vocab

def init_domain_embedding(vocab, glove_embeddings, output_file_path):
    not_found = 0
    found = 0
    idx = 0
    with open(output_file_path, mode='w+') as output_file:
        for word in vocab:
            cleansed_word = cleansing_word(word)
            if cleansed_word in glove_embeddings:
                output_file.write(word + ' ')
                output_file.write(space_concatenator(glove_embeddings[cleansed_word]) + '\n')
                found += 1
            else:
                output_file.write(word + ' ')
                output_file.write(space_concatenator(np.random.rand(100) * 2 -1))
                output_file.write('\n')
                not_found +=1
            idx +=1
        print 'found:' + str(found) + ' not found:' + str(not_found)

def cleansing_word(word):
    word = re.sub(r"'s*\b", '', word)
    word = re.sub(r"\b'", '', word)
    word = re.sub(r'"', '', word)
    return word

def calculate_maximum_length(file_path):
    maximum = 0
    with open(file_path) as file:
        for line in file:
            words = line.split()
            if len(words) > maximum:
                maximum = len(words)
    print "maximum length of sentence: {}".format(maximum)
    return maximum

def generate_seq2seq_training_data():
    reverse_vocab = load_reverse_vocab(Config.vocab_file_path)
    with io.open(Config.train_csv_path, encoding='utf-8') as train_data_file, io.open(Config.test_csv_path, encoding='utf-8') as test_data_file:
        with io.open(Config.seq2seq_training_file_path, mode='w+', encoding='utf-8') as output_file:
            train_csv = pd.read_csv(train_data_file)
            train_csv = train_csv[pd.notnull(train_csv['question1'])]
            train_csv = train_csv[pd.notnull(train_csv['question2'])]

            questions1 = train_csv['question1'].tolist()
            questions2 = train_csv['question2'].tolist()
            is_duplicates = train_csv['is_duplicate'].tolist()
            for idx, is_duplicate in enumerate(is_duplicates):
                question1 = encoding_string(questions1[idx], reverse_vocab)
                question2 = encoding_string(questions2[idx], reverse_vocab)
                output_file.write(question1 + u',' + question1 + u'\n')
                output_file.write(question2 + u',' + question2 + u'\n')
                if is_duplicate:
                    output_file.write(question1 + u',' + question2 + u'\n')
                    output_file.write(question2 + u',' + question1 + u'\n')
                print 'processed: ' + str(idx)

            test_csv = pd.read_csv(test_data_file)
            test_csv = train_csv[pd.notnull(test_csv['question1'])]
            test_csv = train_csv[pd.notnull(test_csv['question2'])]

            questions1 = test_csv['question1'].tolist()
            questions2 = test_csv['question2'].tolist()
            for idx, question in enumerate(questions1):
                question1 = encoding_string(questions1[idx], reverse_vocab)
                question2 = encoding_string(questions2[idx], reverse_vocab)
                output_file.write(question1 + u',' + question1 + u'\n')
                output_file.write(question2 + u',' + question2 + u'\n')
                print 'processed: ' + str(idx)

if __name__ == "__main__":
    #extract_qeustions('../data/test.csv', '../data/questions.txt')
    #extract_qeustions('../data/train.csv', '../data/questions.txt', mode='a')
    #calculate_maximum_length('../data/questions.txt')
    #create_vocab('../data/questions.txt', '../data/vocab.txt')
    #numerical_encoding('../data/questions.txt', '../data/vocab.txt', '../data/numerical_questions.txt')
    #numerical_encode_train('../data/train.csv', '../data/vocab.txt', '../data/numerical_train.csv')
    #embeddings = load_glove('../data/glove.6B.100d.txt')
    #vocab = load_vocab('../data/vocab.txt')
    #init_domain_embedding(vocab, embeddings, '../data/embedding_init.txt')
    generate_seq2seq_training_data()