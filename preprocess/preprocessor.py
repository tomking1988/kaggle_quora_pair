import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
import string
import io

def extract_qeustions(input_file_path, output_file_path):
    with io.open(input_file_path, encoding='utf-8') as input_file:
        with open(output_file_path, mode='w+') as output_file:
            csv = pd.read_csv(input_file)
            questions1 = csv['question1'].tolist()
            questions2 = csv['question2'].tolist()
            questions = questions1 + questions2
            for idx, question in enumerate(questions):
                words = text_to_word_sequence(str(question))
                output_file.write(space_concatenator(words) + '\n')
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
        with open(output_file_path, mode='w+') as output_file:
            for word in vocab:
                output_file.write(word + '\n')

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


if __name__ == "__main__":
    #extract_qeustions('../data/test.csv', '../data/questions.txt')
    #create_vocab('../data/questions.txt', '../data/vocab.txt')
    numerical_encoding('../data/questions.txt', '../data/vocab.txt', '../data/numerical_questions.txt')