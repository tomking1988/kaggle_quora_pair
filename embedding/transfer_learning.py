from preprocess import preprocessor
import re
import numpy as np

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
                output_file.write(preprocessor.space_concatenator(glove_embeddings[cleansed_word]) + '\n')
                found += 1
            else:
                output_file.write(word + ' ')
                output_file.write(preprocessor.space_concatenator(np.random.rand(100) * 2 -1))
                output_file.write('\n')
                not_found +=1
            idx +=1
        print 'found:' + str(found) + ' not found:' + str(not_found)

def cleansing_word(word):
    word = re.sub(r"'s*\b", '', word)
    word = re.sub(r"\b'", '', word)
    word = re.sub(r'"', '', word)
    return word


if __name__ == "__main__":
    embeddings = load_glove('../data/glove.6B.100d.txt')
    vocab = load_vocab('../data/vocab.txt')
    init_domain_embedding(vocab, embeddings, '../data/embedding_init.txt')
