# coding: utf-8
import sys
import os
sys.path.append('..')
try:
    import urllib.request
except ImportError:
    raise ImportError('Use Python3!')
import pickle
import numpy as np


key_file = {
    'train':'Harry.train.txt',
    'test':'Harry.test.txt',
    'valid':'Harry.valid.txt'
}
save_file = {
    'train':'Harry.train.npy',
    'test':'Harry.test.npy',
    'valid':'Harry.valid.npy'
}
vocab_file = 'Harry.vocab.pkl'

dataset_dir = os.path.dirname(os.path.abspath(__file__))

def load_vocab():
    vocab_path = dataset_dir + '/' + vocab_file

    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
        return word_to_id, id_to_word

    word_to_id = {}
    id_to_word = {}
    data_type = 'train'
    file_name = key_file[data_type]
    file_path = dataset_dir + '/' + "Harry.all.txt"

    #_download(file_name)

    words = open(file_path).read().replace('\n', '<eos>').strip().split()

    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

    with open(vocab_path, 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f)

    return word_to_id, id_to_word


def load_data(data_type='train'):
    '''
        :param data_type: 데이터 유형: 'train' or 'test' or 'valid (val)'
        :return:
    '''
    if data_type == 'val': data_type = 'valid'
    save_path = dataset_dir + '/' + save_file[data_type]

    word_to_id, id_to_word = load_vocab()
    
    if os.path.exists(save_path):
        corpus = np.load(save_path)
        return corpus, word_to_id, id_to_word
    
    file_name = key_file[data_type]
    file_path = dataset_dir + '\\' + file_name
    #_download(file_name)
    print(file_path)
    #file_point = open('./dataset/ptb.train.txt', 'rt', encoding='UTF8')
    #words = open('./ptb.train.txt').read().replace('\n', '<eos>').strip().split()
    #words = open(file_path,'rt',encoding="utf-8").read()
    words = open(file_path).read().replace('\n', '<eos>').strip().split()
    corpus = np.array([word_to_id[w] for w in words])

    np.save(save_path, corpus)
    return corpus, word_to_id, id_to_word


if __name__ == '__main__':
    load_vocab()
    for data_type in ('train', 'val', 'test'):
        print(data_type)
        load_data(data_type)
