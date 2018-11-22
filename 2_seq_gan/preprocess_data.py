import re
import collections
from konlpy.tag import Twitter, Kkma
import pickle
import collections
import random
import numpy as np

def _save_pickle(path, data):
    # save pkl
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()


def _get_before_dataset():
    with open('../data/train_data.txt', encoding='utf-8') as f:
        texts = f.readlines()
        print(len(texts))

    count = 0
    stories = list()
    for desc in texts:
        #print(desc)
        sentences = list()
        text = desc.split('\t')[1].replace('\n', '').split('|')
        # print(text)
        for sent in text:
            if sent == '':
                continue
            sent = re.sub(r"[^ㄱ-힣a-zA-Z0-9]+", ' ', sent)
            try:
                if sent:
                    sent = analyzer.pos(sent)
                sentences.append(sent)
            except:
                print('sentence :',sent)
                pass

            count += 1
        stories.append(sentences)

    _save_pickle("../data/before_input_data.pkl", stories)

def create_sequence(seq_length, stories):
    data = list()
    for sent in stories:
        # 문장 개수만큼 for 문
        for i in range(len(sent)):
            seq_data = list()
            # seq_data 개수가 seq_length가 될때 까지
            while True:
                flag = 0
                for word in sent[i]:
                    if seq_length <= len(seq_data):
                        flag = 1
                        break
                    seq_data.append(word)
                if flag == 1:
                    break
                i += 1
                if i >= len(sent):
                    i -= 1
                    while seq_length > len(seq_data):
                        seq_data.append(('UNK', ''))
            data.append(seq_data)

    _save_pickle("../data/preprocessed_data.pkl", data)

    f = open('../data/preprocessed_data.txt', 'w')
    for tokens in data:
        for word in tokens:
            word = str(word) + ' '
            f.write(word)
        f.write('\n')
    f.close()

def data_to_index(dataset, pos2idx):
    idx_dataset = list()
    for sent in dataset:
        idx_sentence = list()
        for word in sent:
            idx_sentence.append(pos2idx[word[0]])
        idx_dataset.append(idx_sentence)

    _save_pickle("../data/data_index.pkl", idx_dataset)

    # save pk_data_index.txt
    f = open('../data/data_index.txt', 'w')
    for idx_sent in idx_dataset:
        for word in idx_sent:
            word = str(word) + ' '
            f.write(word)
        f.write('\n')
    f.close()

if __name__ == "__main__":
    seq_length = 30  # max 52
    analyzer = Kkma()

    # 이미 pkl 만들었으면 주석 처리, 처음 사용시 주석 해제
    _get_before_dataset()

    # load before dataset
    a = open("../data/before_input_data.pkl", 'rb')
    stories = pickle.load(a)

    print("Create Sequence in a length of seq_length...")
    create_sequence(seq_length, stories)

    print("Complete Creating sequence !!")

    # load after dataset
    a = open("../data/preprocessed_data.pkl", 'rb')
    dataset = pickle.load(a)

    # load pos to index
    a = open("../data/pos2idx.pkl", 'rb')
    pos2idx = pickle.load(a)

    print("Replace Sequence to Index...")
    data_to_index(dataset, pos2idx)

    print("Complete Creating sequence to index !!")