import pickle
import numpy as np
import re

from src import paths


class DataUtils:

    @staticmethod
    def conll2lists(filename):
        with open(filename) as f:
            corpus = [[w.split() for w in s.split('\n')] for s in f.read().split('\n\n')]
        return corpus

    @staticmethod
    def conll2wordset(filename):
        train = DataUtils.conll2lists(filename)
        words = set((w[1] for s in train for w in s if w))
        return words

    @staticmethod
    def glove2dict_filtered(glove_filename, train_filename, test_filename):
        with open(glove_filename, 'r') as f:
            words_train = DataUtils.conll2wordset(train_filename)
            words_test = DataUtils.conll2wordset(test_filename)
            dictionary = {}

            for line in f:
                line = line.split()
                if line[0] in words_train or line[0] in words_test:
                    dictionary[line[0]] = np.array([float(n) for n in line[1:]])

        return dictionary

    @staticmethod
    def save_glove_dict(dictionary, filename):
        out = open(filename, 'wb')
        pickle.dump(dictionary, out)
        out.close()

    @staticmethod
    def load_glove_dict(filename):
        read = open(filename, 'rb')
        dic = pickle.load(read)
        return dic

    @staticmethod
    def replace_numbers(in_file, out_file):
        l = []
        with open(in_file, 'r') as f:
            for line in f:
                if line != '\n':
                    line = line.split()
                    if line[7] == 'num' or line[7] == 'number':
                        line[1] = '<NUM/>'
                    l.append('\t'.join(line) + '\n')
                else:
                    l.append('\n')

        with open(out_file, 'w') as f:
            f.writelines(l)

    @staticmethod
    def add_unk_glove(glove):
        vec_sum = np.zeros(300)

        for word in glove:
            vec_sum += glove[word] / np.linalg.norm(glove[word])

        glove['<NUM/>'] = vec_sum / np.linalg.norm(vec_sum)

    @staticmethod
    def add_num_glove(glove):
        vec_sum = np.zeros(300)
        regex = re.compile('^[^A-Za-z]*[0-9][^A-Za-z]*$')

        for word in glove:
            if re.match(regex, word) is not None:
                vec_sum += glove[word] / np.linalg.norm(glove[word])

        glove['<NUM/>'] = vec_sum / np.linalg.norm(vec_sum)

    # @staticmethod
    # def conll2word_tag(filename):
    #     train = DataUtils.conll2lists(filename)
    #     words = set(((w[1], w[7]) for s in train for w in s if w))
    #     return words

    @staticmethod
    def save_corpus(sentences, out_file):
        with open(out_file, 'w') as f:
            for i, s in enumerate(sentences):
                for j, w in enumerate(s):
                    if i != len(sentences) - 1 or j != len(s) - 1:
                        f.write('\t'.join(w) + '\n')
                    else:
                        f.write('\t'.join(w))

                if i != len(sentences) - 1:
                    f.write('\n')

    @staticmethod
    def output_format_corpus(sentences, actions, out_file):
        l = []
        for s, a in zip(sentences, actions):
            l.append(DataUtils.output_format(s, a))
        DataUtils.save_corpus(l, out_file)

    @staticmethod
    def output_format(sentence, actions):
        ac_res = [a.split('-') for a in actions]
        indexes = [0] + [int(w[0]) for w in sentence]

        x = y = 0

        for i, a in enumerate(ac_res):
            if a[0] == 'shift':
                x = y
                for y in range(y + 1, max(indexes) + 1, 1):  # set y to the first index avaible on the right
                    if y in indexes:
                        break
            elif a[1] == 'left':
                sentence[x][6] = str(y + 1)  # set the values of father and..
                sentence[x][7] = a[2]  # reason of reduction in the sentence
                indexes.remove(x)  # remove redced word from index
                for x in range(x - 1, -1, -1):  # set x to first index available on the left
                    if x in indexes:
                        break
            elif a[1] == 'right':
                sentence[y][6] = str(x + 1) if i < len(ac_res) - 1 else str(x)  # set the values of father and...
                sentence[y][7] = a[2]  # reason of reduction in the sentence
                indexes.remove(y)  # remove reduced word from index
                y = x  # since y has been removed set its value to x and..
                for x in range(x - 1, -1, -1):  # set x to first index available on the left
                    if x in indexes:
                        break

        return sentence

    @staticmethod
    def build_dataset(train_dependency_dataset, test_dependency_dataset, glove_file):
        train_file = train_dependency_dataset
        train_file_tagged = paths.PATH_TRAIN_STANFORD_TAGGED
        test_file = test_dependency_dataset
        test_file_tagged = paths.PATH_TEST_STANFORD_TAGGED

        DataUtils.replace_numbers(train_file, train_file_tagged)
        DataUtils.replace_numbers(test_file, test_file_tagged)

        glove = DataUtils.glove2dict_filtered(glove_file, train_file_tagged, test_file_tagged)
        DataUtils.save_glove_dict(glove, glove_file)

        glove = DataUtils.load_glove_dict(glove_file)
        DataUtils.add_num_glove(glove)
        DataUtils.add_unk_glove(glove)

        DataUtils.save_glove_dict(glove, paths.PATH_GLOVE_FILTERED_TAGGED)
