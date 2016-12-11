import pickle
import numpy as np
import re


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
        f = open(glove_filename, 'r')
        words_train = DataUtils.conll2wordset(train_filename)
        words_test = DataUtils.conll2wordset(test_filename)
        dictionary = {}
        for line in f:
            line = line.split()
            if line[0] in words_train or line[0] in words_test:
                dictionary[line[0]] = np.array([float(n) for n in line[1:]])
        f.close()
        return dictionary

    @staticmethod
    def save_glove_dict(dictionary, filename):
        out = open(filename, 'wb')
        pickle.dump(dictionary, out)
        out.close()

    # @staticmethod
    # def save_filtered_glove_dict(filename):
    #     dictionary = DataUtils.glove2dict_filtered()
    #     DataUtils.save_glove_dict(dictionary, filename)

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
                        line[1] = '<NUM>'
                    l.append('\t'.join(line) + '\n')
                else:
                    l.append('\n')

        with open(out_file, 'w') as f:
            f.writelines(l)

    @staticmethod
    def add_num_glove(glove):
        vec_sum = np.zeros(300)
        regex = re.compile('^[^A-Za-z]*[0-9][^A-Za-z]*$')

        for word in glove:
            if re.match(regex, word) is not None:
                vec_sum += glove[word]

        glove['<NUM>'] = vec_sum / np.linalg.norm(vec_sum)

    @staticmethod
    def conll2word_tag(filename):
        train = DataUtils.conll2lists(filename)

        words = set(((w[1], w[7]) for s in train for w in s if w))
        return words

    @staticmethod
    def output_format_corpus(sentences, actions, out_file):
        l = []
        for s, a in zip(sentences, actions):
            l.append(DataUtils.output_format(s, a))

        with open(out_file, 'w') as f:
            for s in l:
                for w in s:
                    f.write('\t'.join(w) + '\n')
                f.write('\n')

    @staticmethod
    def output_format(sentence, actions):
        ac_res = [a.split('-') for a in actions]
        indexes = [0] + [int(w[0]) for w in sentence]

        x = y = 0

        for i, a in enumerate(ac_res):
            if a[0] == 'shift':  ##Shift
                x = y
                for y in range(y + 1, max(indexes) + 1, 1):  # set y to the first index avaible on the right
                    if y in indexes:
                        break
            elif a[1] == 'left':  ##Reduce Left
                sentence[x][6] = str(y + 1)  # set the values of father and..
                sentence[x][7] = a[2]  # reason of reduction in the sentence
                indexes.remove(x)  # remove reducted word from index
                for x in range(x - 1, -1, -1):  # set x to first index avaible on the left
                    if x in indexes:
                        break
            elif a[1] == 'right':  ##Reduce Right
                sentence[y][6] = str(x + 1) if i < len(ac_res) - 1 else str(x)  # set the values of father and...
                sentence[y][7] = a[2]  # reason of reduction in the sentence
                indexes.remove(y)  # remove reducted word from index
                y = x  # since y has been removed set its value to x and..
                for x in range(x - 1, -1, -1):  # set x to first index avaible on the left
                    if x in indexes:
                        break

        return sentence