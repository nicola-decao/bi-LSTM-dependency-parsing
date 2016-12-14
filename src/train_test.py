import numpy as np
import time

from src.data_utils import DataUtils
from src.dataset_parser import DatasetParser


class TrainTest:
    def __init__(self, word2vec):
        self.__word2vec = word2vec

    @staticmethod
    def __action2onehot(action):
        target = np.zeros(99)

        for i, e in enumerate(('reduce-left-abbrev', 'reduce-left-acomp', 'reduce-left-advcl', 'reduce-left-advmod',
                              'reduce-left-amod', 'reduce-left-appos', 'reduce-left-attr', 'reduce-left-aux',
                              'reduce-left-auxpass', 'reduce-left-cc',
                              'ccomp', 'reduce-left-complm', 'reduce-left-conj', 'reduce-left-cop', 'reduce-left-csubj',
                              'reduce-left-csubjpass', 'reduce-left-dep', 'reduce-left-det', 'reduce-left-dobj',
                              'reduce-left-expl',
                              'infmod', 'reduce-left-iobj', 'reduce-left-mark', 'reduce-left-mwe', 'reduce-left-neg',
                              'reduce-left-nn', 'reduce-left-npadvmod', 'reduce-left-nsubj', 'reduce-left-nsubjpass',
                              'reduce-left-num',
                              'number', 'reduce-left-parataxis', 'reduce-left-partmod', 'reduce-left-pcomp',
                              'reduce-left-pobj', 'reduce-left-poss', 'reduce-left-possessive', 'reduce-left-preconj',
                              'reduce-left-predet',
                              'prep', 'reduce-left-prt', 'reduce-left-punct', 'reduce-left-purpcl',
                              'reduce-left-quantmod', 'reduce-left-rcmod', 'reduce-left-rel', 'reduce-left-root',
                              'reduce-left-tmod', 'reduce-left-xcomp', 'shift', 'reduce-right-abbrev',
                              'reduce-right-acomp', 'reduce-right-advcl', 'reduce-right-advmod', 'reduce-right-amod',
                              'reduce-right-appos', 'reduce-right-attr', 'reduce-right-aux', 'reduce-right-auxpass',
                              'reduce-right-cc',
                              'ccomp', 'reduce-right-complm', 'reduce-right-conj', 'reduce-right-cop',
                              'reduce-right-csubj', 'reduce-right-csubjpass', 'reduce-right-dep', 'reduce-right-det',
                              'reduce-right-dobj', 'reduce-right-expl',
                              'infmod', 'reduce-right-iobj', 'reduce-right-mark', 'reduce-right-mwe',
                              'reduce-right-neg', 'reduce-right-nn', 'reduce-right-npadvmod', 'reduce-right-nsubj',
                              'reduce-right-nsubjpass', 'reduce-right-num',
                              'number', 'reduce-right-parataxis', 'reduce-right-partmod', 'reduce-right-pcomp',
                              'reduce-right-pobj', 'reduce-right-poss', 'reduce-right-possessive',
                              'reduce-right-preconj', 'reduce-right-predet',
                              'prep', 'reduce-right-prt', 'reduce-right-punct', 'reduce-right-purpcl',
                              'reduce-right-quantmod', 'reduce-right-rcmod', 'reduce-right-rel', 'reduce-right-root',
                              'reduce-right-tmod', 'reduce-right-xcomp')):
            if action == e:
                target[i] = 1

        return target

    @staticmethod
    def __onehot2action(target):
        return np.array(['reduce-left-abbrev', 'reduce-left-acomp', 'reduce-left-advcl', 'reduce-left-advmod',
                              'reduce-left-amod', 'reduce-left-appos', 'reduce-left-attr', 'reduce-left-aux',
                              'reduce-left-auxpass', 'reduce-left-cc',
                              'ccomp', 'reduce-left-complm', 'reduce-left-conj', 'reduce-left-cop', 'reduce-left-csubj',
                              'reduce-left-csubjpass', 'reduce-left-dep', 'reduce-left-det', 'reduce-left-dobj',
                              'reduce-left-expl',
                              'infmod', 'reduce-left-iobj', 'reduce-left-mark', 'reduce-left-mwe', 'reduce-left-neg',
                              'reduce-left-nn', 'reduce-left-npadvmod', 'reduce-left-nsubj', 'reduce-left-nsubjpass',
                              'reduce-left-num',
                              'number', 'reduce-left-parataxis', 'reduce-left-partmod', 'reduce-left-pcomp',
                              'reduce-left-pobj', 'reduce-left-poss', 'reduce-left-possessive', 'reduce-left-preconj',
                              'reduce-left-predet',
                              'prep', 'reduce-left-prt', 'reduce-left-punct', 'reduce-left-purpcl',
                              'reduce-left-quantmod', 'reduce-left-rcmod', 'reduce-left-rel', 'reduce-left-root',
                              'reduce-left-tmod', 'reduce-left-xcomp', 'shift', 'reduce-right-abbrev',
                              'reduce-right-acomp', 'reduce-right-advcl', 'reduce-right-advmod', 'reduce-right-amod',
                              'reduce-right-appos', 'reduce-right-attr', 'reduce-right-aux', 'reduce-right-auxpass',
                              'reduce-right-cc',
                              'ccomp', 'reduce-right-complm', 'reduce-right-conj', 'reduce-right-cop',
                              'reduce-right-csubj', 'reduce-right-csubjpass', 'reduce-right-dep', 'reduce-right-det',
                              'reduce-right-dobj', 'reduce-right-expl',
                              'infmod', 'reduce-right-iobj', 'reduce-right-mark', 'reduce-right-mwe',
                              'reduce-right-neg', 'reduce-right-nn', 'reduce-right-npadvmod', 'reduce-right-nsubj',
                              'reduce-right-nsubjpass', 'reduce-right-num',
                              'number', 'reduce-right-parataxis', 'reduce-right-partmod', 'reduce-right-pcomp',
                              'reduce-right-pobj', 'reduce-right-poss', 'reduce-right-possessive',
                              'reduce-right-preconj', 'reduce-right-predet',
                              'prep', 'reduce-right-prt', 'reduce-right-punct', 'reduce-right-purpcl',
                              'reduce-right-quantmod', 'reduce-right-rcmod', 'reduce-right-rel', 'reduce-right-root',
                              'reduce-right-tmod', 'reduce-right-xcomp'])[np.argmax(target[0])]

    @staticmethod
    def __tag2target(tag):
        a = tag.split('-')
        if len(a) > 1:
            a = a[1]
        else:
            a = a[0]

        target = np.zeros(3)
        for i, e in enumerate(('left', 'shift', 'right')):
            if a == e:
                target[i] = 1
        return target

    @staticmethod
    def __tag2targetMLP(tag):
        target = np.zeros(49)
        for i, e in enumerate(('abbrev', 'acomp', 'advcl', 'advmod', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'cc',
                               'ccomp', 'complm', 'conj', 'cop', 'csubj', 'csubjpass', 'dep', 'det', 'dobj', 'expl',
                               'infmod', 'iobj', 'mark', 'mwe', 'neg', 'nn', 'npadvmod', 'nsubj', 'nsubjpass', 'num',
                               'number', 'parataxis', 'partmod', 'pcomp', 'pobj', 'poss', 'possessive', 'preconj',
                               'predet',
                               'prep', 'prt', 'punct', 'purpcl', 'quantmod', 'rcmod', 'rel', 'root', 'tmod', 'xcomp')):
            if tag == e:
                target[i] = 1
        return target

    @staticmethod
    def __target2tag(target):
        return np.array(['reduce-left', 'shift', 'reduce-right'])[np.argsort(target[0])][::-1]

    @staticmethod
    def __target2tagMLP(target, second=False):
        labels = np.array(
            ['abbrev', 'acomp', 'advcl', 'advmod', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'cc', 'ccomp', 'complm',
             'conj', 'cop', 'csubj', 'csubjpass', 'dep', 'det', 'dobj', 'expl', 'infmod', 'iobj', 'mark', 'mwe', 'neg',
             'nn', 'npadvmod', 'nsubj', 'nsubjpass', 'num', 'number', 'parataxis', 'partmod', 'pcomp', 'pobj', 'poss',
             'possessive', 'preconj', 'predet', 'prep', 'prt', 'punct', 'purpcl', 'quantmod', 'rcmod', 'rel', 'root',
             'tmod', 'xcomp'])
        if second:
            return labels[np.argsort(target[0])][-2]
        else:
            return labels[np.argmax(target[0])]

    def training_complete(self, comlpete_biLSTM, sentences, save):
        t = 0

        for i, s in enumerate(sentences):

            start = time.time()
            parser = DatasetParser(s, self.__word2vec)
            s2v = parser.sentence2vec()

            ms1l = []
            ms2l = []
            mbl = []
            targets = []

            while parser.has_next():
                (ms1, ms2), mb = parser.get_stack(), parser.get_buffer()
                ms1l.append(parser.index2mask(ms1))
                ms2l.append(parser.index2mask(ms2))
                mbl.append(parser.index2mask(mb))

                tag = parser.next()
                target = self.__action2onehot(tag)
                targets.append(target)

            ms1l = np.array(ms1l)
            ms2l = np.array(ms2l)
            mbl = np.array(mbl)

            targets = np.array(targets)

            print(i, comlpete_biLSTM.get_model().train_on_batch([np.array([s2v] * ms1l.shape[0]), ms1l, ms2l, mbl], targets))
            t += (time.time() - start)
            print(t)

            if i % 100 == 99:
                comlpete_biLSTM.get_model().save_weights(save)
                print('Saved!')

        comlpete_biLSTM.get_model().save_weights(save)

    def training(self, biLSTM, sentences, save):
        t = 0

        for i, s in enumerate(sentences):

            start = time.time()
            parser = DatasetParser(s, self.__word2vec)
            s2v = parser.sentence2vec()

            ms1l = []
            ms2l = []
            mbl = []
            targets = []

            while parser.has_next():
                (ms1, ms2), mb = parser.get_stack(), parser.get_buffer()
                ms1l.append(parser.index2mask(ms1))
                ms2l.append(parser.index2mask(ms2))
                mbl.append(parser.index2mask(mb))

                tag = parser.next()
                target = self.__tag2target(tag)
                targets.append(target)

            ms1l = np.array(ms1l)
            ms2l = np.array(ms2l)
            mbl = np.array(mbl)

            targets = np.array(targets)

            print(i, biLSTM.get_model().train_on_batch([np.array([s2v] * ms1l.shape[0]), ms1l, ms2l, mbl], targets))
            t += (time.time() - start)
            print(t)

            if i % 100 == 99:
                biLSTM.get_model().save_weights(save)
                print('Saved!')

        biLSTM.get_model().save_weights(save)

    def __predict(self, biLSTM, sentence):
        parser = DatasetParser(sentence, self.__word2vec)
        s2v = np.array([parser.sentence2vec()])

        tags = []
        while parser.has_next():
            (ms1, ms2), mb = parser.get_stack(), parser.get_buffer()
            ms1 = np.array([parser.index2mask(ms1)])
            ms2 = np.array([parser.index2mask(ms2)])
            mb = np.array([parser.index2mask(mb)])

            target = biLSTM.get_model().predict([s2v, ms1, ms2, mb])
            tag = self.__target2tag(target)
            true_tag = parser.next(tag)
            tags.append(true_tag)

        return tags

    def test(self, biLSTM, sentences, out_file):
        actions = []
        for i, s in enumerate(sentences):
            actions.append(self.__predict(biLSTM, s))
            print(i + 1, '/', len(sentences))
        DataUtils.output_format_corpus(sentences, actions, out_file)

    @staticmethod
    def __sentence2couples(sentence):
        return [[int(w[6]), i + 1, w[7]] for i, w in enumerate(sentence)]

    def training_tagger(self, slave_BiLSTM, MLPTags_model, sentences, save):
        t = 0

        for i, s in enumerate(sentences):

            start = time.time()
            parser = DatasetParser(s, self.__word2vec)
            s2v = parser.sentence2vec()

            lstm = slave_BiLSTM.get_model().predict(np.array([s2v]))[0]

            heads = []
            tails = []
            targets = []

            for c in self.__sentence2couples(s):
                heads.append(lstm[c[0]])
                tails.append(lstm[c[1]])
                targets.append(self.__tag2targetMLP(c[2]))

            heads = np.array(heads)
            tails = np.array(tails)
            targets = np.array(targets)

            print(i, MLPTags_model.get_model().train_on_batch([heads, tails], targets))
            t += (time.time() - start)
            print(t)

            if i % 100 == 99:
                MLPTags_model.get_model().save_weights(save)
                print('Saved!')

        MLPTags_model.get_model().save_weights(save)

    def test_tagger(self, slave_BiLSTM, MLPTags_model, in_file, out_file):
        sentences = DataUtils.conll2lists(in_file)

        for i, s in enumerate(sentences):
            parser = DatasetParser(s, self.__word2vec)
            s2v = parser.sentence2vec()
            dep = self.__sentence2couples(s)
            lstm = slave_BiLSTM.get_model().predict(np.array([s2v]))[0]

            for j, d in enumerate(dep):
                target = MLPTags_model.get_model().predict([np.reshape(lstm[d[0]], (1, 400)),
                                                            np.reshape(lstm[d[1]], (1, 400))])
                tag = self.__target2tagMLP(target)
                if tag == 'root' and j != len(dep) - 1:
                    tag = self.__target2tagMLP(target, True)
                elif j == len(dep) - 1:
                    tag = 'root'

                s[d[1] - 1][7] = tag
            print(i, '/', len(sentences))

        DataUtils.save_corpus(sentences, out_file)

    @staticmethod
    def __sentence2couplesParent(sentence):
        sentence = [['0', '<ROOT/>', '_', 'ROOT', '_', '_', '0', '_', '_', '_']] + sentence
        return [[int(sentence[int(w[6])][6]), int(w[6]), i, w[7]] for i, w in enumerate(sentence)]

    def training_tagger_parent(self, slave_BiLSTM, MLPTagsParent_model, sentences, save):
        t = 0

        for i, s in enumerate(sentences):

            start = time.time()
            parser = DatasetParser(s, self.__word2vec)
            s2v = parser.sentence2vec()

            lstm = slave_BiLSTM.get_model().predict(np.array([s2v]))[0]

            parents = []
            heads = []
            tails = []
            targets = []

            for c in self.__sentence2couplesParent(s):
                parents.append(lstm[c[0]])
                heads.append(lstm[c[1]])
                tails.append(lstm[c[2]])
                targets.append(self.__tag2targetMLP(c[3]))

            parents = np.array(parents)
            heads = np.array(heads)
            tails = np.array(tails)
            targets = np.array(targets)

            print(i, MLPTagsParent_model.get_model().train_on_batch([parents, heads, tails], targets))
            t += (time.time() - start)
            print(t)

            if i % 100 == 99:
                MLPTagsParent_model.get_model().save_weights(save)
                print('Saved!')

        MLPTagsParent_model.get_model().save_weights(save)

    def test_tagger_parent(self, slave_BiLSTM, MLPTagsParent_model, in_file, out_file):
        sentences = DataUtils.conll2lists(in_file)

        for i, s in enumerate(sentences):
            parser = DatasetParser(s, self.__word2vec)
            s2v = parser.sentence2vec()
            dep = [d for d in self.__sentence2couplesParent(s) if d[3] != '_']
            lstm = slave_BiLSTM.get_model().predict(np.array([s2v]))[0]

            for j, d in enumerate(dep):
                target = MLPTagsParent_model.get_model().predict([np.reshape(lstm[d[0]], (1, 400)),
                                                                  np.reshape(lstm[d[1]], (1, 400)),
                                                                  np.reshape(lstm[d[2]], (1, 400))])
                tag = self.__target2tagMLP(target)
                if tag == 'root' and j != len(dep) - 1:
                    tag = self.__target2tagMLP(target, True)
                elif j == len(dep) - 1:
                    tag = 'root'

                s[d[1] - 1][7] = tag
            print(i, '/', len(sentences))

        DataUtils.save_corpus(sentences, out_file)

    def training_tagger_glove(self, MLPTagsGlove_model, sentences, save):
        t = 0

        for i, s in enumerate(sentences):

            start = time.time()
            parser = DatasetParser(s, self.__word2vec)
            s2v = parser.sentence2vec()

            heads = []
            tails = []
            targets = []

            for c in self.__sentence2couples(s):
                heads.append(s2v[c[0]])
                tails.append(s2v[c[1]])
                targets.append(self.__tag2targetMLP(c[2]))

            heads = np.array(heads)
            tails = np.array(tails)
            targets = np.array(targets)

            print(i, MLPTagsGlove_model.get_model().train_on_batch([heads, tails], targets))
            t += (time.time() - start)
            print(t)

            if i % 100 == 99:
                MLPTagsGlove_model.get_model().save_weights(save)
                print('Saved!')

        MLPTagsGlove_model.get_model().save_weights(save)

    def test_tagger_glove(self, MLPTagsGlove_model, in_file, out_file):
        sentences = DataUtils.conll2lists(in_file)

        for i, s in enumerate(sentences):
            parser = DatasetParser(s, self.__word2vec)
            s2v = parser.sentence2vec()
            dep = self.__sentence2couples(s)

            for j, d in enumerate(dep):
                target = MLPTagsGlove_model.get_model().predict([np.reshape(s2v[d[0]], (1, 300)),
                                                                 np.reshape(s2v[d[1]], (1, 300))])
                tag = self.__target2tagMLP(target)
                if tag == 'root' and j != len(dep) - 1:
                    tag = self.__target2tagMLP(target, True)
                elif j == len(dep) - 1:
                    tag = 'root'

                s[d[1] - 1][7] = tag
            print(i, '/', len(sentences))

        DataUtils.save_corpus(sentences, out_file)
