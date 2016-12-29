import numpy as np
import time

from src.data_utils import DataUtils
from src.dataset_parser import DatasetParser


class TrainTest:
    __ACTIONS = ['reduce-left-abbrev', 'reduce-left-acomp', 'reduce-left-advcl', 'reduce-left-advmod',
                 'reduce-left-amod', 'reduce-left-appos', 'reduce-left-attr', 'reduce-left-aux', 'reduce-left-auxpass',
                 'reduce-left-cc', 'ccomp', 'reduce-left-complm', 'reduce-left-conj', 'reduce-left-cop',
                 'reduce-left-csubj', 'reduce-left-csubjpass', 'reduce-left-dep', 'reduce-left-det', 'reduce-left-dobj',
                 'reduce-left-expl', 'infmod', 'reduce-left-iobj', 'reduce-left-mark', 'reduce-left-mwe',
                 'reduce-left-neg', 'reduce-left-nn', 'reduce-left-npadvmod', 'reduce-left-nsubj',
                 'reduce-left-nsubjpass', 'reduce-left-num', 'number', 'reduce-left-parataxis', 'reduce-left-partmod',
                 'reduce-left-pcomp', 'reduce-left-pobj', 'reduce-left-poss', 'reduce-left-possessive',
                 'reduce-left-preconj', 'reduce-left-predet', 'prep', 'reduce-left-prt', 'reduce-left-punct',
                 'reduce-left-purpcl', 'reduce-left-quantmod', 'reduce-left-rcmod', 'reduce-left-rel',
                 'reduce-left-root', 'reduce-left-tmod', 'reduce-left-xcomp', 'shift', 'reduce-right-abbrev',
                 'reduce-right-acomp', 'reduce-right-advcl', 'reduce-right-advmod', 'reduce-right-amod',
                 'reduce-right-appos', 'reduce-right-attr', 'reduce-right-aux', 'reduce-right-auxpass',
                 'reduce-right-cc', 'ccomp', 'reduce-right-complm', 'reduce-right-conj', 'reduce-right-cop',
                 'reduce-right-csubj', 'reduce-right-csubjpass', 'reduce-right-dep', 'reduce-right-det',
                 'reduce-right-dobj', 'reduce-right-expl', 'infmod', 'reduce-right-iobj', 'reduce-right-mark',
                 'reduce-right-mwe', 'reduce-right-neg', 'reduce-right-nn', 'reduce-right-npadvmod',
                 'reduce-right-nsubj', 'reduce-right-nsubjpass', 'reduce-right-num', 'number', 'reduce-right-parataxis',
                 'reduce-right-partmod', 'reduce-right-pcomp', 'reduce-right-pobj', 'reduce-right-poss',
                 'reduce-right-possessive', 'reduce-right-preconj', 'reduce-right-predet', 'prep', 'reduce-right-prt',
                 'reduce-right-punct', 'reduce-right-purpcl', 'reduce-right-quantmod', 'reduce-right-rcmod',
                 'reduce-right-rel', 'reduce-right-root', 'reduce-right-tmod', 'reduce-right-xcomp']

    __LABELS = ['abbrev', 'acomp', 'advcl', 'advmod', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'cc', 'ccomp',
                'complm', 'conj', 'cop', 'csubj', 'csubjpass', 'dep', 'det', 'dobj', 'expl', 'infmod', 'iobj', 'mark',
                'mwe', 'neg', 'nn', 'npadvmod', 'nsubj', 'nsubjpass', 'num', 'number', 'parataxis', 'partmod', 'pcomp',
                'pobj', 'poss', 'possessive', 'preconj', 'predet', 'prep', 'prt', 'punct', 'purpcl', 'quantmod',
                'rcmod', 'rel', 'root', 'tmod', 'xcomp']

    __TRANSITIONS = ['reduce-left', 'shift', 'reduce-right']

    __ND_ACTIONS = np.array(__ACTIONS)
    __ND_LABELS = np.array(__LABELS)
    __ND_TRANSITIONS = np.array(__TRANSITIONS)

    def __init__(self, word2vec):
        self.__word2vec = word2vec

    def __action2onehot(self, action):
        target = np.zeros(99)
        target[self.__ACTIONS.index(action)] = 1

        return target

    def __onehot2action(self, target):
        return self.__ND_ACTIONS[np.argmax(target[0])]

    def __transition2onehot(self, tag):
        a = tag.split('-')
        a = a[0] + '-' + a[1] if len(a) > 1 else a[0]

        target = np.zeros(3)
        target[self.__TRANSITIONS.index(a)] = 1
        return target

    def __onehot2transition(self, target):
        return self.__ND_TRANSITIONS[np.argsort(target[0])][::-1]

    def __label2onehot(self, label):
        target = np.zeros(49)
        target[self.__LABELS.index(label)] = 1
        return target

    def __onehot2label(self, target, second=False):
        return self.__ND_LABELS[np.argsort(target[0])][-2] if second else self.__ND_LABELS[np.argmax(target[0])]

    def __training_template(self, model, sentences, save, f, **kwargs):
        t = 0

        for i, s in enumerate(sentences):

            start = time.time()
            parser = DatasetParser(s, self.__word2vec)
            s2v = parser.sentence2vec()

            x, y = f(parser, s2v, **kwargs)

            print(i, model.get_model().train_on_batch(x, y))
            t += (time.time() - start)
            print(t)

            if i % 100 == 99:
                model.get_model().save_weights(save)
                print('Saved!')

        model.get_model().save_weights(save)
        print('Saved!')

    def __f_transitions(self, parser, s2v):
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
            target = self.__transition2onehot(tag)
            targets.append(target)

        ms1l = np.array(ms1l)
        ms2l = np.array(ms2l)
        mbl = np.array(mbl)
        targets = np.array(targets)

        return [np.array([s2v] * ms1l.shape[0]), ms1l, ms2l, mbl], targets

    def training_transitions(self, biLSTM, sentences, save):
        self.__training_template(biLSTM, sentences, save, self.__f_transitions)

    def __f_complete(self, parser, s2v):
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

        return [np.array([s2v] * ms1l.shape[0]), ms1l, ms2l, mbl], targets

    def training_complete(self, comlpete_biLSTM, sentences, save):
        self.__training_template(comlpete_biLSTM, sentences, save, self.__f_complete)

    def __f_labeler(self, parser, s2v, slave):
        lstm = slave.get_model().predict(np.array([s2v]))[0]

        heads = []
        tails = []
        targets = []

        for c in parser.sentence2couples():
            heads.append(lstm[c[0]])
            tails.append(lstm[c[1]])
            targets.append(self.__label2onehot(c[2]))

        heads = np.array(heads)
        tails = np.array(tails)
        targets = np.array(targets)

        return [heads, tails], targets

    def training_labeler(self, slave_BiLSTM, MLPTags_model, sentences, save):
        self.__training_template(MLPTags_model, sentences, save, self.__f_labeler, slave=slave_BiLSTM)

    def __f_labeler_parent(self, parser, s2v, slave):
        lstm = slave.get_model().predict(np.array([s2v]))[0]

        parents = []
        heads = []
        tails = []
        targets = []

        for c in parser.sentence2couples_parent():
            parents.append(lstm[c[0]])
            heads.append(lstm[c[1]])
            tails.append(lstm[c[2]])
            targets.append(self.__label2onehot(c[3]))

        parents = np.array(parents)
        heads = np.array(heads)
        tails = np.array(tails)
        targets = np.array(targets)

        return [parents, heads, tails], targets

    def training_labeler_parent(self, slave_BiLSTM, MLPTagsParent_model, sentences, save):
        self.__training_template(MLPTagsParent_model, sentences, save, self.__f_labeler_parent, slave=slave_BiLSTM)

    def __f_labeler_glove(self, parser, s2v):
        heads = []
        tails = []
        targets = []

        for c in parser.sentence2couples():
            heads.append(s2v[c[0]])
            tails.append(s2v[c[1]])
            targets.append(self.__label2onehot(c[2]))

        heads = np.array(heads)
        tails = np.array(tails)
        targets = np.array(targets)

        return [heads, tails], targets

    def training_labeler_glove(self, MLPTagsGlove_model, sentences, save):
        self.__training_template(MLPTagsGlove_model, sentences, save, self.__f_labeler_glove)

    def __predict_transition(self, biLSTM, sentence):
        parser = DatasetParser(sentence, self.__word2vec)
        s2v = np.array([parser.sentence2vec()])

        tags = []
        while parser.has_next():
            (ms1, ms2), mb = parser.get_stack(), parser.get_buffer()
            ms1 = np.array([parser.index2mask(ms1)])
            ms2 = np.array([parser.index2mask(ms2)])
            mb = np.array([parser.index2mask(mb)])

            target = biLSTM.get_model().predict([s2v, ms1, ms2, mb])
            tag = self.__onehot2transition(target)
            true_tag = parser.next(tag)
            tags.append(true_tag)

        return tags

    def test_transitions(self, biLSTM, sentences, out_file):
        actions = []
        for i, s in enumerate(sentences):
            actions.append(self.__predict_transition(biLSTM, s))
            print(i + 1, '/', len(sentences))

        DataUtils.output_format_corpus(sentences, actions, out_file)

    def __test_labeler_template(self, model, in_file, out_file, dep, p, slave=None):
        sentences = DataUtils.conll2lists(in_file)

        for i, s in enumerate(sentences):
            parser = DatasetParser(s, self.__word2vec)
            s2v = parser.sentence2vec()
            dep = dep(parser)
            lstm = slave.get_model().predict(np.array([s2v]))[0] if slave else s2v

            for j, d in enumerate(dep):
                target = model.get_model().predict(p(lstm, d))

                tag = self.__onehot2label(target)
                if tag == 'root' and j != len(dep) - 1:
                    tag = self.__onehot2label(target, True)
                elif j == len(dep) - 1:
                    tag = 'root'

                s[d[1] - 1][7] = tag
            print(i, '/', len(sentences))

        DataUtils.save_corpus(sentences, out_file)

    def __dep_labeler(self, parser):
        return parser.sentence2couples()

    def __p_labeler(self, lstm, d):
        return [np.reshape(lstm[d[0]], (1, 400)), np.reshape(lstm[d[1]], (1, 400))]

    def test_labeler(self, slave_BiLSTM, MLPTags_model, in_file, out_file):
        self.__test_labeler_template(MLPTags_model, in_file, out_file, self.__dep_labeler, self.__p_labeler,
                                     slave_BiLSTM)

    def __dep_labeler_parent(self, parser):
        return [d for d in parser.sentence2couples_parent() if d[3] != '_']

    def __p_labeler_parent(self, lstm, d):
        return [np.reshape(lstm[d[0]], (1, 400)), np.reshape(lstm[d[1]], (1, 400)), np.reshape(lstm[d[2]], (1, 400))]

    def test_labeler_parent(self, slave_BiLSTM, MLPTagsParent_model, in_file, out_file):
        self.__test_labeler_template(MLPTagsParent_model, in_file, out_file, self.__dep_labeler_parent,
                                     self.__p_labeler, slave_BiLSTM)

    def __p_labeler_glove(self, s2v, d):
        return [np.reshape(s2v[d[0]], (1, 300)), np.reshape(s2v[d[1]], (1, 300))]

    def test_labeler_glove(self, MLPTagsGlove_model, in_file, out_file):
        self.__test_labeler_template(MLPTagsGlove_model, in_file, out_file, self.__dep_labeler, self.__p_labeler_glove)
