from src.BiLSTM import BiLSTM, slave_BiLSTM, MLPTags
from src.data_utils import DataUtils
from src.dataset_parser import DatasetParser
import src.paths as paths
import numpy as np
import time
import os


def tag2target(tag):
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


def tag2targetMLP(tag):
    target = np.zeros(49)
    for i, e in enumerate(('abbrev', 'acomp', 'advcl', 'advmod', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'cc',
                           'ccomp', 'complm', 'conj', 'cop', 'csubj', 'csubjpass', 'dep', 'det', 'dobj', 'expl',
                           'infmod', 'iobj', 'mark', 'mwe', 'neg', 'nn', 'npadvmod', 'nsubj', 'nsubjpass', 'num',
                           'number', 'parataxis', 'partmod', 'pcomp', 'pobj', 'poss', 'possessive', 'preconj', 'predet',
                           'prep', 'prt', 'punct', 'purpcl', 'quantmod', 'rcmod', 'rel', 'root', 'tmod', 'xcomp')):
        if tag == e:
            target[i] = 1
    return target


def target2tag(target):
    return np.array(['reduce-left', 'shift', 'reduce-right'])[np.argsort(target[0])][::-1]


def target2tagMLP(target):
    return np.array(
        ['abbrev', 'acomp', 'advcl', 'advmod', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'cc', 'ccomp', 'complm',
         'conj', 'cop', 'csubj', 'csubjpass', 'dep', 'det', 'dobj', 'expl', 'infmod', 'iobj', 'mark', 'mwe', 'neg',
         'nn', 'npadvmod', 'nsubj', 'nsubjpass', 'num', 'number', 'parataxis', 'partmod', 'pcomp', 'pobj', 'poss',
         'possessive', 'preconj', 'predet', 'prep', 'prt', 'punct', 'purpcl', 'quantmod', 'rcmod', 'rel', 'root',
         'tmod', 'xcomp'])[np.argmax(target[0])]


def predict(biLSTM, sentence, word2vec):
    parser = DatasetParser(sentence, word2vec)
    s2v = np.array([parser.sentence2vec()])

    tags = []
    while parser.has_next():
        (ms1, ms2), mb = parser.get_stack(), parser.get_buffer()
        ms1 = np.array([parser.index2mask(ms1)])
        ms2 = np.array([parser.index2mask(ms2)])
        mb = np.array([parser.index2mask(mb)])

        target = biLSTM.get_model().predict([s2v, ms1, ms2, mb])
        tag = target2tag(target)
        true_tag = parser.next(tag)
        tags.append(true_tag)

    return tags


def test(biLSTM, sentences, word2vec, out_file):
    actions = []
    for i, s in enumerate(sentences):
        actions.append(predict(biLSTM, s, word2vec))
        print(i + 1, '/', len(sentences))
    DataUtils.output_format_corpus(sentences, actions, out_file)


def training(biLSTM, word2vec, sentences):
    t = 0

    for i, s in enumerate(sentences):

        start = time.time()
        parser = DatasetParser(s, word2vec)
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
            target = tag2target(tag)
            targets.append(target)

        ms1l = np.array(ms1l)
        ms2l = np.array(ms2l)
        mbl = np.array(mbl)

        targets = np.array(targets)

        print(i, biLSTM.get_model().train_on_batch([np.array([s2v] * ms1l.shape[0]), ms1l, ms2l, mbl], targets))
        t += (time.time() - start)
        print(t)

        if i % 100 == 99:
            biLSTM.get_model().save_weights('BiLSTM_90.h5f')
            print('Saved!')


def training_tagger(slave_BiLSTM, MLPTags_model, sentences, word2vec):
    t = 0

    for i, s in enumerate(sentences):

        start = time.time()
        parser = DatasetParser(s, word2vec)
        s2v = parser.sentence2vec()

        lstm = slave_BiLSTM.get_model().predict(np.array([s2v]))[0]

        # parents = []
        heads = []
        tails = []
        targets = []

        for c in sentence2couples(s):
            # parents.append(lstm[c[0]])
            heads.append(lstm[c[0]])
            tails.append(lstm[c[1]])
            targets.append(tag2targetMLP(c[2]))

        # parents = np.array(parents)
        heads = np.array(heads)
        tails = np.array(tails)
        targets = np.array(targets)

        print(i, MLPTags_model.get_model().train_on_batch([heads, tails], targets))
        t += (time.time() - start)
        print(t)

        if i % 100 == 99:
            MLPTags_model.get_model().save_weights('MLPTags_model.h5f')
            print('Saved!')


def sentence2couples(sentence):
    return [[int(w[6]), i + 1, w[7]] for i, w in enumerate(sentence)]


def sentence2couplesParent(sentence):
    sentence = [['0', '<ROOT/>', '_', 'ROOT', '_', '_', '0', '_', '_', '_']] + sentence
    return [[int(sentence[int(w[6])][6]), int(w[6]), i, w[7]] for i, w in enumerate(sentence)]


# sentences = DataUtils.conll2lists(paths.PATH_TRAIN_STANFORD_TAGGED) + DataUtils.conll2lists(paths.PATH_TEST_STANFORD_TAGGED)
# tags = set(w[7] for s in sentences for w in s)
# print(list(tags))


# BUILD GLOVE TAGGED
# build_dataset(paths.PATH_TRAIN_STANFORD_RAW,paths.PATH_TEST_STANFORD_RAW, paths.PATH_GLOVE_FILTERED)

slave_BiLSTM = slave_BiLSTM(200)
slave_BiLSTM.get_model().load_weights('BiLSTM.h5f', True)
MLPTags_model = MLPTags(200, 400, 100)
MLPTags_model.get_model().load_weights('MLPTags_model.h5f')
# MLPTags_model = MLPTagsV2(200, 600, 300)
# MLPTags_model.get_model().load_weights('MLPTags_model.h5f')

biLSTM = BiLSTM(200, 200)
biLSTM.get_model().load_weights('BiLSTM.h5f')

word2vec = DataUtils.load_glove_dict(paths.PATH_GLOVE_FILTERED_TAGGED)
sentences = DataUtils.conll2lists(paths.PATH_TRAIN_STANFORD_TAGGED)

# print(len(sentences))
# TRAINING
# training(biLSTM, word2vec, sentences)
# training_tagger(slave_BiLSTM, MLPTags_model, sentences, word2vec)

# TESTING

l = []
se = DataUtils.conll2lists('../data/test-stanford-raw-tagged-num.conll')

for s in se:
    for w in s:
        if w[7] == 'num':
            w[7] = 'number'
    l.append(s)

with open('../data/test-stanford-raw-tagged-num.conll', 'w') as f:
    for s in l:
        for w in s:
            f.write('\t'.join(w) + '\n')
        f.write('\n')

#
# test_sentences = DataUtils.conll2lists('../data/test-super-num.conll')
# # test(biLSTM, test_sentences, word2vec, '../data/test-super.conll')
#
# for i, s in enumerate(test_sentences):
#     parser = DatasetParser(s, word2vec)
#     s2v = parser.sentence2vec()
#     # depp = sentence2couplesParent(s)
#     dep = sentence2couples(s)
#     lstm = slave_BiLSTM.get_model().predict(np.array([s2v]))[0]
#
#     # depp = [d for d in depp if d[3] != '_']
#
#     assert len(dep) == len(s)
#
#     for d in dep:
#         target = MLPTags_model.get_model().predict([np.reshape(lstm[d[0]], (1, 400)),
#                                                     np.reshape(lstm[d[1]], (1, 400))])
#         tag = target2tagMLP(target)
#         if tag == 'num':
#             tag = 'number'
#         s[d[1] - 1][7] = tag
#     print(i, '/', len(test_sentences))
#
# with open('../data/test-super-tagged-num.conll', 'w') as f:
#     for s in test_sentences:
#         for w in s:
#             f.write('\t'.join(w) + '\n')
#         f.write('\n')


# with open('../data/fuck-gold', 'w') as f:
#     for s in test_sentences:
#         for w in s:
#             f.write('\t'.join(w) + '\n')
#         f.write('\n')

# test(biLSTM, test_sentences, word2vec, paths.PATH_TEST_STANFORD_TAGGED, '../data/test-bilstm.conll')
