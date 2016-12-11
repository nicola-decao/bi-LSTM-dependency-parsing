from src.BiLSTM import BiLSTM
from src.data_utils import DataUtils
from src.dataset_parser import DatasetParser
import src.paths as paths
import numpy as np
import time


def build_dataset(train_dependency_dataset, test_dependency_dataset, glove_file):
    train_file = train_dependency_dataset
    train_file_tagged = paths.PATH_TRAIN_STANFORD_TAGGED
    test_file = test_dependency_dataset
    test_file_tagged = paths.PATH_TEST_STANFORD_TAGGED

    DataUtils.replace_numbers(train_file, train_file_tagged)
    DataUtils.replace_numbers(test_file, test_file_tagged)

    glove = DataUtils.load_glove_dict(glove_file)
    DataUtils.add_num_glove(glove)
    DataUtils.save_glove_dict(glove, paths.PATH_GLOVE_FILTERED_TAGGED)


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


def target2tag(target):
    return np.array(['reduce-left', 'shift', 'reduce-right'])[np.argsort(target[0])]


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
        tagx = tag + '-unknown' if tag != 'shift' else tag
        tags.append(tagx)
        parser.next(tagx)

    return tags


def test(biLSTM, sentences, word2vec, out_file):
    actions = [predict(biLSTM, s, word2vec) for s in sentences]
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
            biLSTM.get_model().save_weights('BiLSTM.h5f')
            print ('Saved!')


# BUILD GLOVE TAGGED
# build_dataset(paths.PATH_TRAIN_STANFORD_RAW,paths.PATH_TEST_STANFORD_RAW, paths.PATH_GLOVE_FILTERED)

biLSTM = BiLSTM(200, 200)
biLSTM.get_model().load_weights('BiLSTM.h5f')
word2vec = DataUtils.load_glove_dict(paths.PATH_GLOVE_FILTERED_TAGGED)
sentences = DataUtils.conll2lists(paths.PATH_TRAIN_STANFORD_TAGGED)[11600:]

# TRAINING
training(biLSTM, word2vec, sentences)

# TESTING
# predict(biLSTM, sentence, word2vec)
# test(biLSTM, sentences, word2vec, 'fuck')

