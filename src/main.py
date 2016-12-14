from src.BiLSTM import BiLSTM, SlaveBiLSTM, MLPTags, MLPTagsParent, MLPTagsGlove, CompleteBiLSTM
from src.data_utils import DataUtils
from src.train_test import TrainTest
import src.paths as paths
import os


def num_coversion(filename):
    l = []
    se = DataUtils.conll2lists(filename)

    for s in se:
        for w in s:
            if w[7] == 'num':
                w[7] = 'number'
        l.append(s)

    DataUtils.save_corpus(l, filename)


# BUILD GLOVE TAGGED
# build_dataset(paths.PATH_TRAIN_STANFORD_RAW,paths.PATH_TEST_STANFORD_RAW, paths.PATH_GLOVE_FILTERED)

# CREATE MODELS
biLSTM = BiLSTM(200, 200)
biLSTM.get_model().load_weights('BiLSTM.h5f')

slave_BiLSTM = SlaveBiLSTM(200)
slave_BiLSTM.get_model().load_weights('BiLSTM.h5f', True)

MLPTags_model = MLPTags(200, 400, 100)
MLPTags_model.get_model().load_weights('MLPTags_model.h5f')

MLPTagsParent_model = MLPTagsParent(200, 400, 100)
MLPTagsParent_model.get_model().load_weights('MLPTagsParent_model.h5f')

MLPTagsGlove_model = MLPTagsGlove(400, 100)
MLPTagsGlove_model.get_model().load_weights('MLPTagsGlove_model.h5f')

comlpete_biLSTM = CompleteBiLSTM(200, 200)
# CompleteBiLSTM.get_model().load_weights('comlpete_biLSTM.h5f')

# LOAD DATASETS
word2vec = DataUtils.load_glove_dict(paths.PATH_GLOVE_FILTERED_TAGGED)
train_sentences = DataUtils.conll2lists(paths.PATH_TRAIN_STANFORD_TAGGED)
test_sentences = DataUtils.conll2lists(paths.PATH_TEST_STANFORD_TAGGED)

tt = TrainTest(word2vec)
epochs = 3

# TRAINING
for _ in range(epochs):
    # tt.training(biLSTM, train_sentences, save='BiLSTM.h5f')
    # tt.training_tagger(slave_BiLSTM, MLPTags_model, train_sentences, save='MLPTags_model.h5f')
    # tt.training_tagger_parent(slave_BiLSTM, MLPTagsParent_model, train_sentences, save='MLPTagsParent_model.h5f')
    # tt.training_tagger_glove(MLPTagsGlove_model, train_sentences, save='MLPTagsGlove_model.h5f')
    tt.training_complete(comlpete_biLSTM, train_sentences, 'comlpete_biLSTM.h5f')

# TESTING
# tt.test(biLSTM, test_sentences, '../data/results-arcs.conll')
# tt.test_tagger(slave_BiLSTM, MLPTags_model, '../data/results-arcs.conll', '../data/results-arcs-labeled.conll')
# tt.test_tagger_parent(slave_BiLSTM, MLPTagsParent_model, '../data/results-arcs.conll', '../data/results-arcs-labeled-parent.conll')
# tt.test_tagger_glove(MLPTagsGlove_model, '../data/results-arcs.conll', '../data/results-arcs-labeled-glove.conll')

# os.system('perl ../data/eval.pl -g {} -s {} -q'.format(paths.PATH_TEST_STANFORD_TAGGED, '../data/results-arcs-labeled-glove.conll'))

# num_coversion('../data/test-stanford-raw-tagged-num.conll')
# num_coversion('../data/results-arcs-labeled-num.conll')
