from keras.engine import Input, Model, merge
from keras.layers import LSTM, Lambda, Bidirectional, Dense, Flatten
from keras import backend as K
import tensorflow as tf


class BiLSTM:

    def __init__(self, hidden_LSTM, hidden_MLP):

        input = Input(shape=(None, 300), name='sentence2vec')
        ms1 = Input(shape=(None,), dtype=tf.bool, name='ms1')
        ms2 = Input(shape=(None,), dtype=tf.bool, name='ms2')
        mb = Input(shape=(None,), dtype=tf.bool, name='mb')

        lstm = Bidirectional(LSTM(input_dim=300, output_dim=hidden_LSTM, return_sequences=True, name='LSTM'), merge_mode='concat', name='BI')(input)

        stack1 = Lambda(lambda x: tf.boolean_mask(x, ms1), name='stack1')(lstm)
        stack2 = Lambda(lambda x: tf.boolean_mask(x, ms2), name='stack2')(lstm)
        buffer = Lambda(lambda x: tf.boolean_mask(x, mb), name='buffer')(lstm)

        input_MLP = merge([stack1, stack2, buffer], mode='concat', name='input_MLP')

        h0 = Dense(input_dim=hidden_LSTM * 6, output_dim=hidden_MLP, activation='tanh', name='h0')(input_MLP)
        output = Dense(input_dim=hidden_MLP, output_dim=3, activation='softmax', name='output')(h0)

        self.__model = Model(input=[input, ms1, ms2, mb], output=output)

        self.__model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def get_model(self):
        return self.__model


class slave_BiLSTM:

    def __init__(self, hidden_LSTM):

        input = Input(shape=(None, 300), name='sentence2vec')
        lstm = Bidirectional(LSTM(input_dim=300, output_dim=hidden_LSTM, return_sequences=True, name='LSTM'), merge_mode='concat', name='BI')(input)
        self.__model = Model(input=input, output=lstm)

    def get_model(self):
        return self.__model


class MLPTags:

    def __init__(self, hidden_LSTM, hidden_MLP1, hidden_MLP2):
        head = Input(shape=(hidden_LSTM * 2,), name='head')
        tail = Input(shape=(hidden_LSTM * 2,), name='tail')

        input_MLP = merge([head, tail], mode='concat', name='input_MLPTags')

        h0 = Dense(input_dim=hidden_LSTM * 4, output_dim=hidden_MLP1, init='glorot_normal', activation='linear', name='h0MLPTags')(input_MLP)
        h1 = Dense(input_dim=hidden_MLP1, output_dim=hidden_MLP2, init='glorot_normal', activation='tanh', name='h1MLPTags')(h0)

        output = Dense(input_dim=hidden_MLP2, output_dim=49, init='glorot_normal', activation='softmax', name='outputMLPTags')(h1)

        self.__model = Model(input=[head, tail], output=output)

        self.__model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def get_model(self):
        return self.__model

#
# class MLPTagsV2:
#     def __init__(self, hidden_LSTM, hidden_MLP1, hidden_MLP2):
#         parent = Input(shape=(hidden_LSTM * 2,), name='parent')
#         head = Input(shape=(hidden_LSTM * 2,), name='head')
#         tail = Input(shape=(hidden_LSTM * 2,), name='tail')
#
#         input_MLP = merge([parent, head, tail], mode='concat', name='input_MLPTags')
#
#         h0 = Dense(input_dim=hidden_LSTM * 6, output_dim=hidden_MLP1, init='glorot_normal', activation='linear',
#                    name='h0MLPTags')(input_MLP)
#         h1 = Dense(input_dim=hidden_MLP1, output_dim=hidden_MLP2, init='glorot_normal', activation='tanh',
#                    name='h1MLPTags')(h0)
#
#         output = Dense(input_dim=hidden_MLP2, output_dim=49, init='glorot_normal', activation='softmax',
#                        name='outputMLPTags')(h1)
#
#         self.__model = Model(input=[parent, head, tail], output=output)
#
#         self.__model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     def get_model(self):
#         return self.__model
