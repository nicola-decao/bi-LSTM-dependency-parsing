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

        lstm = Bidirectional(LSTM(input_dim=300, output_dim=hidden_LSTM, return_sequences=True), merge_mode='concat')(input)

        stack1 = Lambda(lambda x: tf.boolean_mask(x, ms1))(lstm)
        stack2 = Lambda(lambda x: tf.boolean_mask(x, ms2))(lstm)
        buffer = Lambda(lambda x: tf.boolean_mask(x, mb))(lstm)

        input_MLP = merge([stack1, stack2, buffer], mode='concat')

        h0 = Dense(input_dim=hidden_LSTM * 6, output_dim=hidden_MLP, activation='tanh')(input_MLP)
        output = Dense(input_dim=hidden_MLP, output_dim=3, activation='softmax')(h0)

        self.__model = Model(input=[input, ms1, ms2, mb], output=output)

        self.__model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def get_model(self):
        return self.__model

# class BiLSTM2:
#     def __init__(self, hidden_LSTM, hidden_MLP):
#         input = Input(shape=(None, 300), name='sentence2vec')
#         indexes = Input(shape=(None,), dtype=tf.int32, name='indexes')
#
#         lstm = Bidirectional(LSTM(input_dim=300, output_dim=hidden_LSTM, return_sequences=True),
#                              merge_mode='concat')(input)
#
#         d = Lambda(lambda x: tf.gather(x, indexes))(lstm)
#         k = Lambda(lambda x: tf.reshape(x, shape=(None, None, None, 1200)))
#         print k
#         input_MLP = Flatten()(d)
#
#         h0 = Dense(input_dim=hidden_LSTM * 6, output_dim=hidden_MLP, activation='tanh')(input_MLP)
#         output = Dense(input_dim=hidden_MLP, output_dim=3, activation='softmax')(h0)
#
#         self.__model = Model(input=[input, ms1, ms2, mb], output=output)
#
#         self.__model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     def get_model(self):
#         return self.__model