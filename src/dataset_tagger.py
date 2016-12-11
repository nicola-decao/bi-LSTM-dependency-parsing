# import re
# import numpy as np
#
# from data_utils import DataUtils
#
#
# class DatasetTagger:
#
#     @staticmethod
#     def replace_numbers(in_file, out_file):
#         l = []
#         with open(in_file, 'r') as f:
#             for line in f:
#                 line = line.split()
#                 if line[7] == 'num' or line[7] == 'number':
#                     line[1] = '<NUM>'
#                 l.append('\t'.join(line))
#
#         with open(out_file, 'w') as f:
#             f.writelines(l)
#
#     @staticmethod
#     def add_num_glove(glove):
#         vec_sum = np.zeros(300)
#         regex = re.compile('^[^A-Za-z]*[0-9][^A-Za-z]*$')
#
#         for word in glove:
#             if re.match(regex, word) is not None:
#                 vec_sum += glove[word]
#
#         glove['<NUM>'] = vec_sum / np.linalg.norm(vec_sum)
#
#     # @staticmethod
#     # def tag_sentence_dataset(in_file, out_file, check_list):
#     #     for line in in_file:
#     #         words = line.split()
#     #         out_line = ""
#     #         for word in words:
#     #             out_word = word
#     #             for (check, tag) in check_list:
#     #                 if check(word):
#     #                     out_word = tag
#     #                     break
#     #             out_line += out_word + ' '
#     #         out_file.write(out_line + '\n')
#     #
#     # def tag_dependency_dataset(self, in_file, out_file):
#     #     tag_list = [(self.isNumber_conll_row, '<NUM>')]  # , (self.isUnknown, '<UNK>')]
#     #     for line in in_file:
#     #         editted_line = line
#     #         if line.strip():
#     #             splitted = line.split()
#     #             for (check, tag) in tag_list:
#     #                 #             print line
#     #                 if check(splitted):
#     #                     word_split = line.split(splitted[1])
#     #                     editted_line = word_split[0] + tag + " ".join(word_split[1:])
#     #                     break
#     #         out_file.write(editted_line)
#     # #
#     # def isNumber(self, word):
#     #     return re.match(self.__number_regex, word) is not None
#     #
#     # def isNumber_row(self, row):
#     #     word = row[1]
#     #     return re.match(self.__number_regex, word) is not None
#     #
#     # def isNumber_conll_row(self, row):
#     #     return self.isNumber_conll((row[1],row[7]))
#     #
#     # @staticmethod
#     # def isNumber_conll(x):
#     #     if x[1] == 'num' or x[1] == 'number':
#     #         return True
#     #
#     # # def isNumber(self, word):
#     # #     return re.match(self.__number_regex, word) is not None
#     #
#     # def isUnknown(self, word):
#     #     return word not in self.__words_train
#
#     # def add_number_vec(self, word_vec_dict):
#     #     vec_sum=np.zeros(300)
#     #     for n in word_vec_dict:
#     #         if self.isNumber(n):
#     #             vec_sum += word_vec_dict[n]
#     #     word_vec_dict['<NUM>'] = vec_sum/np.linalg.norm(vec_sum)
#     #     return word_vec_dict
#     #
#     # def __init__(self, words_train=None, filename=None):
#     #     self.__number_regex = re.compile('^[^A-Za-z]*[0-9][^A-Za-z]*$')
#     #     if not words_train:
#     #         if filename is not None:
#     #             self.__words_train = DataUtils.conll2wordset(filename)
#     #             # word_tag_lst = DataUtils.conll2word_tag(filename)
#     #             # self.__numbers = map(lambda x: x[0], filter(DatasetTagger.isNumber_conll, word_tag_lst))
#     #     else:
#     #         self.__words_train=words_train
