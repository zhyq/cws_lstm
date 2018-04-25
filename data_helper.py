import codecs
import numpy as np
import re
from tqdm import tqdm
import pandas as pd
from itertools import chain
import os
import pickle
class DataHelper():
    def __init__(self,model_path):
        self.max_len = 32
        self.vocab_size = None
        self.df_data = None
        self.word2id = None
        self.id2word = None
        self.tag2id = None
        self.id2tag = None
        self.X = None
        self.y = None
        self.model_path = model_path

    # if not load model ,handle data
    def datahander(self,textfile):
        texts = self.read_data(textfile)
        texts = u"".join(map(self.clean,texts))
        sentences = re.split(u'[，。！？、‘’“”]/[bems]', texts)
        datas = list()
        labels = list()
        for sentence in tqdm(iter(sentences)):
            result = self.get_Xy(sentence)
            if result:
                datas.append(result[0])
                labels.append(result[1])

        df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
        #　句子长度
        df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))
        self.trance(df_data)
        df_data['X'] = df_data['words'].apply(self.X_padding)
        df_data['y'] = df_data['tags'].apply(self.y_padding)
        self.df_data = df_data
        X = np.asarray(list(df_data['X'].values))
        y = np.asarray(list(df_data['y'].values))
        self.X = X
        self.y = y
        self.labels = labels
        self.savemodel(self.model_path)

    def savemodel(self,model_path):
        with open(model_path, 'wb') as outp:
            pickle.dump(self.X, outp)
            pickle.dump(self.y, outp)
            pickle.dump(self.word2id, outp)
            pickle.dump(self.id2word, outp)
            pickle.dump(self.tag2id, outp)
            pickle.dump(self.id2tag, outp)
            pickle.dump(self.labels, outp)
            print ('** Finished saving the data.')

    # if saved model,just load model
    def loadmodel(self,model_path):
        with open(model_path, 'rb') as inp:
            self.X = pickle.load(inp)
            self.y = pickle.load(inp)
            self.word2id = pickle.load(inp)
            self.id2word = pickle.load(inp)
            self.tag2id = pickle.load(inp)
            self.id2tag = pickle.load(inp)
            self.labels = pickle.load(inp)

    def clean(self,s):
        if u'“/s' not in s:  # 句子中间的引号不应去掉
            return s.replace(u' ”/s', '')
        elif u'”/s' not in s:
            return s.replace(u'“/s ', '')
        elif u'‘/s' not in s:
            return s.replace(u' ’/s', '')
        elif u'’/s' not in s:
            return s.replace(u'‘/s ', '')
        else:
            return s

    def read_data(self,textfile):
        return [line for line in codecs.open(textfile,'r','gbk').readlines()]
        #with open(textfile, 'rb') as inp:
        #    texts = inp.read().decode('gbk')
        #    sentences = texts.split('\r\n')  # 根据换行切分
        #return sentences

    def get_Xy(self,sentence):
        """将 sentence 处理成 [word1, w2, ..wn], [tag1, t2, ...tn]"""
        words_tags = re.findall('(.)/(.)', sentence)
        if words_tags:
            words_tags = np.asarray(words_tags)
            words = words_tags[:, 0]
            tags = words_tags[:, 1]
            return words, tags # 所有的字和tag分别存为 data / label

    def trance(self,df_data):
        # 1.用 chain(*lists) 函数把多个list拼接起来
        all_words = list(chain(*df_data['words'].values))
        # 2.统计所有 word

        sr_allwords = pd.Series(all_words)
        sr_allwords = sr_allwords.value_counts()
        set_words = sr_allwords.index
        self.vocab_size = len(set_words)
        set_ids = range(1, len(set_words)+1) # 注意从1开始，因为我们准备把0作为填充值
        tags = [ 'x', 's', 'b', 'm', 'e']
        tag_ids = range(len(tags))

        # 3. 构建 words 和 tags 都转为数值 id 的映射（使用 Series 比 dict 更加方便）
        word2id = pd.Series(set_ids, index=set_words)
        id2word = pd.Series(set_words, index=set_ids)
        tag2id = pd.Series(tag_ids, index=tags)
        id2tag = pd.Series(tags, index=tag_ids)
        self.word2id = word2id
        self.id2word = id2word
        self.tag2id = tag2id
        self.id2tag = id2tag


    def X_padding(self,words):
        """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
        max_len = self.max_len
        word2id = self.word2id
        ids = list(word2id[words])
        if len(ids) >= max_len:  # 长则弃掉
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) # 短则补全
        return ids

    def y_padding(self,tags):
        """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
        max_len = self.max_len
        tag2id = self.tag2id
        ids = list(tag2id[tags])
        if len(ids) >= max_len:  # 长则弃掉
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) # 短则补全
        return ids

    def text2ids(self,text):
        """把字片段text转为 ids."""
        words = list(text)
        max_len = self.max_len
        ids = list(self.word2id[words])
        if len(ids) >= max_len:  # 长则弃掉
            print ("输出片段超过%d部分无法处理" % (max_len))
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) # 短则补全
        ids = np.asarray(ids).reshape([-1, max_len])
        return ids

class BatchGenerator(object):
    """ Construct a Data generator. The input X, y should be ndarray or list like type.

    Example:
        Data_train = BatchGenerator(X=X_train_all, y=y_train_all, shuffle=False)
        Data_test = BatchGenerator(X=X_test_all, y=y_test_all, shuffle=False)
        X = Data_train.X
        y = Data_train.y
        or:
        X_batch, y_batch = Data_train.next_batch(batch_size)
     """

    def __init__(self, X, y, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._y = self._y[new_index]

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def num_examples(self):
        return self._number_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]
