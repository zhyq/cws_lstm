import argparse
import data_helper
from sklearn.model_selection import train_test_split
import re
import lstm
from lstm import *
import time
from viterbi import Viterbi
xrange = range



def simple_cut(text,dh,lm,viterbi):
    """对一个片段text（标点符号把句子划分为多个片段）进行预测。"""
    if text:
        #print("text: %s" %text)
        text_len = len(text)
        X_batch = dh.text2ids(text)  # 这里每个 batch 是一个样本
        fetches = [lm.y_pred]
        feed_dict = {lm.X_inputs:X_batch, lm.lr:1.0, lm.batch_size:1, lm.keep_prob:1.0}
        _y_pred = sess.run(fetches, feed_dict)[0][:text_len]  # padding填充的部分直接丢弃
        nodes = [dict(zip(['s','b','m','e'], each[1:])) for each in _y_pred]
        #print(type(dh.labels))
        #print(dh.labels)

        tags = viterbi.viterbi(nodes)
        words = []
        for i in range(len(text)):
            if tags[i] in ['s', 'b']:
                words.append(text[i])
            else:
                words[-1] += text[i]
        return words
    else:
        return []


def cut_word(sentence,dh,lm,viterbi):
    """首先将一个sentence根据标点和英文符号/字符串划分成多个片段text，然后对每一个片段分词。"""
    not_cuts = re.compile(u'([0-9\da-zA-Z ]+)|[。，、？！.\.\?,!]')
    result = []
    start = 0
    for seg_sign in not_cuts.finditer(sentence):
        result.extend(simple_cut(sentence[start:seg_sign.start()],dh,lm,viterbi))
        result.append(sentence[seg_sign.start():seg_sign.end()])
        start = seg_sign.end()
    result.extend(simple_cut(sentence[start:],dh,lm,viterbi))
    return result

def predict(dh,lm,viterbi,sentence):
    # 例一# 例一
    result = cut_word(sentence,dh,lm,viterbi)
    rss = ''
    for each in result:
        rss = rss + each + ' / '
    print (rss)


def main():
    parser = argparse.ArgumentParser(description = "lstm segment args.")
    parser.add_argument("-a","--action",type=str,default="predict",help="train or predict")
    parser.add_argument("-c","--corpus",type=str,default="data/msr_train.txt",help="train file")
    parser.add_argument("-v","--vocab_model",type=str,default="model/vocab_model.pkl",help="vocab model file")
    parser.add_argument("-m","--lstm_model",type=str,default="model/bi-lstm.ckpt-6",help="lstm model file")

    args = parser.parse_args()
    corpus = args.corpus
    vocab_model = args.vocab_model
    action = args.action
    lstm_model = args.lstm_model
    dh = data_helper.DataHelper(vocab_model)
    dh.datahander(corpus)
    #dh.loadmodel(vocab_model)

    if action == "predict":
        lm = lstm.LSTM(lstm_model)
        viterbi = Viterbi(dh.labels)
        saver = tf.train.Saver()
        saver.restore(sess, lm.model_path)

        sentence = u'人们思考问题往往不是从零开始的。就好像你现在阅读这篇文章一样，你对每个词的理解都会依赖于你前面看到的一些词，而不是把你前面看的内容全部抛弃了，忘记了，再去理解这个单词。也就是说，人们的思维总是会有延续性的。'
        predict(dh,lm,viterbi,sentence)
        while True:
            sentence = input("input words for cut .EXIT for exit:\n")
            if sentence == "EXIT":
                break
            predict(dh,lm,viterbi,sentence)
if __name__ == "__main__":
    main()
