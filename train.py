import argparse
import data_helper
from sklearn.model_selection import train_test_split
import lstm
from lstm import *
import time
xrange = range
def test_epoch(dataset,lm):
    """Testing or valid."""
    _batch_size = 500
    fetches = [lm.accuracy, lm.cost]
    _y = dataset.y
    data_size = _y.shape[0]
    batch_num = int(data_size / _batch_size)
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    for i in xrange(batch_num):
        X_batch, y_batch = dataset.next_batch(_batch_size)
        feed_dict = {lm.X_inputs:X_batch, lm.y_inputs:y_batch, lm.lr:1e-5, lm.batch_size:_batch_size, lm.keep_prob:1.0}
        _acc, _cost = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost
    mean_acc= _accs / batch_num
    mean_cost = _costs / batch_num
    return mean_acc, mean_cost

def train(dh):
    X = dh.X
    y = dh.y

    # 划分测试集/训练集/验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,  test_size=0.2, random_state=42)
    print ('X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
    X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape))

    data_train = data_helper.BatchGenerator(X_train, y_train, shuffle=True)
    data_valid = data_helper.BatchGenerator(X_valid, y_valid, shuffle=False)
    data_test = data_helper.BatchGenerator(X_test, y_test, shuffle=False)
    lm = lstm.LSTM()


    sess.run(tf.global_variables_initializer())
    tr_batch_size = 128
    max_max_epoch = 6
    max_epoch = lm.max_epoch
    display_num = 5  # 每个 epoch 显示是个结果
    tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
    ### just for faster
    #tr_batch_num = int(data_train.y.shape[0] / (tr_batch_size*100))  # 每个 epoch 中包含的 batch 数
    print("tr_batch_num %s" % tr_batch_num)
    display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次
    saver = tf.train.Saver(max_to_keep=10)  # 最多保存的模型数量
    for epoch in xrange(max_max_epoch):
        _lr = 1e-4
        if epoch > max_epoch:
            _lr = _lr * ((decay) ** (epoch - max_epoch))
        print ('EPOCH %d， lr=%g' % (epoch+1, _lr))
        start_time = time.time()
        _costs = 0.0
        _accs = 0.0
        show_accs = 0.0
        show_costs = 0.0
        for batch in xrange(tr_batch_num):
            fetches = [lm.accuracy, lm.cost, lm.train_op]
            X_batch, y_batch = data_train.next_batch(tr_batch_size)
            feed_dict = {lm.X_inputs:X_batch, lm.y_inputs:y_batch, lm.lr:_lr, lm.batch_size:tr_batch_size, lm.keep_prob:0.5}
            _acc, _cost, _ = sess.run(fetches, feed_dict) # the cost is the mean cost of one batch
            _accs += _acc
            _costs += _cost
            show_accs += _acc
            show_costs += _cost
            if (batch + 1) % display_batch == 0:
                valid_acc, valid_cost = test_epoch(data_valid,lm)  # valid
                print ('\ttraining acc=%g, cost=%g;  valid acc= %g, cost=%g ' % (show_accs / display_batch,
                                                    show_costs / display_batch, valid_acc, valid_cost))
                show_accs = 0.0
                show_costs = 0.0
        mean_acc = _accs / tr_batch_num
        mean_cost = _costs / tr_batch_num
        if (epoch + 1) % 3 == 0:  # 每 3 个 epoch 保存一次模型
            save_path = saver.save(sess, lm.model_path, global_step=(epoch+1))
            print ('the save path is %s'% save_path)
        print ('\ttraining %d, acc=%g, cost=%g ' % (data_train.y.shape[0], mean_acc, mean_cost))
        print ('Epoch training %d, acc=%g, cost=%g, speed=%g s/epoch' % (data_train.y.shape[0], mean_acc, mean_cost, time.time()-start_time))
    # testing
    print ('**TEST RESULT:')
    test_acc, test_cost = test_epoch(data_test,lm)
    print ('**Test %d, acc=%g, cost=%g' % (data_test.y.shape[0], test_acc, test_cost) )

def predict(dh,lstm_model):
    lm = lstm.LSTM(lstm_model)
    saver = tf.train.Saver()
    saver.restore(sess, lm.model_path)


def main():
    parser = argparse.ArgumentParser(description = "lstm segment args.")
    parser.add_argument("-a","--action",type=str,default="train",help="train or predict")
    parser.add_argument("-c","--corpus",type=str,default="data/msr_train.txt",help="train file")
    parser.add_argument("-v","--vocab_model",type=str,default="model/vocab_model.pkl",help="vocab model file")
    parser.add_argument("-m","--lstm_model",type=str,default="model/bi-lstm.ckpt-3",help="lstm model file")

    args = parser.parse_args()
    corpus = args.corpus
    vocab_model = args.vocab_model
    action = args.action
    lstm_model = args.lstm_model
    dh = data_helper.DataHelper(vocab_model)
    dh.datahander(corpus)
    #dh.loadmodel(vocab_model)
    if action == "train":
        train(dh)

    if action == "predict":
        predict(dh,lstm_model)


if __name__ == "__main__":
    main()
