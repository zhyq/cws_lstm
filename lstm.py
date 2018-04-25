import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from tensorflow.contrib import rnn
import numpy as np
class LSTM(object):
    '''
    For Chinese word segmentation.
    '''
    def __init__(self, model_path="model/bi-lstm.ckpt"):
        # ##################### config ######################
        self.decay = 0.85
        self.max_epoch = 5
        self.max_max_epoch = 10
        self.max_len = 32           # 句子长度
        self.timestep_size = self.max_len
        self.vocab_size = 5159    # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
        self.input_size = 64
        self.embedding_size = self.input_size       # 字向量长度
        self.class_num = 5
        self.hidden_size = 128    # 隐含层节点数
        self.layer_num = 2        # bi-lstm 层数
        self.max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）

        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
        self.model_path = model_path  # 模型保存位置


        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)


        with tf.variable_scope('Inputs'):
            self.X_inputs = tf.placeholder(tf.int32, [None, self.timestep_size], name='X_input')
            self.y_inputs = tf.placeholder(tf.int32, [None, self.timestep_size], name='y_input')

        self.bilstm_output = self.bi_lstm(self.X_inputs)

        with tf.variable_scope('outputs'):
            self.softmax_w = self.weight_variable([self.hidden_size * 2, self.class_num])
            self.softmax_b = self.bias_variable([self.class_num])
            self.y_pred = tf.matmul(self.bilstm_output, self.softmax_w) + self.softmax_b

        # adding extra statistics to monitor
        # y_inputs.shape = [batch_size, timestep_size]
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_pred, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(self.y_inputs, [-1]), logits = self.y_pred))

        # ***** 优化求解 *******
        self.tvars = tf.trainable_variables()  # 获取模型的所有参数
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, self.tvars), self.max_grad_norm)  # 获取损失函数对于每个参数的梯度
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)   # 优化器

        # 梯度下降计算
        self.train_op = self.optimizer.apply_gradients( zip(self.grads, self.tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        print ('Finished creating the bi-lstm model.')


    def weight_variable(self,shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def lstm_cell(self):
        cell = rnn.LSTMCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)
        return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def bi_lstm(self,X_inputs):
        """build the bi-LSTMs network. Return the y_pred"""
        # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)

        # ** 1.构建前向后向多层 LSTM
        cell_fw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)
        cell_bw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)

        # ** 2.初始状态
        initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

        # ** 3. bi-lstm 计算（展开）
        with tf.variable_scope('bidirectional_rnn'):
            # *** 下面，两个网络是分别计算 output 和 state
            # Forward direction
            outputs_fw = list()
            state_fw = initial_state_fw
            with tf.variable_scope('fw'):
                for timestep in range(self.timestep_size):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                    outputs_fw.append(output_fw)

            # backward direction
            outputs_bw = list()
            state_bw = initial_state_bw
            with tf.variable_scope('bw') as bw_scope:
                inputs = tf.reverse(inputs, [1])
                for timestep in range(self.timestep_size):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                    outputs_bw.append(output_bw)
            # *** 然后把 output_bw 在 timestep 维度进行翻转
            # outputs_bw.shape = [timestep_size, batch_size, hidden_size]
            outputs_bw = tf.reverse(outputs_bw, [0])
            # 把两个oupputs 拼成 [timestep_size, batch_size, hidden_size*2]
            output = tf.concat([outputs_fw, outputs_bw], 2)
            output = tf.transpose(output, perm=[1,0,2])
            output = tf.reshape(output, [-1, self.hidden_size*2])
        # ***********************************************************
        return output # [-1, hidden_size*2]
