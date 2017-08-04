#-*-coding:UTF-8 -*-
import tensorflow as tf
import numpy as np
import input_data


TIME_STEPS = 4                  # 设置 LSTM 的时间步
BATCH_SIZE = 128                # 设置随机梯度下降的 BATCH 大小
INPUT_SIZE = 8                  # 输入向量维数 
OUTPUT_SIZE = 8                 # 输出向量维数
CELL_SIZE = 128                 # 隐藏单元节点数
LEARNING_RATE_BASE = 0.00056    # learning rate base
LEARNING_RATE_DECAY = 0.00009   # learning rate decay

# 可视化后的汇总数据的存放地址
LOG_DIR = '/Users/bobbycho/DeepTraffic/traffic_with_summaries'
# 模型存放地址和模型名
MODEL_SAVE_PATH = "/Users/bobbycho/DeepTraffic/model/"
MODEL_NAME = "model.ckpt"


# 获取训练集和测试集数据
def get_data(train):
    X_train, y_train, X_test, y_test = input_data.split_dataset()
    #train_batch_X = X_train[(step-1)*batch_size:step*batch_size]
    #train_batch_y = y_train[(step-1)*batch_size:step*batch_size]
    if train:
        #data_X = X_train[(step-1)*batch_size:step*batch_size]
        #data_y = y_train[(step-1)*batch_size:step*batch_size]
        return X_train, y_train
    else:
        #data_X, data_y = X_test, y_test 
        return X_test, y_test 


def next_batch(data, step, batch_size):
    data_batch = data[(step-1)*batch_size:step*batch_size]
    return data_batch


# 定义 LSTMRNN 的主体结构
class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.global_step = tf.Variable(0, trainable=False)
        # 指数衰减学习率
        self.learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            self.global_step,
            10364/BATCH_SIZE, LEARNING_RATE_DECAY)

        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('accuracy'):
            self.evaluate_model()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost, global_step=self.global_step)

    # 输入层计算
    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='input_layer_x')
    #   # Ws(in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        self.variable_summaries(Ws_in)
        # bs(cell_size)
        bs_in = self._bias_variable([self.cell_size,])
        self.variable_summaries(bs_in)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
            tf.summary.histogram('pre_activations', l_in_y)
        activations = tf.nn.sigmoid(l_in_y, name='activation')
        tf.summary.histogram('activations', activations)
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='input_layer_y')

    # LSTM 单元计算
    def add_cell(self): 
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True) 
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn( lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False) 
    
    #  输出层计算
    def add_output_layer(self):
        # shape = (batch*steps, cell_size)
#       l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        l_out_x = tf.unstack(tf.transpose(self.cell_outputs, [1, 0, 2]))
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        #shape = (batch*steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x[-1], Ws_out) + bs_out


    # 定义损失函数
    def compute_cost(self):
        self.cost= tf.reduce_sum(-self.ys*tf.log(tf.clip_by_value(self.pred, 1e-10, 1.0)))
        tf.summary.scalar('cost', self.cost)


    # 计算迭代准确率
    def evaluate_model(self):
        normalize_pred = tf.nn.l2_normalize(tf.clip_by_value(self.pred, 1e-10, 1.0), dim=1)
        normalize_y = tf.nn.l2_normalize(self.ys, dim=1)
        cosine_similarity = tf.matmul(normalize_pred, tf.transpose(normalize_y, [1, 0]))
        self.accuracy = tf.reduce_mean(cosine_similarity)
        tf.summary.scalar('accuracy', self.accuracy)


    # 权重和偏置初始化 
    def _weight_variable(self, shape, name='weights'):
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(shape=shape, initializer=initializer, name=name)


    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


    # 数据汇总以便可视化 
    def variable_summaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

if __name__ == '__main__':
    # 搭建 LSTMRNN 模型
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE) 
    # 数据汇总
    merged = tf.summary.merge_all()
    # 保存模型
    saver = tf.train.Saver()
    
    sess = tf.Session()
    # 把每一次训练的模型数据写入到LOG_DIR相应的子目录下
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train5', sess.graph)
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test')

    sess.run(tf.global_variables_initializer())
    i = 1
    while i * BATCH_SIZE < 20000:
        # 每隔10次迭代计算测试集上的准确率
        if i % 10 == 0:
            xs, ys = get_data(False)
            feed_dict = {
                model.xs: xs,
                model.ys: ys
            }
            summary, acc, step = sess.run([merged, model.accuracy, model.global_step], feed_dict=feed_dict)
            test_writer.add_summary(summary, i)
            print "Iter " + str(i*BATCH_SIZE) + ", Test Accuracy= " + \
                "{:.6f}".format(acc)
        # 训练模型
        else:
            xs, ys = get_data(True)
            if i == 1:
                feed_dict = {
                    model.xs: next_batch(xs, i, BATCH_SIZE),
                    model.ys: next_batch(ys, i, BATCH_SIZE)
                }
            else:
                feed_dict = {
                    model.xs: next_batch(xs, i, BATCH_SIZE),
                    model.ys: next_batch(ys, i, BATCH_SIZE),
                    model.cell_init_state: state 
                }
            summary, _, loss, acc, state, pred, step = sess.run(
                [merged, model.train_op, model.cost, model.accuracy,
                model.cell_final_state, model.pred, model.global_step], feed_dict=feed_dict)
            train_writer.add_summary(summary, i)
            print "Iter " + str(i*BATCH_SIZE) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                "{:.6f}".format(acc)
        # 每训练1000个样本保存一次模型
        if (i*BATCH_SIZE % 1000 == 0):
            saver.save(sess, os.path.join(
                MODEL_SAVE_PATH, MODEL_NAME), global_step=model.global_step)
        i += 1
    print "Training finished!"

    # save model 
    # restore model
    #sess.run(tf.global_variables_initializer())
    #saver.restore(sess, save_path)
    train_writer.close()
    test_writer.close()
    sess.close()



