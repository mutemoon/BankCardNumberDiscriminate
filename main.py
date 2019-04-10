# encoding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 截断正态分布，此函数原型为尺寸、均值、标准差
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # strides第0位和第3为一定为1，剩下的是卷积的横向和纵向步长


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 参数同上，ksize是池化块的大小


x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# 图像转化为一个四维张量，第一个参数代表样本数量，-1表示不定，第二三参数代表图像尺寸，最后一个参数代表图像通道数
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积加池化
w_conv1 = weight_variable([5, 5, 1, 32])  # 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积加池化
w_conv2 = weight_variable([5, 5, 32, 64])  # 多通道卷积，卷积出64个特征
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 原图像尺寸28*28，第一轮图像缩小为14*14，共有32张，第二轮后图像缩小为7*7，共有64张

w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # 展开，第一个参数为样本数量，-1未知
f_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout操作，减少过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(f_fc1, keep_prob)

w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))  # 定义交叉熵为loss函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 调用优化器优化
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(2000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images[0:500], y_: mnist.test.labels[0:500], keep_prob: 1.0}))


model = {
    'input_layer_batch_size': 10,
    'input_layer_image_shape': [28, 28],
    'input_layer_image_channel': 3,

    'conv_and_pool_layer_num': 2,
    'conv_and_pool_layer_filter_ksize': [
        [5, 5],
        [5, 5]
    ],
    'conv_and_pool_layer_filter_strides': [
        [1, 1],
        [1, 1]
    ],
    'conv_and_pool_layer_filter_cores': [32, 64],

    'conv_and_pool_layer_pool_ksize': [
        [3, 3],
        [3, 3]
    ],
    'conv_and_pool_layer_pool_strides': [
        [2, 2],
        [2, 2]
    ],

    'fully_connected_layer_num': 2,
    'fully_connected_layer_units': [128, 128],

    'output_layer_units': 2
}


class CnnModel:
    def __init__(self, model_shape):
        self.sess = tf.Session()
        self.model_info = model_shape

        self.input_batch = None
        self.output = None
        self.accuracy = None
        self.loss = None
        self.train = None

        self.sess_init = tf.global_variables_initializer()
        self.sess.run(self.sess_init)

    def create_model(self):
        batch_shape = [
            self.model_info['input_layer_batch_size'],
            self.model_info['input_layer_image_shape'][0],
            self.model_info['input_layer_image_shape'][1],
            self.model_info['input_layer_image_channel']
        ]
        self.input_batch = tf.placeholder(dtype=tf.float32, shape=batch_shape)
        layer_input = self.input_batch
        for layer_id in self.model_info['conv_and_pool_layer_num']:
            layer_input = self.create_conv_and_pool_layer(layer_id, layer_input)

        for layer_id in self.model_info['fully_connected_layer_num']:
            layer_input = self.create_fully_connected_layer(layer_id, layer_input)

    def create_conv_and_pool_layer(self, layer_id, layer_input):
        conv_ksize = [
            self.model_info["conv_and_pool_layer_filter_shape"][layer_id][0],
            self.model_info["conv_and_pool_layer_filter_shape"][layer_id][1],
            self.model_info["input_layer_image_channel"],
            self.model_info["conv_and_pool_layer_filter_cores"][layer_id]
        ]
        if layer_id > 0:
            conv_ksize[2] = self.model_info["conv_and_pool_layer_filter_cores"][layer_id - 1]
        conv_strides = [
            1,
            self.model_info["conv_and_pool_layer_filter_strides"][layer_id][0],
            self.model_info["conv_and_pool_layer_filter_strides"][layer_id][1],
            1
        ]

        pool_ksize = [
            1,
            self.model_info["conv_and_pool_layer_pool_ksize"][layer_id][0],
            self.model_info["conv_and_pool_layer_pool_ksize"][layer_id][1],
            1
        ]
        pool_strides = [
            1,
            self.model_info["conv_and_pool_layer_pool_strides"][layer_id][0],
            self.model_info["conv_and_pool_layer_pool_strides"][layer_id][1],
            1
        ]

        with tf.variable_scope("conv_and_pool_layer_" + str(layer_id)) as scope:
            weight = tf.get_variable(name="weight",
                                     shape=conv_ksize,
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            bias = tf.get_variable("bias",
                                   shape=[conv_ksize[3]],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(layer_input, weight, strides=conv_strides, padding="SAME")
            conv = tf.nn.relu(tf.nn.bias_add(conv, bias))
            pool = tf.nn.max_pool(conv, ksize=pool_ksize, strides=pool_strides, padding="SAME")

        return pool

    def create_fully_connected_layer(self, layer_id, layer_input):
        with tf.variable_scope("fully_connected_layer_" + str(layer_id)) as scope:
            output = tf.layers.dense(layer_input,
                                  self.model_info["fully_connected_layer_units"][layer_id],
                                  tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.1, dtype=tf.float32),
                                  bias_initializer=tf.constant_initializer(0.1))
            return output