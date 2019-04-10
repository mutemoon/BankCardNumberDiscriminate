import tensorflow as tf
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


model_info = {
    'learning_rate': 0.001,
    'input_layer_batch_size': 100,
    'input_layer_image_shape': [20, 20],
    'input_layer_image_channels': 1,
    'conv_and_pool_layer_num': 2,
    'conv_and_pool_layer_filter_ksize': [
        [3, 3],
        [3, 3]
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
    'fully_connected_layer_num': 1,
    'fully_connected_layer_units': [20],
    'output_layer_units': 2
}

print("data load successful...")


class CnnModel:
    def __init__(self, model_info):
        self.data_path = r'.\data\is_num_images'
        self.model_path = '.\data\is_num_model'
        self.model_info = model_info
        self.output = None
        self.loss = None
        self.minimizer = None
        self.accuracy = None
        self.image_batch, self.label_batch = None, None

    def input_files(self, input_num=8000):
        image_batch = []
        label_batch = []
        i = 0
        for f in os.listdir(self.data_path):
            image_batch.append(self.data_path + "\\" + f)
            if f.split("-")[0] == "0":
                label_batch.append(0)
            else:
                label_batch.append(1)
            if i >= input_num:
                break
            i += 1
        return image_batch, label_batch

    def get_batch(self, image_batch, label_batch):
        image = tf.cast(image_batch, tf.string)
        label = tf.cast(label_batch, tf.int32)
        input_queue = tf.train.slice_input_producer([image, label])
        image = input_queue[0]
        label = input_queue[1]
        image = tf.read_file(image)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize_image_with_crop_or_pad(image, self.model_info['input_layer_image_shape'][0],
                                                       self.model_info['input_layer_image_shape'][1])
        image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                          batch_size=self.model_info['input_layer_batch_size'],
                                                          num_threads=64, capacity=1000,
                                                          min_after_dequeue=self.model_info['input_layer_batch_size'])
        self.image_batch = tf.cast(image_batch, tf.float32)
        self.label_batch = tf.cast(label_batch, tf.int32)

    def create_model(self):
        layer_input = self.image_batch
        for layer_id in range(self.model_info['conv_and_pool_layer_num']):
            layer_input = self.create_conv_and_pool_layer(layer_id, layer_input)

        layer_input = tf.reshape(layer_input, [self.model_info['input_layer_batch_size'], -1])

        for layer_id in range(self.model_info['fully_connected_layer_num']):
            layer_input = self.create_fully_connected_layer(layer_id, layer_input)
        self.output = self.create_output_layer(layer_input)
        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.label_batch))
        self.minimizer = tf.train.AdamOptimizer(self.model_info['learning_rate']).minimize(self.loss)
        correct = tf.nn.in_top_k(self.output, self.label_batch, 1)
        correct = tf.cast(correct, tf.float16)
        self.accuracy = tf.reduce_mean(correct)

    def create_conv_and_pool_layer(self, layer_id, layer_input):
        conv_ksize = [
            self.model_info["conv_and_pool_layer_filter_ksize"][layer_id][0],
            self.model_info["conv_and_pool_layer_filter_ksize"][layer_id][1],
            self.model_info["input_layer_image_channels"],
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

    def create_output_layer(self, layer_input):
        with tf.variable_scope("output_layer") as scope:
            output = tf.layers.dense(layer_input,
                                     self.model_info["output_layer_units"],
                                     None,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1, dtype=tf.float32),
                                     bias_initializer=tf.constant_initializer(0.1))
            print(layer_input.shape)
            print(output.shape)
        return output

    def train(self):
        self.image_batch, self.label_batch = self.input_files()
        self.get_batch(self.image_batch, self.label_batch)
        self.create_model()
        self.sess = tf.Session()
        self.sess_init = tf.global_variables_initializer()
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        self.saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        print("start training...")
        total_accuracy = 0
        step = 0
        for i in range(1, 100000):
            step += 1
            _, loss, accuracy = self.sess.run([self.minimizer, self.loss, self.accuracy])
            total_accuracy += float(accuracy)

            # if i % 125 == 0:
            print(i, total_accuracy / step, loss)

            if i % 500 == 0:
                self.saver.save(self.sess, self.model_path + '\model.ckpt', global_step=i)
                total_accuracy = 0
                step = 0

    def evaluate(self, image_path):
        self.model_info['input_layer_batch_size'] = 1
        self.get_batch([image_path], [0])
        self.create_model()
        self.sess = tf.Session()
        self.sess_init = tf.global_variables_initializer()
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        self.saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        output, loss, accuracy = self.sess.run([self.output, self.loss, self.accuracy])
        print(list(output))


cnn = CnnModel(model_info)
cnn.evaluate(r'.\data\is_num_images\0-297.jpg')

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#
#
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)  # 截断正态分布，此函数原型为尺寸、均值、标准差
#     return tf.Variable(initial)
#
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
#
# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # strides第0位和第3为一定为1，剩下的是卷积的横向和纵向步长
#
#
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 参数同上，ksize是池化块的大小
#
#
# x = tf.placeholder("float", shape=[None, 784])
# y_ = tf.placeholder("float", shape=[None, 10])
#
# # 图像转化为一个四维张量，第一个参数代表样本数量，-1表示不定，第二三参数代表图像尺寸，最后一个参数代表图像通道数
# x_image = tf.reshape(x, [-1, 28, 28, 1])
#
# # 第一层卷积加池化
# w_conv1 = weight_variable([5, 5, 1, 32])  # 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征
# b_conv1 = bias_variable([32])
#
# h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# # 第二层卷积加池化
# w_conv2 = weight_variable([5, 5, 32, 64])  # 多通道卷积，卷积出64个特征
# b_conv2 = bias_variable([64])
#
# h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
#
# # 原图像尺寸28*28，第一轮图像缩小为14*14，共有32张，第二轮后图像缩小为7*7，共有64张
#
# w_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # 展开，第一个参数为样本数量，-1未知
# f_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
#
# # dropout操作，减少过拟合
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(f_fc1, keep_prob)
#
# w_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
#
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))  # 定义交叉熵为loss函数
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 调用优化器优化
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
# sess = tf.InteractiveSession()
# sess.run(tf.initialize_all_variables())
# for i in range(2000):
#     batch = mnist.train.next_batch(50)
#     if i % 100 == 0:
#         train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
#         print("step %d, training accuracy %g" % (i, train_accuracy))
#     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#
# print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images[0:500], y_: mnist.test.labels[0:500], keep_prob: 1.0}))