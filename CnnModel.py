import tensorflow as tf
from pylab import *


MODEL_INFO = {
    'learning_rate': 0.001,
    'input_layer_batch_size': 128,
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


class CnnModel:
    def __init__(self, model_info):
        self.data_path = r'.\data\is_num_images'
        self.model_path = r'.\data\is_num_model'
        self.model_info = model_info

        self.image_batch, self.label_batch = None, None
        self.output = None
        self.loss = None
        self.minimizer = None
        self.accuracy = None

        self.sess = None
        self.saver = None

    def get_image_paths_and_labels_from_image_files(self, input_max_size=60000):
        image_paths = []
        labels = []
        i = 0
        for f in os.listdir(self.data_path):
            image_paths.append(self.data_path + "\\" + f)
            if f.split("-")[0] == "0":
                labels.append(0)
            else:
                labels.append(1)

            if i >= input_max_size:
                break
            i += 1
        return image_paths, labels

    def create_batch(self, image_paths, labels, is_shuffle=True):
        image = tf.cast(image_paths, tf.string)
        label = tf.cast(labels, tf.int32)
        input_queue = tf.train.slice_input_producer([image, label], shuffle=is_shuffle)
        image = input_queue[0]
        label = input_queue[1]

        image = tf.read_file(image)
        image = tf.image.decode_jpeg(image, channels=3)
        if self.model_info['input_layer_image_channels'] == 1:
            image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize_image_with_crop_or_pad(image, self.model_info['input_layer_image_shape'][0],
                                                       self.model_info['input_layer_image_shape'][1])
        image, label = tf.train.batch([image, label],
                                      batch_size=self.model_info['input_layer_batch_size'],
                                      num_threads=64,
                                      capacity=1 + self.model_info['input_layer_batch_size'])

        self.image_batch = tf.cast(image, tf.float32)
        self.label_batch = tf.cast(label, tf.int32)
        print("batch load successful...")

    def create_model(self):
        if self.image_batch is None and self.label_batch is None:
            self.image_batch = tf.placeholder(dtype=tf.float32, shape=[self.model_info['input_layer_batch_size'],
                                                                       self.model_info['input_layer_image_shape'][0],
                                                                       self.model_info['input_layer_image_shape'][1],
                                                                       self.model_info['input_layer_image_channels']])
            self.label_batch = tf.placeholder(dtype=tf.int32, shape=[self.model_info['input_layer_batch_size']])
        layer_input = self.image_batch
        for layer_id in range(self.model_info['conv_and_pool_layer_num']):
            layer_input = self.create_conv_and_pool_layer(layer_id, layer_input)

        layer_input = tf.reshape(layer_input, [self.model_info['input_layer_batch_size'], -1])

        for layer_id in range(self.model_info['fully_connected_layer_num']):
            layer_input = self.create_fully_connected_layer(layer_id, layer_input)
        self.output = self.create_output_layer(layer_input)

        self.loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.label_batch))
        self.minimizer = tf.train.AdamOptimizer(self.model_info['learning_rate']).minimize(self.loss)

        correct = tf.nn.in_top_k(self.output, self.label_batch, 1)
        correct = tf.cast(correct, tf.float16)
        self.accuracy = tf.reduce_mean(correct)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=self.sess, coord=coord)

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
        return output
