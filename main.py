import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import os
from PIL import Image
from pylab import *
import random

MODEL_INFO = {
    'learning_rate': 0.0001,
    'input_layer_batch_size': 1000,
    'input_layer_image_shape': [10, 10],
    'input_layer_image_channels': 1,
    'conv_and_pool_layer_num': 2,
    'conv_and_pool_layer_filter_ksize': [
        [2, 2],
        [2, 2]
    ],
    'conv_and_pool_layer_filter_strides': [
        [1, 1],
        [1, 1]
    ],
    'conv_and_pool_layer_filter_cores': [32, 64],

    'conv_and_pool_layer_pool_ksize': [
        [3, 3],
        [2, 2]
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
        self.prob = None
        self.output = None
        self.drop_output = None
        self.loss = None
        self.minimizer = None
        self.accuracy = None

        self.sess = None
        self.saver = None

    def get_image_paths_and_labels_from_image_files(self, input_max_size=200000):
        image_paths = []
        labels = []
        i = 0
        for f in os.listdir(self.data_path):
            # if int(f.split("-")[1].split(".")[0]) > 3282:
            #     continue
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
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize_image_with_crop_or_pad(image, self.model_info['input_layer_image_shape'][0],
                                                       self.model_info['input_layer_image_shape'][1])

        image_paths, labels = tf.train.batch([image, label],
                                             batch_size=self.model_info['input_layer_batch_size'],
                                             num_threads=64,
                                             capacity=1 + self.model_info['input_layer_batch_size'])

        self.image_batch = tf.cast(image_paths, tf.float32)
        self.label_batch = tf.cast(labels, tf.int32)
        print("batch load successful...")

    def create_model(self):
        layer_input = self.image_batch
        for layer_id in range(self.model_info['conv_and_pool_layer_num']):
            layer_input = self.create_conv_and_pool_layer(layer_id, layer_input)

        layer_input = tf.reshape(layer_input, [self.model_info['input_layer_batch_size'], -1])

        for layer_id in range(self.model_info['fully_connected_layer_num']):
            layer_input = self.create_fully_connected_layer(layer_id, layer_input)
        self.output = self.create_output_layer(layer_input)
        self.prob = tf.placeholder(dtype=tf.float32)
        self.drop_output = tf.nn.dropout(self.output, self.prob)
        self.loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.drop_output, labels=self.label_batch))
        self.minimizer = tf.train.AdamOptimizer(self.model_info['learning_rate']).minimize(self.loss)

        correct = tf.nn.in_top_k(self.drop_output, self.label_batch, 1)
        correct = tf.cast(correct, tf.float16)
        self.accuracy = tf.reduce_mean(correct)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
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

    def train(self):
        print("start training...")
        epoch_accuracy = 0
        epoch_step = 0
        for step in range(1, 100000):
            _, loss, accuracy = self.sess.run([self.minimizer, self.loss, self.accuracy], feed_dict={self.prob: 0.5})
            epoch_accuracy += float(accuracy)

            if step % 20 == 0:
                print("step:" + str(step), epoch_accuracy / epoch_step, loss)

            if step % 500 == 0:
                self.saver.save(self.sess, self.model_path + r'\model.ckpt', global_step=step)
                epoch_accuracy = 0
                epoch_step = 0
            epoch_step += 1


class Trainer:
    def __init__(self, model_info):
        self.cnn = CnnModel(model_info)

    def start_train(self):
        image_paths, labels = self.cnn.get_image_paths_and_labels_from_image_files()
        self.cnn.create_batch(image_paths, labels)
        self.cnn.create_model()
        self.cnn.train()


class Evaluate:
    def __init__(self, model_info):
        model_info["input_layer_batch_size"] = 1
        self.cnn = CnnModel(model_info)

    def evaluate_is_num(self, image_path):
        self.cnn.create_batch([image_path], [0])
        self.cnn.create_model()
        output = self.cnn.sess.run(tf.arg_max(self.cnn.output, 1), {1})
        print(output)
        return int(output)

    def find_num_position(self, slice_image_path):
        crop_size = self.cnn.model_info['input_layer_image_shape']
        # image = self.resize_image(slice_image_path)
        image = Image.open(slice_image_path)
        slice_images = []
        slice_image_paths = []
        slice_image_position = []
        for x in range(image.size[0] - int(crop_size[0] / 2)):
            for y in range(image.size[1] - int(crop_size[1] / 2)):
                if (x % crop_size[0] == 0) and (y % crop_size[1] == 0):# or x % crop_size[0] == int(crop_size[0] / 2)) and (y % crop_size[1] == 0 or y % crop_size[1] == int(crop_size[1] / 2)):
                    slice_images.append(image.crop((x, y, x + crop_size[0], y + crop_size[1])))
                    slice_image_position.append([x, y])
        for i in range(len(slice_images)):
            slice_image_path = r'.\data\temp\\' + str(i) + ".jpg"
            slice_image_paths.append(slice_image_path)
            slice_images[i].save(slice_image_path)
        self.cnn.model_info["input_layer_batch_size"] = len(slice_image_paths)
        self.cnn.create_batch(slice_image_paths, [0] * len(slice_image_paths), False)
        self.cnn.create_model()
        output = list(self.cnn.sess.run(tf.arg_max(self.cnn.drop_output, 1), feed_dict={self.cnn.prob: 1}))
        show_square_image = array(image)
        for i in range(len(output)):
            if output[i] == 1:
                for x in range(slice_image_position[i][0], slice_image_position[i][0] + crop_size[0]):
                    for y in range(slice_image_position[i][1], slice_image_position[i][1] + crop_size[1]):
                        if x >= image.size[0] or y >= image.size[1]:
                            continue
                        show_square_image[y][x][0] += 1
                        show_square_image[y][x][1] = 0
                        show_square_image[y][x][2] = 0
        # for i in range(len(most_vote)):
        #     for j in range(most_vote[i] * 10):
        #         show_square_image[i * crop_size[1] + 9][j] = [0, 0, 255]
        #         show_square_image[i * crop_size[1] + 10][j] = [0, 0, 255]
        #         show_square_image[i * crop_size[1] + 11][j] = [0, 0, 255]
        # print(most_vote, most_vote_y, most_vote[most_vote_y])
        imshow(show_square_image)
        show()
        for i in slice_image_paths:
            os.remove(i)
        return list(output)

    @staticmethod
    def resize_image(image_path, mwidth=960, mheight=720):
        image = Image.open(image_path)
        w, h = image.size
        if w <= mwidth and h <= mheight:
            return image
        if (1.0 * w / mwidth) > (1.0 * h / mheight):
            scale = 1.0 * w / mwidth
            new_im = image.resize((int(w / scale), int(h / scale)), Image.ANTIALIAS)

        else:
            scale = 1.0 * h / mheight
            new_im = image.resize((int(w / scale), int(h / scale)), Image.ANTIALIAS)

        return new_im


if __name__ == "__main__":
    cnn = Evaluate(MODEL_INFO)
    cnn.find_num_position(r'.\data\test_images\9.jpeg')
    # cnn.evaluate_is_num(r'.\data\temp\1027.jpg')
    # trainer = Trainer(MODEL_INFO)
    # trainer.start_train()
