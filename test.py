from PIL import Image
from pylab import *
import random
import os


path = r'.\data\test_images\2.jpeg'
out_path = r'.\data\is_num_images\\'
original_image = Image.open(path)
count_1 = 3282
count_0 = 3708
xy1 = [140, 640]
xy2 = [1280, 700]
for i in range(3000):
    x = random.randint(0, original_image.size[0])
    y = random.randint(0, original_image.size[1])
    # x = random.randint(xy1[0], xy2[0])
    # y = random.randint(xy1[1], xy2[1])
    image = original_image.crop((x, y, x + 20, y + 20))
    if xy1[0] < x < xy2[0] and xy1[1] < y < xy2[1]:
        count_1 += 1
        image.save(out_path + '1-' + str(count_1) + '.jpg')
    else:
        count_0 += 1
        image.save(out_path + '0-' + str(count_0) + '.jpg')
        # os.remove(out_path + '0-' + str(count_0) + '.jpg')
imshow(original_image)
show()

# import tensorflow as tf
# import os
# import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
#
# data_path = r'.\data\images'
# image_batch = []
# label_batch = []
# input_num = 100
# i = 0
# for f in os.listdir(data_path):
#     i += 1
#     if i > input_num:
#         break
#     input_file = data_path + "\\" + f
#     image = tf.image.decode_png(tf.read_file(input_file), 3)
#     image = tf.image.resize_image_with_crop_or_pad(image, 120, 46)
#     image_batch.append(image)
#     label_batch.append([1, 0])
# print(image_batch)


# import tensorflow as tf
#
#
# out = tf.constant([[[10, 10, 2, 3, 4, 5], [10, 10, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0]], [[10, 10, 2, 3, 4, 5], [10, 10, 2, 3, 4, 5], [10, 10, 2, 3, 4, 0]]], dtype=tf.float32)
# softmax_out = tf.nn.softmax(out)
# out_clipped = tf.clip_by_value(softmax_out, 1e-10, 0.999999)
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     o = sess.run(softmax_out)
#     print(list(o))
#     o = sess.run(out_clipped)
#     print(list(o))
#     o = sess.run(tf.reduce_mean(out_clipped))
#     print(o)
