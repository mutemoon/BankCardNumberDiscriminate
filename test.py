from PIL import Image
from pylab import *
import random
import os

crop_size = 50
path = r'.\data\test_images\2.jpeg'
out_path = r'.\data\is_num_images\\'
original_image = Image.open(path)
count_1 = 3282
count_0 = 3708
xy1 = [91, 434]
xy2 = [857, 472]


def process_image(filename, mwidth=960, mheight=720):
    image = Image.open(filename)
    w, h = image.size
    if w <= mwidth and h <= mheight:
        print(filename, 'is OK.')
        return
    if (1.0 * w / mwidth) > (1.0 * h / mheight):
        scale = 1.0 * w / mwidth
        new_im = image.resize((int(w / scale), int(h / scale)), Image.ANTIALIAS)

    else:
        scale = 1.0 * h / mheight
        new_im = image.resize((int(w / scale), int(h / scale)), Image.ANTIALIAS)

    imshow(array(new_im))
    show()
    return new_im


resize_image = process_image(path)


for i in range(6000):
    if i > 3000:
        x = random.randint(0, resize_image.size[0])
        y = random.randint(0, resize_image.size[1])
    else:
        x = random.randint(xy1[0] - crop_size, xy2[0] - crop_size)
        y = random.randint(xy1[1] - crop_size, xy2[1] - crop_size)
    image = resize_image.crop((x, y, x + crop_size, y + crop_size))
    if xy1[0] - crop_size < x < xy2[0] - crop_size and xy1[1] - crop_size < y < xy2[1] - crop_size:
        count_1 += 1
        image.save(out_path + '1-' + str(count_1) + '.jpg')
    else:
        count_0 +=
        image.save(out_path + '0-' + str(count_0) + '.jpg')
        # os.remove(out_path + '0-' + str(count_0) + '.jpg')

# imshow(array(original_image))
# show()
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
