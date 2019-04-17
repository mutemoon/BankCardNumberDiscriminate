import CnnModel
import tensorflow as tf
from PIL import Image
from pylab import *
import random
import os


class Evaluate:
    def __init__(self, model_info):
        self.batch_size = 100
        model_info['input_layer_batch_size'] = self.batch_size
        self.cnn = CnnModel.CnnModel(model_info)
        self.cnn.create_model()

    def evaluate_is_num(self, image_path):
        self.cnn.create_batch([image_path], [0])
        self.cnn.create_model()
        output = self.cnn.sess.run(tf.arg_max(self.cnn.output, 1))
        print(output)
        return int(output)

    def find_num_position(self, slice_image_path):
        input_image_shape = self.cnn.model_info['input_layer_image_shape']
        image = self.resize_image(slice_image_path)
        # image = Image.open(slice_image_path)
        slice_images = []
        slice_image_position = []
        for i in range(self.batch_size):
            x = random.randint(0, image.size[0] - input_image_shape[0])
            y = random.randint(0, image.size[1] - input_image_shape[1])
            slice_image = array(image.crop((x, y, x + input_image_shape[0], y + input_image_shape[1])).convert('L'))
            slice_images.append(slice_image[:, :, np.newaxis])
            slice_image_position.append([x, y])
        output = list(self.cnn.sess.run(tf.arg_max(self.cnn.output, 1), feed_dict={self.cnn.image_batch: slice_images}))
        original_image = []
        for i in range(11):
            original_image.append(np.array(image).copy())
        for xy in range(len(slice_image_position)):
            for x in range(slice_image_position[xy][0], slice_image_position[xy][0] + input_image_shape[0]):
                for y in range(slice_image_position[xy][1], slice_image_position[xy][1] + input_image_shape[1]):
                    original_image[output[xy]][y][x][0] += 1
                    original_image[output[xy]][y][x][1] = 0
                    original_image[output[xy]][y][x][2] = 0
        for i in range(len(original_image)):
            subplot(2, 6, i + 1)
            imshow(original_image[i])
        show()

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
    cnn = Evaluate(CnnModel.MODEL_INFO)
    for f in os.listdir(r'.\data\test_images'):
        cnn.find_num_position(r'.\data\test_images\\' + f)
