from PIL import Image
from pylab import *
import random
import os


def process_image(filename, mwidth=960, mheight=720):
    image = Image.open(filename)
    w, h = image.size
    if w <= mwidth and h <= mheight:
        return image
    if (1.0 * w / mwidth) > (1.0 * h / mheight):
        scale = 1.0 * w / mwidth
        new_im = image.resize((int(w / scale), int(h / scale)), Image.ANTIALIAS)

    else:
        scale = 1.0 * h / mheight
        new_im = image.resize((int(w / scale), int(h / scale)), Image.ANTIALIAS)

    # imshow(array(new_im))
    # show()
    return new_im


crop_size = 20
path = r'.\data\test_images'
out_path = r'.\data\is_num_images\\'
count_1 = 0
count_0 = 0


xy = [[[165, 369], [791, 416]],
      [[95, 435], [861, 474]],
      [[113, 422], [817, 467]],
      [[66, 240], [592, 275]],
      [[102, 275], [559, 298]],
      [[286, 347], [915, 390]],
      [[101, 358], [824, 409]],
      [[81, 169], [458, 198]],
      [[81, 244], [534, 277]]]

file_i = 0
for file in os.listdir(path):
    resize_image = process_image(os.path.join(path, file))
    # imshow(array(resize_image))
    # show()
    for i in range(6000):
        if i > 3000:
            x = random.randint(0, resize_image.size[0])
            y = random.randint(0, resize_image.size[1])
        else:
            x = random.randint(xy[file_i][0][0] - crop_size, xy[file_i][1][0] - crop_size)
            y = random.randint(xy[file_i][0][1] - crop_size, xy[file_i][1][1] - crop_size)
        image = resize_image.crop((x, y, x + 20, y + 20))
        if xy[file_i][0][0] - crop_size < x < xy[file_i][1][0] - crop_size and xy[file_i][0][1] - crop_size < y < xy[file_i][1][1] - crop_size:
            count_1 += 1
            image.save(out_path + '1-' + str(count_1) + '.jpg')
        else:
            count_0 += 1
            image.save(out_path + '0-' + str(count_0) + '.jpg')
    file_i += 1

# for i in range(3283, 6584):
#     os.remove(r".\data\is_num_images\\1-" + str(i) + ".jpg")
