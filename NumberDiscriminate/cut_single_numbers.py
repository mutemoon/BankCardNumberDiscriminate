from PIL import Image
from pylab import *
import os


file_id = [0] * 10
images_path = r'.\data\images\\'
for f in os.listdir(images_path):
    file_path = os.path.join(images_path, f)
    image = Image.open(file_path)
    slice_images = []
    n_slice_images = 4
    for i in range(n_slice_images):
        slice_images.append(image.crop(((image.size[0] / n_slice_images) * i, 0, (image.size[0] / n_slice_images) * (i + 1), image.size[1])))

    for i in range(n_slice_images):
        if f[i] != '_':
            slice_path = r'.\data\single_images\\' + f[i] + "-" + str(file_id[int(f[i])]) + ".jpg"
            slice_images[i].save(slice_path)
            file_id[int(f[i])] += 1
            print(slice_path)