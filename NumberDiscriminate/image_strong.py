import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)  # 图像变换函数


def strong_aug(p=.5):
    return Compose([
        # RandomRotate90(),
        # Flip(),
        # Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


save_path = r'.\data\strong_single_images'
images_path = r'.\data\single_images'
# file_id = [0] * 10
# files = os.listdir(images_path)
#
# for f in files:
#     if f[0] != "_":
#         image_path = os.path.join(images_path, f)
#         image = cv2.imread(image_path, 1)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         aug = strong_aug(100)
#         for i in range(20):
#             strong_image = aug(image=image)['image']
#             strong_image = Image.fromarray(strong_image)
#             strong_image.save(save_path + "\\" + f[0] + "-" + str(file_id[int(f[0])]) + ".jpg")
#             file_id[int(f[0])] += 1
#
# file_id = 0
# files = os.listdir(images_path)
#
#
# save_path = r'.\data\strong_single_images'
# images_path = r'.\data\single_images'
# for f in files:
#     if f[0] == "_":
#         image_path = os.path.join(images_path, f)
#         image = cv2.imread(image_path, 1)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         aug = strong_aug(100)
#         for i in range(20):
#             strong_image = aug(image=image)['image']
#             strong_image = Image.fromarray(strong_image)
#             strong_image.save(save_path + "\\" + f[0] + "-" + str(file_id) + ".jpg")
#             file_id += 1


image = cv2.imread(r'.\data\single_images\0-0.jpg', 1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
aug = strong_aug(1)

w = 10
h = 10
for i in range(w * h):
    plt.subplot(w, h, i + 1)
    img_strong_aug = aug(image=image)['image']
    # img_strong_aug = Image.fromarray(img_strong_aug)
    plt.imshow(img_strong_aug)
plt.imshow(image)
plt.show()
