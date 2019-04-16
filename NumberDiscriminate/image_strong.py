import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)  # 图像变换函数


image_path = r'.\data\single_images\0-0.jpg'
image = cv2.imread(image_path, 1)  # BGR
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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


w = 4
h = 4
for i in range(1, w * h):
    plt.subplot(w, h, i)
    aug = strong_aug(100)
    img_strong_aug = aug(image=image)['image']
    plt.imshow(img_strong_aug)
plt.subplot(w, h, 16)
plt.imshow(image)
image = Image.fromarray(image)
image.save("1.jpg")
plt.show()

save_path = r'.\d'
for i in range()