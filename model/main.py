




import cv2

from dataset.dataset import Dataset
from preprocess.image_preprocess import ImagePreprocess
import torchvision as tv

import kagglehub


if __name__ == '__main__':
    dataset = Dataset("Train")

    