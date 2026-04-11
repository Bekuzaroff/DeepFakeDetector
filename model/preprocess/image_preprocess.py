import cv2 as cv
import numpy as np


class ImagePreprocess():
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])


    def read_image(self, im_path: str, with_channels: bool = False):
        try:
            mat_im = cv.imread(im_path, int(with_channels))

            if mat_im is None:
                raise Exception("no such image, please check the image path")
            
            return mat_im
        except Exception as e:
            print(e)

    def im_preprocess(self, im_matrix, resize_to):
        im_matrix = cv.resize(im_matrix, resize_to)
        im_matrix = cv.cvtColor(im_matrix, cv.COLOR_RGB2BGR)
        im_matrix = im_matrix.astype(np.float32) / 255.0
        im_matrix = (im_matrix - self.mean) / self.std

        return im_matrix
        
