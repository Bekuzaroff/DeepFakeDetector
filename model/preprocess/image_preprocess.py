import cv2 as cv


class ImagePreprocess():
    def __init__(self):
        pass


    def read_image(self, im_path: str, with_channels: bool = 0):
        try:
            mat_im = cv.imread(im_path, int(with_channels))

            if mat_im is None:
                raise Exception("no such image, please check the image path")
            
            return mat_im
        except Exception as e:
            print(e)

    def im_preprocess(self, im_matrix):
        pass
