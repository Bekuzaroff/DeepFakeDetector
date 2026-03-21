




import cv2

from preprocess.image_preprocess import ImagePreprocess


if __name__ == '__main__':
    im_preproc = ImagePreprocess()
    im = im_preproc.read_image("cathedral.jpg")
    print(im)