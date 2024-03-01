from plantcv import plantcv as pcv
import matplotlib.pyplot as plt


class Transformation:
    def __init__(self, pathname: str, debug: str | None = None) -> None:
        # Reading image
        self.image = None
        self.path = None
        self.filename = None

        # Debug
        self.debug = debug

    def _grayscale_hsv(self):
        # Converting RGB image to HSV grayscale by extracting the saturation channel.
        s_channel = pcv.rgb2gray_hsv(rgb_img=self.image, channel='s')

        # Apply binary thresholding to RGB image to obtain grayscale image.
        s_binary = pcv.threshold.binary(gray_img=s_channel, threshold=55, object_type='light')
        return s_binary

    def _grayscale_lab(self):
        # Converting RGB image to LAB grayscale by extracting the B channel.
        s_channel = pcv.rgb2gray_lab(rgb_img=self.image, channel='b')

        # Apply binary thresholding to RGB image to obtain grayscale image.
        s_binary = pcv.threshold.binary(gray_img=s_channel, threshold=130, object_type='light')
        return s_binary

    def original(self):
        if self.debug == 'plot' or self.debug == 'print':
            pcv.params.debug = self.debug

        self.image, self.path, self.filename = pcv.readimage(pathname)
        pcv.params.debug = None
        return self.image

    def gaussian_blur(self):
        s_gray = self._grayscale_hsv()

        if self.debug == 'plot' or self.debug == 'print':
            pcv.params.debug = self.debug

        s_blur = pcv.gaussian_blur(s_gray, ksize=(5, 5))
        pcv.params.debug = None
        return s_blur

    def transformations(self):
        self.original()
        self.gaussian_blur()

if __name__ == "__main__":
    pathname = './datasets/images/Apple/image_test.JPG'

    transformation = Transformation(pathname, debug='print')
    transformation.transformations()