from plantcv import plantcv as pcv
import matplotlib.pyplot as plt


class Transformation:
    def __init__(self, pathname: str, filter: str, debug: str | None = None) -> None:
        # Reading image
        self.image = None
        self.path = None
        self.filename = None
        self.pathname = pathname

        # Debug
        self.debug = debug
        self.filter = filter

    def _grayscale_hsv(self):
        # Converting RGB image to HSV grayscale by extracting the saturation channel.
        s_channel = pcv.rgb2gray_hsv(rgb_img=self.image, channel='s')

       # Apply binary thresholding to grayscale image.
        # Pixels with intensity values greater than 55 will be classified as foreground (white),
        # while pixels with intensity values less than or equal to 55 will be classified as background (black).
        s_binary = pcv.threshold.binary(gray_img=s_channel, threshold=55, object_type='light')
        return s_binary

    def _grayscale_lab(self):
        # Converting RGB image to LAB grayscale by extracting the B (blue-yellow) channel.
        b_channel = pcv.rgb2gray_lab(rgb_img=self.image, channel='b')

        # Apply binary thresholding to grayscale image.
        # Pixels with intensity values greater than 130 will be classified as foreground (white),
        # while pixels with intensity values less than or equal to 130 will be classified as background (black).
        s_binary = pcv.threshold.binary(gray_img=b_channel, threshold=130, object_type='light')
        return s_binary

    def original(self):
        if self.filter in ('all', 'original') and self.debug in ('plot', 'print'):
            pcv.params.debug = self.debug

        self.image, self.path, self.filename = pcv.readimage(self.pathname)
        pcv.params.debug = None
        return self.image

    def gaussian_blur(self):
        s_gray = self._grayscale_hsv()

        if self.filter in ('all', 'gaussian_blur') and self.debug in ('plot', 'print'):
            pcv.params.debug = self.debug

        s_blur = pcv.gaussian_blur(s_gray, ksize=(5, 5))
        pcv.params.debug = None
        return s_blur

    def _median_blur(self):
        s_gray = self._grayscale_hsv()

        s_blur = pcv.median_blur(s_gray, ksize=(5, 5))
        return s_blur

    def mask(self):
        def mask_1():
            b_gray = self._grayscale_lab()

            if self.filter in ('all', 'mask_1') and self.debug in ('plot', 'print'):
                pcv.params.debug = self.debug

            m_blur = self._median_blur()

            l_or = pcv.logical_or(bin_img1=m_blur, bin_img2=b_gray)

            masked = pcv.apply_mask(self.image, mask=l_or, mask_color="white")
            pcv.params.debug = None
            return masked

        return mask_1()

    def transformations(self):
        self.original()
        self.gaussian_blur()
        self.mask()

if __name__ == "__main__":
    pathname = './datasets/images/Apple/image_test.JPG'

    transformation = Transformation(pathname, filter='mask_1', debug='print')
    transformation.transformations()