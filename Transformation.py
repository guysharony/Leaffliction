from plantcv import plantcv as pcv


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

    def _grayscale_hsv(self, channel: str, threshold: int, object_type: str = 'light'):
        # Converting RGB image to HSV grayscale by extracting the saturation channel.
        s_channel = pcv.rgb2gray_hsv(rgb_img=self.image, channel=channel)

       # Apply binary thresholding to grayscale image.
        # Pixels with intensity values greater than the threshold will be classified as foreground (white),
        # while pixels with intensity values less than or equal to the threshold will be classified as background (black).
        s_binary = pcv.threshold.binary(gray_img=s_channel, threshold=threshold, object_type=object_type)
        return s_binary

    def _grayscale_lab(self, channel: str, threshold: int, object_type: str = 'light'):
        # Converting RGB image to LAB grayscale by extracting the B (blue-yellow) channel.
        b_channel = pcv.rgb2gray_lab(rgb_img=self.image, channel=channel)

        # Apply binary thresholding to grayscale image.
        # Pixels with intensity values greater than 130 will be classified as foreground (white),
        # while pixels with intensity values less than or equal to 160 will be classified as background (black).
        s_binary = pcv.threshold.binary(gray_img=b_channel, threshold=threshold, object_type=object_type)
        return s_binary

    def original(self):
        if self.filter in ('all', 'original') and self.debug in ('plot', 'print'):
            pcv.params.debug = self.debug

        self.image, self.path, self.filename = pcv.readimage(self.pathname)
        pcv.params.debug = None
        return self.image

    def gaussian_blur(self):
        s_gray = self._grayscale_hsv(channel='s', threshold=58)

        if self.filter in ('all', 'gaussian_blur') and self.debug in ('plot', 'print'):
            pcv.params.debug = self.debug

        s_blur = pcv.gaussian_blur(s_gray, ksize=(5, 5))
        pcv.params.debug = None
        return s_blur

    def median_blur(self):
        s_gray = self._grayscale_hsv(channel='s', threshold=58)

        s_blur = pcv.median_blur(s_gray, ksize=(5, 5))
        return s_blur

    def mask(self):
        def background_mask():
            b_gray = self._grayscale_lab(channel='b', threshold=160)
            m_blur = self.median_blur()

            l_or = pcv.logical_or(bin_img1=m_blur, bin_img2=b_gray)

            masked = pcv.apply_mask(self.image, mask=l_or, mask_color="white")
            return masked

        masked_background = background_mask()

        def plant_mask():
            # Convert RGB to LAB and extract the Green-Magenta and Blue-Yellow channels
            masked_a = pcv.rgb2gray_lab(rgb_img=masked_background, channel='a')
            masked_b = pcv.rgb2gray_lab(rgb_img=masked_background, channel='b')

            # Threshold the green-magenta and blue images
            maskeda_thresh = pcv.threshold.binary(gray_img=masked_a, threshold=115, object_type='dark')
            maskeda_thresh1 = pcv.threshold.binary(gray_img=masked_a, threshold=110, object_type='light')
            maskedb_thresh = pcv.threshold.binary(gray_img=masked_b, threshold=130, object_type='light')

            # Join the thresholded saturation and blue-yellow images (OR)
            ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
            ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)

            # Fill small objects
            ab_fill = pcv.fill(ab, 200)

            return ab_fill

        masked_plant = plant_mask()

        if self.filter in ('all', 'mask') and self.debug in ('plot', 'print'):
            pcv.params.debug = self.debug

        # Apply mask (for VIS images, mask_color=white)
        masked = pcv.apply_mask(masked_background, masked_plant, mask_color='white')
        pcv.params.debug = None
        return masked

    def transformations(self):
        self.original()
        self.gaussian_blur()
        self.mask()

if __name__ == "__main__":
    pathname = './datasets/images/Apple/image_test.JPG'

    transformation = Transformation(pathname, filter='mask', debug='print')
    transformation.transformations()