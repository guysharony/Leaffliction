from plantcv import plantcv as pcv


class Transformation:
    def __init__(self, pathname: str, filter: str, debug: str) -> None:
        # Reading image
        self.pathname = pathname
        self.image, self.path, self.filename = pcv.readimage(self.pathname)

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

        g_blur = pcv.gaussian_blur(s_gray, ksize=(5, 5))
        pcv.params.debug = None

        self._g_blur = g_blur
        return g_blur

    def median_blur(self):
        s_gray = self._grayscale_hsv(channel='s', threshold=58)

        if self.filter in ('median_blur') and self.debug in ('plot', 'print'):
            pcv.params.debug = self.debug

        m_blur = pcv.median_blur(s_gray, ksize=(5, 5))
        pcv.params.debug = None

        self._m_blur = m_blur
        return m_blur

    def background_mask(self):
        b_gray = self._grayscale_lab(channel='b', threshold=160)

        l_or = pcv.logical_or(bin_img1=self._m_blur, bin_img2=b_gray)

        masked = pcv.apply_mask(self.image, mask=l_or, mask_color="white")
        self._background_mask = masked
        return masked

    def plant_mask(self):
        # Convert RGB to LAB and extract the Green-Magenta and Blue-Yellow channels
        masked_a = pcv.rgb2gray_lab(rgb_img=self._background_mask, channel='a')
        masked_b = pcv.rgb2gray_lab(rgb_img=self._background_mask, channel='b')

        # Threshold the green-magenta and blue images
        maskeda_thresh = pcv.threshold.binary(gray_img=masked_a, threshold=115, object_type='dark')
        maskeda_thresh1 = pcv.threshold.binary(gray_img=masked_a, threshold=135, object_type='light')
        maskedb_thresh = pcv.threshold.binary(gray_img=masked_b, threshold=128, object_type='light')

        # Join the thresholded saturation and blue-yellow images (OR)
        ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
        ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)

        # Fill small objects
        plant_mask = pcv.fill(bin_img=ab, size=200)
        self._plant_mask = plant_mask
        return plant_mask

    def mask(self):
        if self.filter in ('all', 'mask') and self.debug in ('plot', 'print'):
            pcv.params.debug = self.debug

        # Apply mask (for VIS images, mask_color=white)
        masked = pcv.apply_mask(self._background_mask, self._plant_mask, mask_color='white')
        pcv.params.debug = None

        self._mask = masked
        return masked

    def roi_border(self, image, border_width: int):
        # HAS TO BE IMPROVED

        image_width = image.shape[0]
        image_height = image.shape[1]

        for x in range(0, image_width):
            for y in range(0, image_height):
                if (
                    (
                        0 <= x <= image_width and
                        0 <= y <= border_width
                    ) or
                    (
                        0 <= x <= image_width and
                        image_height - border_width <= y <= image_height
                    ) or
                    (
                        0 <= x <= border_width and
                        0 <= y <= image_height
                    ) or
                    (
                        image_width - border_width <= x <= image_width and
                        0 <= y <= image_height
                    )
                ):
                    image[x, y] = (255, 0, 0)

        return image


    def roi_objects(self):
        # HAS TO BE IMPROVED

        roi = pcv.roi.rectangle(
            img=self._plant_mask,
            x=0,
            y=0,
            w=self.image.shape[0],
            h=self.image.shape[1],
        )

        if self.filter in ('all', 'roi_objects') and self.debug in ('plot', 'print'):
            pcv.params.debug = self.debug

        kept_mask = pcv.roi.filter(
            mask=self._plant_mask,
            roi=roi,
            roi_type='partial'
        )

        roi_image = self.image.copy()
        roi_image[kept_mask != 0] = (0, 255, 0)
        self.roi_border(roi_image, 5)

        if self.filter in ('all', 'roi_objects') and self.debug in ('plot', 'print'):
            pcv.print_image(roi_image, 'roi_objects.png')

        pcv.params.debug = None

        return None

    def analyze_object(self):
        if self.filter in ('all', 'analyze_object') and self.debug in ('plot', 'print'):
            pcv.params.debug = self.debug

        shape_image = pcv.analyze.size(img=self.image, labeled_mask=self._plant_mask, n_labels=1)

        pcv.params.debug = None

        return None

    def pseudolandmarks(self):
        if self.filter in ('all', 'pseudolandmarks') and self.debug in ('plot', 'print'):
            pcv.params.debug = self.debug

        pcv.homology.x_axis_pseudolandmarks(img=self.image, mask=self._plant_mask)

        pcv.params.debug = None

        return None

    def transformations(self):
        self.original()
        self.gaussian_blur()

        self.median_blur()
        self.background_mask()
        self.plant_mask()
        self.mask()

        self.roi_objects()

        self.analyze_object()

        self.pseudolandmarks()

if __name__ == "__main__":
    pathname = './datasets/images/Apple/image_test.JPG'

    transformation = Transformation(pathname, filter='pseudolandmarks', debug='print')
    transformation.transformations()