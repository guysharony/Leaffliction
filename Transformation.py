import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv


class Transformation:
    def __init__(self, source: str, transformation: str, debug: str = "plot"):
        """
        Initializes a Transformation object.

        Args:
            source (str): The path to the source image.
            transformation (str): The type of transformation to apply.
        """
        # Reading image
        self.source = source
        self.destination = None
        self.image, self.path, self.filename = pcv.readimage(self.source)
        self.image_name = os.path.splitext(self.filename)[0]

        # Debug
        self.debug = debug
        self.transformation = transformation

    def _grayscale_hsv(self, channel: str, threshold: int):
        """
        Converts an image to grayscale using the HSV color space.

        Args:
            channel (str): The channel to extract.
            threshold (int): The threshold value for binarization.

        Returns:
            numpy.ndarray: The binary image.
        """
        s_channel = pcv.rgb2gray_hsv(rgb_img=self.image, channel=channel)

        s_binary = pcv.threshold.binary(
            gray_img=s_channel, threshold=threshold, object_type="light"
        )
        return s_binary

    def _grayscale_lab(self, channel: str, threshold: int):
        """
        Converts an image to grayscale using the LAB color space.

        Args:
            channel (str): The channel to extract.
            threshold (int): The threshold value for binarization.

        Returns:
            numpy.ndarray: The binary image.
        """
        b_channel = pcv.rgb2gray_lab(rgb_img=self.image, channel=channel)

        s_binary = pcv.threshold.binary(
            gray_img=b_channel, threshold=threshold, object_type="light"
        )
        return s_binary

    def set_destination(self, destination):
        """
        Sets the destination directory for saving transformed images.

        Args:
            destination (str): The path to the destination directory.
        """
        if destination is None:
            return

        if not os.path.exists(destination):
            os.makedirs(destination)

        self.debug = "print"
        self.outdir = f"./{destination}"
        pcv.params.debug_outdir = f"./{destination}"

    def plot_image(self, image, transformation):
        """
        Plots or prints the transformed image based on debug mode and
        transformation type.

        Args:
            image (numpy.ndarray): The transformed image to plot or
                print.
            transformation (str): The type of transformation applied
                to the image.
        """
        if self.transformation not in ("all", transformation):
            return

        if self.debug == "print":
            image_filename = f"{self.image_name}_{transformation}.jpg"
            pcv.print_image(
                image,
                os.path.join(self.outdir, image_filename),
            )
        elif self.debug == "plot":
            pcv.plot_image(image)

    def original(self):
        """
        Retrieves the original image and plots or prints it.
        """
        self.image, self.path, self.filename = pcv.readimage(self.source)
        self.plot_image(self.image, "original")

    def blur(self):
        """
        Applies Gaussian blur to the image.

        Returns:
            numpy.ndarray: The blurred image.
        """
        s_gray = self._grayscale_hsv(channel="s", threshold=58)

        g_blur = pcv.gaussian_blur(s_gray, ksize=(5, 5))
        self.plot_image(g_blur, "blur")

        self._g_blur = g_blur
        return g_blur

    def median_blur(self):
        """
        Applies median blur to the image.

        Returns:
            numpy.ndarray: The blurred image.
        """
        s_gray = self._grayscale_hsv(channel="s", threshold=58)

        self._m_blur = pcv.median_blur(s_gray, ksize=(5, 5))
        return self._m_blur

    def background_mask(self):
        """
        Generates a mask for the background of the image.

        Returns:
            numpy.ndarray: The masked image.
        """
        b_gray = self._grayscale_lab(channel="b", threshold=160)

        l_or = pcv.logical_or(bin_img1=self._m_blur, bin_img2=b_gray)

        masked = pcv.apply_mask(self.image, mask=l_or, mask_color="white")
        self._background_mask = masked
        return masked

    def plant_mask(self):
        """
        Generates a mask for the plant in the image.

        Returns:
            numpy.ndarray: The mask for the plant.
        """
        masked_a = pcv.rgb2gray_lab(rgb_img=self._background_mask, channel="a")
        masked_b = pcv.rgb2gray_lab(rgb_img=self._background_mask, channel="b")

        maskeda_thresh = pcv.threshold.binary(
            gray_img=masked_a, threshold=115, object_type="dark"
        )
        maskeda_thresh1 = pcv.threshold.binary(
            gray_img=masked_a, threshold=135, object_type="light"
        )
        maskedb_thresh = pcv.threshold.binary(
            gray_img=masked_b, threshold=128, object_type="light"
        )

        # Join the thresholded saturation and blue-yellow images (OR)
        ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
        ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)

        # Fill small objects
        plant_mask = pcv.fill(bin_img=ab, size=200)
        self._plant_mask = plant_mask
        return plant_mask

    def mask(self):
        """
        Applies the combined background and plant mask to the original image.

        Returns:
            numpy.ndarray: The masked image.
        """
        masked = pcv.apply_mask(
            self._background_mask, self._plant_mask, mask_color="white"
        )
        self.plot_image(masked, "mask")

        self._mask = masked
        return masked

    def roi_border(self, image, border_width: int):
        """
        Adds a border around the region of interest (ROI) in the image.

        Args:
            image (numpy.ndarray): The input image.
            border_width (int): The width of the border to add.

        Returns:
            numpy.ndarray: The image with the added border.

        """
        image_width = image.shape[0]
        image_height = image.shape[1]

        for x in range(0, image_width):
            for y in range(0, image_height):
                if (
                    (0 <= x <= image_width and 0 <= y <= border_width)
                    or (
                        0 <= x <= image_width
                        and image_height - border_width <= y <= image_height
                    )
                    or (0 <= x <= border_width and 0 <= y <= image_height)
                    or (
                        image_width - border_width <= x <= image_width
                        and 0 <= y <= image_height
                    )
                ):
                    image[x, y] = (255, 0, 0)

        return image

    def roi(self):
        """
        Identifies and extracts the region of interest (ROI) in the image.
        """
        plant_mask = self._plant_mask
        roi_type = "partial"

        roi = pcv.roi.rectangle(
            img=plant_mask,
            x=0,
            y=0,
            w=self.image.shape[0],
            h=self.image.shape[1],
        )

        kept_mask = pcv.roi.filter(mask=plant_mask, roi=roi, roi_type=roi_type)

        roi_image = self.image.copy()
        roi_image[kept_mask != 0] = (0, 255, 0)
        self.roi_border(roi_image, 5)

        self.plot_image(roi_image, "roi")

        self._kept_mask = kept_mask

    def analyze(self):
        """
        Performs object analysis on the image using the plant mask.
        """
        analyze = pcv.analyze.size(
            img=self.image, labeled_mask=self._plant_mask, n_labels=1
        )
        self.plot_image(analyze, "analyze")

    def pseudolandmarks(self):
        """
        Detects pseudolandmarks on the image if the debug mode allows.
        """
        img = self.image
        plant_mask = self._plant_mask
        debug = self.debug and self.debug in ("plot", "print")
        transforme = self.transformation in ("all", "pseudolandmarks")

        if debug and transforme:
            pcv.params.debug = self.debug

        output_image = os.path.join(
            pcv.params.debug_outdir,
            (str(pcv.params.device) + "_x_axis_pseudolandmarks.png"),
        )
        pcv.homology.x_axis_pseudolandmarks(img=img, mask=plant_mask)

        if self.debug and self.debug in ("print"):
            image, _, _ = pcv.readimage(output_image)
            self.plot_image(image, "pseudolandmarks")
            os.remove(output_image)

        pcv.params.debug = None

    def _adjust_color_channels(self):
        """
        Adjusts and plots color channels of the plant observation.
        """
        channels = [
            "blue",
            "blue-yellow",
            "green",
            "green-magenta",
            "hue",
            "lightness",
            "red",
            "saturation",
            "value",
        ]

        plant_output = pcv.outputs.observations["plant_1"]

        for channel in channels:
            label = f"{channel}_frequencies"
            if label in plant_output.keys():
                y = plant_output[label]["value"]
                x = plant_output[label]["label"]

                if channel in ("blue-yellow", "green-magenta"):
                    x = [x + 128 for x in x]

                if channel in ("lightness", "saturation", "value"):
                    x = [x * 2.55 for x in x]

                if channel == "hue":
                    x = [x for x in x if 0 <= x <= 255]
                    y = y[: len(x)]

                plt.plot(x, y, label=channel)

    def colors(self):
        """
        Generates and plots the color histogram distribution of the plant.
        """
        img = self.image
        mask, _ = pcv.create_labels(mask=self._kept_mask)
        pcv.analyze.color(
            rgb_img=img, colorspaces="all", labeled_mask=mask, label="plant"
        )

        self._adjust_color_channels()

        plt.legend()
        plt.title("Color histogram")
        plt.xlabel("Pixel intensity")
        plt.ylabel("Proportion of pixels (%)")
        plt.grid(linestyle="--")

        debug = self.debug and self.debug in ("plot", "print")
        transforme = self.transformation in ("all", "colors")
        image_filename = "{self.image_name}_colors.jpg"
        if transforme and debug:
            if self.debug == "plot":
                plt.show()
            elif self.debug == "print":
                plt.savefig(os.path.join(self.outdir, image_filename))

        plt.close()

    def transformations(self):
        """
        Applies a series of image transformations and analyses in sequence.
        """
        self.original()
        self.blur()

        self.median_blur()
        self.background_mask()
        self.plant_mask()
        self.mask()

        self.roi()

        self.analyze()

        self.pseudolandmarks()

        self.colors()


def parser():
    """
    Parses command-line arguments for image transformation and analysis.

    Returns:
        Tuple: A tuple containing the path to the image, source directory,
               destination directory, and the transformation type specified.
    Raises:
        ValueError: If conflicting arguments are provided or if no destination
                    folder is specified.
    """
    parser = argparse.ArgumentParser(
        description="""
            An image processing tool for transforming and analyzing images.
        """
    )

    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        help="path to the image for transformation",
    )
    parser.add_argument(
        "-src",
        type=str,
        nargs=1,
        default=None,
        help="transform all images in a folder"
    )
    parser.add_argument(
        "-dst",
        type=str,
        nargs=1,
        default=None,
        help="destination directory for transformed images",
    )

    # Transformations
    parser.add_argument(
        "-original",
        help="extract the original image",
        action="store_true"
    )
    parser.add_argument(
        "-blur",
        help="apply Gaussian blur to the image",
        action="store_true"
    )
    parser.add_argument(
        "-mask",
        help="generate a mask for the image",
        action="store_true"
    )
    parser.add_argument(
        "-roi",
        help="detect and extract region of interest objects",
        action="store_true",
    )
    parser.add_argument(
        "-analyze",
        help="perform object analysis on the image",
        action="store_true"
    )
    parser.add_argument(
        "-pseudolandmarks",
        help="detect pseudolandmarks for the image",
        action="store_true",
    )
    parser.add_argument(
        "-colors",
        help="generate color histogram distribution",
        action="store_true"
    )

    # Parser
    args = parser.parse_args(sys.argv[1::])

    if args.src and not args.dst:
        raise ValueError("[-dst] Destination folder not specified.")
    elif args.src and args.path:
        raise ValueError(
            "[-src, -path] Source and path cannot be \
                specified at the same time."
        )

    transformations = np.array(
        [
            "original",
            "blur",
            "mask",
            "roi",
            "analyze",
            "pseudolandmarks",
            "colors"
        ]
    )

    options = np.array(
        [
            args.original,
            args.blur,
            args.mask,
            args.roi,
            args.analyze,
            args.pseudolandmarks,
            args.colors,
        ]
    )

    transformation_options = transformations[options]
    if len(transformation_options) > 1:
        raise ValueError(
            "[transformation] Only 1 transformation can be specified."
        )

    if len(transformation_options) == 1:
        transformation_value = transformation_options[0]
    else:
        transformation_value = "all"

    return (
        args.path,
        args.src[0] if args.src else None,
        args.dst[0] if args.dst else None,
        transformation_value,
    )


def main():
    """
    Executes the image transformation and analysis process based on
    command-line arguments.

    Raises:
        ValueError: If the source folder doesn't exist, or if it contains
                    non-image files, or if the specified file path doesn't
                    exist.
    """
    path, src, dst, transformation = parser()

    if src and dst:
        if not os.path.exists(src) or not os.path.isdir(src):
            raise ValueError("[-src] Folder doesn't exist.")

        for file in os.listdir(src):
            if os.path.isdir(os.path.join(src, file)):
                raise ValueError(
                    "[-src] Folder must contain only images to transforme."
                )

        for file in os.listdir(src):
            transforme = Transformation(
                source=os.path.join(src, file),
                transformation=transformation
            )
            transforme.set_destination(os.path.join(dst))
            transforme.transformations()
    else:
        if not os.path.exists(path) or not os.path.isfile(path):
            raise ValueError("[-src] File doesn't exist.")

        transforme = Transformation(
            source=path,
            transformation=transformation
        )
        transforme.set_destination(dst)
        transforme.transformations()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}")
