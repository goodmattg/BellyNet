import argparse

from utils.argparse import file_exists


class VisualizeOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            "--keypoints",
            type=file_exists,
            required=True,
            default="keypoints",
            help="path to keypoint file.",
        )

        self.parser.add_argument(
            "--render_shape",
            nargs="+",
            type=int,
            default=(1080, 1920, 3),
            help="original frame size of source video, e.g. 1080 1920 3",
        )

        self.parser.add_argument(
            "--compare_image",
            type=file_exists,
            default=None,
            help="Image to display alongside rendered keypoints (.png)",
        )

        self.parser.add_argument(
            "--save_path",
            type=file_exists,
            default=None,
            help="Save rendered keypoints to file (.png)",
        )

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
