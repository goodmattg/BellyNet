import cv2 as cv
import numpy as np

# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter
from options.visualize_options import VisualizeOptions
from utils.render import readkeypointsfile, renderpose25

opt = VisualizeOptions().parse(save=False)


def display(img, compare=False):
    if compare:
        down = cv.resize(img, (640, 720))
    else:
        down = cv.resize(img, (640, 360))

    cv.imshow("img", down)
    cv.waitKey(0)


def visualize_keypoints(keypoints, canvas):
    # Create the rendering of the keypoints file
    rendered = renderpose25(keypoints, canvas)
    return rendered
    # Assumes your input is a 16:9 aspect ratio
    # image = cv.resize(rendered, (640, 360))

    # cv.imshow("pose", image)
    # cv.waitKey(0)

    # Either (1)  show the rendered keypoints along
    # (2) Show the rendered keypoints splitscreen


def build_compare_figure(img_1, img_2):
    return np.concatenate([img_1, img_2], axis=0)


if __name__ == "__main__":

    # Read in the keypoints file
    keypoints = readkeypointsfile(opt.keypoints)
    canvas = 255 * np.ones(opt.render_shape, dtype="uint8")

    # Render the keypoints to an image
    rendered = visualize_keypoints(keypoints, canvas)

    if opt.compare_image:
        compare_img = cv.imread(opt.compare_image)
        full_fig = build_compare_figure(rendered, compare_img)

        display(full_fig, compare=True)
    else:
        display(rendered, compare=False)
