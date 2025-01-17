import cv2 as cv
import numpy as np

# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter
from options.visualize_options import VisualizeOptions
from data_prep.render import readkeypointsfile, renderpose25

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


def build_compare_figure(img_1, img_2):
    return np.concatenate([img_1, img_2], axis=0)


if __name__ == "__main__":

    render_shape = opt.render_shape

    # Read in the keypoints file
    keypoints = readkeypointsfile(opt.keypoints)

    if opt.compare_image:
        compare_img = cv.imread(opt.compare_image)
        render_shape = compare_img.shape

    canvas = 255 * np.ones(render_shape, dtype="uint8")

    # Render the keypoints to an image
    rendered = visualize_keypoints(keypoints, canvas)

    if opt.compare_image:
        full_fig = build_compare_figure(rendered, compare_img)
        display(full_fig, compare=True)
    else:
        display(rendered, compare=False)
