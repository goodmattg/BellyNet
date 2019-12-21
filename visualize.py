import cv2 as cv
import numpy as np

# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter
from options.visualize_options import VisualizeOptions
from utils.render import readkeypointsfile, renderpose25

opt = VisualizeOptions().parse(save=False)


def visualize_keypoints(keypoints, canvas):
    # Create the rendering of the keypoints file
    rendered = renderpose25(keypoints, canvas)
    # Assumes your input is a 16:9 aspect ratio
    image = cv.resize(rendered, (640, 360))

    cv.imshow("pose", image)
    cv.waitKey(0)
    # Either (1)  show the rendered keypoints along
    # (2) Show the rendered keypoints splitscreen


# Read in the keypoints file
keypoints = readkeypointsfile(opt.keypoints)
canvas = 255 * np.ones(opt.render_shape, dtype="uint8")

visualize_keypoints(keypoints, canvas)
