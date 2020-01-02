import argparse
import os
import pdb

import numpy as np

from PIL import Image
from shutil import copyfile
from render import renderpose, readkeypointsfile, renderface_sparse, renderhand


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--keypoints",
        type=str,
        required=True,
        default="keypoints",
        help="directory where keypoints are stored. Assumes .yml files for now.",
    )

    parser.add_argument(
        "--results",
        type=str,
        default=".",
        help="directory where to save generated files",
    )

    parser.add_argument(
        "--shape",
        nargs="+",
        type=int,
        default=(1080, 1920, 3),
        help="original frame size of source video, e.g. 1080 1920 3",
    )

    parser.add_argument(
        "--max-frame",
        type=int,
        default=10000,
        help="maximum frame number to look up keypoints for",
    )

    parser.add_argument(
        "--base-keypoint-filename",
        type=str,
        default="frame",
        help="base name in keypoint file string.",
    )

    parser.add_argument(
        "--out-height",
        type=int,
        default=512,
        help="Output rendered keypoints are resized to 2:1 aspect ratio regardless of input shape. This sets the height in the 2:1 output image",
    )

    opt = parser.parse_args()

    max_frame = opt.max_frame
    keypoints = opt.keypoints
    shape = tuple(opt.shape)
    results = opt.results
    base_keypoint_fname = opt.base_keypoint_filename
    out_height = opt.out_height

    start_y, start_x, end_y, end_x = 0, 0, shape[0], shape[1]

    if not os.path.exists(results):
        os.makedirs(results)

    # TODO: this shouldn't be a magic constant, but it is in her functions...
    numkeypoints = 8

    # print(max_frame)

    ok = True
    ix = 0
    while ok:

        keypoint_fname = "{}{}".format(base_keypoint_fname, "%06d" % ix)

        if ix == max_frame:
            break

        try:

            posepts = readkeypointsfile(
                os.path.join(keypoints, "{}_{}.yml".format(keypoint_fname, "pose"))
            )
            facepts = readkeypointsfile(
                os.path.join(keypoints, "{}_{}.yml".format(keypoint_fname, "face"))
            )
            rhandpts = readkeypointsfile(
                os.path.join(
                    keypoints, "{}_{}.yml".format(keypoint_fname, "hand_right")
                )
            )
            lhandpts = readkeypointsfile(
                os.path.join(keypoints, "{}_{}.yml".format(keypoint_fname, "hand_left"))
            )

            start_canvas = 255 * np.ones(shape, dtype="uint8")

            # pdb.set_trace()

            canvas = renderpose(posepts, start_canvas)
            canvas = renderface_sparse(facepts, canvas, numkeypoints)
            canvas = renderhand(rhandpts, canvas)
            canvas = renderhand(lhandpts, canvas)

            canvas = canvas[start_y:end_y, start_x:end_x, [2, 1, 0]]
            canvas = Image.fromarray(canvas)

            canvas = canvas.resize((2 * out_height, out_height), Image.ANTIALIAS)

            output_fname = "frame{}_render.png".format("%06d" % ix)

            canvas.save(os.path.join(results, output_fname))

            ix += 1
        except:
            ix += 1

