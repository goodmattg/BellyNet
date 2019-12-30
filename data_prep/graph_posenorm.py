import argparse
import cv2 as cv
import numpy as np

import math
import time
import copy
import matplotlib
import sys
import json
import os
import shutil

#%matplotlib inline

from PIL import Image
from shutil import copyfile
from render import *

disp = False

start = 822
end = 129502
step = 4

n = start
SIZE = 512
poselen = 75

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
##### pose normalization from source ----> target. Then apply pose ----> target mapping to complete transfer.

##### Must specifcy these parameters
parser.add_argument(
    "--target_keypoints",
    type=str,
    default="keypoints",
    help="directory where target keypoint files are stored, assumes .yml format for now.",
)
parser.add_argument(
    "--source_keypoints",
    type=str,
    default="keypoints",
    help="directory where source keypoint files are stored, assumes .yml format for now.",
)
parser.add_argument(
    "--target_shape",
    nargs="+",
    type=int,
    default=(1080, 1920, 3),
    help="original frame size of target video, e.g. 1080 1920 3",
)
parser.add_argument(
    "--source_shape",
    nargs="+",
    type=int,
    default=(1080, 1920, 3),
    help="original frame size of source video, e.g. 1080 1920 3",
)
parser.add_argument(
    "--source_frames",
    type=str,
    default="frames",
    help="directory where source frames are stored. Assumes .png files for now.",
)
parser.add_argument(
    "--results",
    type=str,
    default="frames",
    help="directory where to save generated files",
)

parser.add_argument(
    "--target_spread",
    nargs="+",
    type=int,
    help="range of frames to use for target video, e.g. 0 10000",
)

parser.add_argument(
    "--source_spread",
    nargs="+",
    type=int,
    help="range of frames to use for target video, e.g. 0 10000",
)

#### Optional (have defaults)
parser.add_argument(
    "--target_median_frac",
    type=float,
    default=0.5,
    help="for target video: fraction of distance from maximum toe position to median to use to calculate minimum toe position. Try 0.5 or 0.7 for reasonable videos with normal back/forth motion.",
)
parser.add_argument(
    "--source_median_frac",
    type=float,
    default=0.5,
    help="for source video: fraction of distance from maximum toe position to median to use to calculate minimum toe position. Try 0.5 or 0.7 for reasonable videos with normal back/forth motion.",
)
parser.add_argument(
    "--filestart",
    type=str,
    default="frame",
    help="file start name, files should be named filestart%06d before extension",
)
parser.add_argument(
    "--calculate_scale_translation",
    action="store_true",
    help="use this flag to calculate the translation and scale from scratch. Else, try to load them from a saved file.",
)
parser.add_argument(
    "--format",
    type=str,
    default="json",
    help="file format for keypoint files, only json and yaml are supported, [json|yaml]",
)


def get_keypoints_stats(mypath, myshape, spread, startname="frame", max_frames=20000):
    """
    Returns aggregate statistics for a set of keypoints

    Outputs:
        maxheight           deprecated
        mintoe,             deprecated
        maxtoe,             float64         (argmaximum ankle y-position across all poses)   
        avemintoe,          deprecated
        minmaxtoe,          deprecated
        mediantiptoe,       float64         (median of all maximum ankle y-positions across all poses)   
        getmediantiptoe,    list[float64]   (list of all maximum ankle y-positions across all poses)
        tiptoe_to_height,   dict()          (mapping from max ankle to height)
    """
    filenames = os.listdir(mypath)

    maxheight = 0
    mintoe = myshape[0]
    maxtoe = 0
    count = 0
    avemintoe = 0
    minmaxtoe = myshape[0]
    getmediantiptoe = []
    heights = []
    tiptoe_to_height = {}
    ok = True

    while ok:
        mynum = np.random.randint(low=spread[0], high=spread[1])
        strmynum = "%06d" % mynum
        f_yaml = startname + strmynum + "_pose.yml"
        f_json = startname + strmynum + "_keypoints.json"

        if os.path.isfile(os.path.join(mypath, f_yaml)) or os.path.isfile(
            os.path.join(mypath, f_json)
        ):
            key_name = os.path.join(mypath, f_yaml)

            posepts = []

            ### try yaml
            try:
                posepts = readkeypointsfile(key_name)
            except:
                continue

            if posepts is None:  ## try json
                key_name = os.path.join(mypath, f_json)
                posepts, _, _, _ = readkeypointsfile(key_name)
                if posepts is None:
                    print("unable to read keypoints file")
                    import sys

                    sys.exit(0)

            # if len(posepts) == 75:
            #     posepts = map_25_to_23(posepts)

            if len(posepts) != poselen:
                print("EMPTY stats", key_name, len(posepts))
                continue
            else:
                # get_pose_stats returns None if unable to compute stats
                check_me = get_pose_stats(posepts)
                if check_me:
                    height, min_tip_toe, max_tip_toe = check_me
                    # Track the maximum height seen in any frame
                    maxheight = max(maxheight, height)
                    # Track every height we've seen
                    heights += [height]
                    # Track the min ankle position
                    mintoe = min(mintoe, min_tip_toe)
                    # Track the max ankle position
                    maxtoe = max(maxtoe, max_tip_toe)
                    # TODO according to naming this is an outright bug...
                    avemintoe += max_tip_toe
                    # Keep track of the smallest max-height
                    minmaxtoe = min(max_tip_toe, minmaxtoe)
                    getmediantiptoe += [max_tip_toe]
                    count += 1

                    # Track mapping from max ankle to height
                    if max_tip_toe not in tiptoe_to_height:
                        tiptoe_to_height[max_tip_toe] = [height]
                    else:
                        tiptoe_to_height[max_tip_toe] += [height]

                    # print("cannot find file " + os.path.join(mypath, f_yaml))
                    if count % 100 == 0:
                        print(count)
                    if count >= max_frames:
                        ok = False
                    if count >= spread[1] - spread[0]:
                        ok = False

    # NOTE: THIS IS ACTUALLY THE AVERAGE MAX ANKLE HEIGHT
    avemintoe = avemintoe / float(count)
    # Median of the max ankle positions
    mediantiptoe = np.median(getmediantiptoe)

    return (
        maxheight,
        mintoe,
        maxtoe,
        avemintoe,
        minmaxtoe,
        mediantiptoe,
        getmediantiptoe,
        tiptoe_to_height,
    )


def get_minmax_scales(tiptoe_to_height0, tiptoe_to_height1, translation, frac):
    sorted_tiptoes0 = list(tiptoe_to_height0.keys()).sort()

    t_maxtoe, t_horizon = translation[0]
    s_maxtoe, s_horizon = translation[1]

    range0 = (t_maxtoe - t_horizon) * frac
    range1 = (s_maxtoe - s_horizon) * frac

    toe_keys0 = [
        x for x in list(tiptoe_to_height0.keys()) if abs(x - t_maxtoe) <= range0
    ]
    horizon_keys0 = [
        x for x in list(tiptoe_to_height0.keys()) if abs(x - t_horizon) <= range0
    ]

    max_heightclose0 = 0
    for key in toe_keys0:
        cur_h = max(tiptoe_to_height0[key])
        if cur_h > max_heightclose0:
            max_heightclose0 = cur_h

    max_heightfar0 = 0
    for key in horizon_keys0:
        cur_h = max(tiptoe_to_height0[key])
        if cur_h > max_heightfar0:
            max_heightfar0 = cur_h

    toe_keys1 = [
        x for x in list(tiptoe_to_height1.keys()) if abs(x - s_maxtoe) <= range1
    ]
    horizon_keys1 = [
        x for x in list(tiptoe_to_height1.keys()) if abs(x - s_horizon) <= range1
    ]

    max_heightclose1 = 0
    for key in toe_keys1:
        cur_h = max(tiptoe_to_height1[key])
        if cur_h > max_heightclose1:
            max_heightclose1 = cur_h

    max_heightfar1 = 0
    for key in horizon_keys1:
        cur_h = max(tiptoe_to_height1[key])
        if cur_h > max_heightfar1:
            max_heightfar1 = cur_h

    print("far")
    print(max_heightfar0, max_heightfar1)
    print("near")
    print(max_heightclose0, max_heightclose1)

    max_all0 = max(tiptoe_to_height0.values())[0]
    max_all1 = max(tiptoe_to_height1.values())[0]

    if max_all0 - max_heightclose0 > 0.1 * max_all0:
        print("reset max_heightclose0")
        max_heightclose0 = max_all0

    if max_all1 - max_heightclose1 > 0.1 * max_all1:
        print("reset max_heightclose1")
        max_heightclose1 = max_all1

    scale_close = max_heightclose0 / float(max_heightclose1)
    scale_far = max_heightfar0 / float(max_heightfar1)

    return scale_close, scale_far


def apply_transformation(keypoints, translation, scale):
    """Applies normalization transformation (scale + transform) to keypoints"""
    i = 0
    while i < len(keypoints):
        # Modify the x-coordinate, then the y-coordinate. Skip confidence value.
        keypoints[i] = (keypoints[i] * scale) + translation[0]
        keypoints[i + 1] = (keypoints[i + 1] * scale) + translation[1]
        i += 3
    return keypoints


def calculate_translation(t_coord, translation, scale_ratios):
    """
    Inputs:
        t_coord: target coordinate (x,y)
        translation: [[t_close, t_far], [s_close, s_far]]
        scale_ratios: [scale_ratio_close, scale_ratio_far]
    """
    t_maxtoe, t_horizon = translation[0]
    s_maxtoe, s_horizon = translation[1]

    percentage = (t_coord - s_horizon) / float(s_maxtoe - s_horizon)
    m_coord = t_horizon + percentage * float(t_maxtoe - t_horizon)

    scale_interp = scale_ratios[1] + percentage * float(
        scale_ratios[0] - scale_ratios[1]
    )

    return m_coord - t_coord, scale_interp


def find_horizon(ank_ypos_max, ank_ypos_median, ank_ypos_list, mid_frac):
    """
    Returns horizon - offset from median to minimum y-position via clustering
    """
    dist_max_to_median = ank_ypos_max - ank_ypos_median
    dist_all_to_median = ank_ypos_median - np.array(ank_ypos_list)

    # Collect indices where
    #   (1) median is greater than ankle y-pos
    #   (2) dist. to median less than fraction of dist. from absolute max y-pos to median
    idx = np.where(
        (dist_all_to_median > 0) & (dist_all_to_median < mid_frac * dist_max_to_median)
    )
    horizon = ank_ypos_median - np.amax(dist_all_to_median[idx])

    return horizon


def transform_interp(
    mypath,
    scale_ratios,
    translation,
    myshape,
    savedir,
    spread_t,
    spread_s,
    dir_facepts="",
    framesdir="",
    numkeypoints=0,
    startname="frame",
):
    start = spread_s[0]
    end = spread_s[1]
    numberframesmade = 0

    startx = 0
    endx = 1920
    starty = 0
    endy = 1080
    step = 1

    get_facetexts = True
    saveim = False
    boxbuffer = 70

    tary = 512
    tarx = 1024

    # neck = 0
    # headNose = 18
    # rEye = 19
    # rEar = 20
    # lEye = 21
    # lEar = 22

    w_size = 7
    pose_window = []
    face_window = []
    rhand_window = []
    lhand_window = []

    realframes_window = []

    scaley = float(tary) / float(endy - starty)
    scalex = float(tarx) / float(endx - startx)

    my_neighbors = 0
    my_masks = 0
    mygraphs = 0
    posefaces = 0
    print(numkeypoints)
    if numkeypoints == 0:
        my_neighbors, my_masks, mygraphs, posefaces = readinfacepts(
            dir_facepts, spread_t, numcompare=100000
        )
        print("computed neighbors")

    n = start

    min_unset = True
    skipped = 0

    lastdiff = 0
    lastscale = 0

    noneighbors = []

    while n <= end:
        print(n)
        framesmadestr = "%06d" % numberframesmade
        string_num = "%06d" % n
        key_name = mypath + "/" + startname + string_num
        framenum = "%06d" % n
        frame_name = framesdir + "/" + startname + string_num + ".png"

        posepts = []

        ### try yaml
        posepts = readkeypointsfile(key_name + "_pose")
        facepts = readkeypointsfile(key_name + "_face")
        r_handpts = readkeypointsfile(key_name + "_hand_right")
        l_handpts = readkeypointsfile(key_name + "_hand_left")
        if posepts is None:  ## try json
            posepts, facepts, r_handpts, l_handpts = readkeypointsfile(
                key_name + "_keypoints"
            )
            if posepts is None:
                print("unable to read keypoints file")
                import sys

                sys.exit(0)

        startcanvas = 255 * np.ones(myshape, dtype="uint8")

        if len(posepts) == 75:
            posepts = map_25_to_23(posepts)

        if len(posepts) != poselen:
            print("EMPTY or more than one person")
        else:
            posepts = posepts[:poselen]
            check_me = get_pose_stats(posepts)

            if (not check_me) and min_unset:
                n += step
                continue
            if not check_me:
                skipped += 1
                diff = lastdiff
                scale = lastscale
                startcanvas = 255 * np.ones(myshape, dtype="uint8")
                print(key_name, "my pose is not so good")
            else:
                height, min_tip_toe, max_tip_toe = check_me
                diff, scale = calculate_translation(
                    max_tip_toe, translation, scale_ratios
                )
                lastdiff = diff
                lastscale = scale
                # print diff, scale
                startcanvas = 255 * np.ones(myshape, dtype="uint8")
            if min_unset:
                min_coords = get_min_point(posepts)
                min_coords = (myshape[1] // 2, min_coords[1])
                # min_coords = (min_coords[1], min_coords[0])
                print(min_coords)
                min_unset = False
                print("setting min")
            scaledcoords = (scale * min_coords[0], scale * min_coords[1])
            translateback = (
                min_coords[0] - scaledcoords[0],
                min_coords[1] - scaledcoords[1] + diff,
            )

            posepts = apply_transformation(posepts, translateback, scale)
            facepts = apply_transformation(facepts, translateback, scale)
            r_handpts = apply_transformation(r_handpts, translateback, scale)
            l_handpts = apply_transformation(l_handpts, translateback, scale)

            """ median """

            pose_window += [posepts]
            face_window += [facepts]
            rhand_window += [r_handpts]
            lhand_window += [l_handpts]

            if len(framesdir) > 0:
                realframes_window += [frame_name]

            if len(pose_window) >= w_size:
                h_span = w_size // 2

                med_posepts = getmedians_adapt(pose_window)
                med_facepts = getmedians_adapt(face_window)
                med_rhandpts = getmedians_adapt(rhand_window)
                med_lhandpts = getmedians_adapt(lhand_window, printme=False)

                canvas = renderpose(med_posepts, startcanvas)
                canvas = renderface_sparse(med_facepts, canvas, numkeypoints)
                canvas = renderhand(med_rhandpts, canvas)
                canvas = renderhand(med_lhandpts, canvas)

                canvas = canvas[starty:endy, startx:endx, [2, 1, 0]]
                canvas = Image.fromarray(canvas)

                canvas = canvas.resize((2 * SIZE, SIZE), Image.ANTIALIAS)
                canvas.save(savedir + "/test_label/frame" + framesmadestr + ".png")

                if len(framesdir) > 0:
                    savethisframe = realframes_window[h_span]
                    if os.path.isfile(savethisframe):
                        shutil.copy2(
                            savethisframe,
                            savedir + "/test_img/frame" + framesmadestr + ".png",
                        )  # complete target filename given
                        realframes_window = realframes_window[1:]
                    else:
                        print("no frame at" + savethisframe)

                pose_window = pose_window[1:]
                face_window = face_window[1:]
                rhand_window = rhand_window[1:]
                lhand_window = lhand_window[1:]

                if get_facetexts:
                    ave = aveface(med_posepts)

                    avex = ave[0]
                    avey = ave[1]

                    minx = int((max(avex - boxbuffer, startx) - startx) * scalex)
                    miny = int((max(avey - boxbuffer, starty) - starty) * scaley)
                    maxx = int((min(avex + boxbuffer, endx) - startx) * scalex)
                    maxy = int((min(avey + boxbuffer, endy) - starty) * scaley)

                    miny, maxy, minx, maxx = makebox128(miny, maxy, minx, maxx)

                    """ SAVE FACE TEXTS HERE """
                    myfile = (
                        savedir + "/test_facetexts128/frame" + framesmadestr + ".txt"
                    )
                    F = open(myfile, "w")
                    F.write(
                        str(miny) + " " + str(maxy) + " " + str(minx) + " " + str(maxx)
                    )
                    F.close()

                    if saveim:
                        oriImg = canvas[miny:maxy, minx:maxx, :]
                        oriImg = Image.fromarray(oriImg)
                        oriImg.save(
                            savedir + "/savefaces/frame" + framesmadestr + ".png"
                        )

                print(numberframesmade)

                numberframesmade += 1

        n += step
    print("num skipped = " + str(skipped))


opt = parser.parse_args()

shape1 = tuple(opt.target_shape)
shape2 = tuple(opt.source_shape)

target_keypoints = opt.target_keypoints
source_keypoints = opt.source_keypoints
framesdir = opt.source_frames

spread_t = tuple(opt.target_spread)
spread_s = tuple(opt.source_spread)

if (len(spread_t) != 2) or (len(spread_s) != 2):
    print("spread must include start and end range indices")
    sys.exit(0)

startname = opt.filestart

# TODO: Why is this set to 8 and what does that mean???
numkeypoints = 8

savedir = opt.results

if not os.path.exists(savedir):
    os.makedirs(savedir)
if not os.path.exists(savedir + "/test_label"):
    os.makedirs(savedir + "/test_label")
if not os.path.exists(savedir + "/test_img"):
    os.makedirs(savedir + "/test_img")
if not os.path.exists(savedir + "/test_facetexts128"):
    os.makedirs(savedir + "/test_facetexts128")

t_mid_frac = opt.target_median_frac
s_mid_frac = opt.source_median_frac

calculate_scale_and_translation = opt.calculate_scale_translation

scale = 1
translation = 0
""" Calculate Scale and Translation Here """
if calculate_scale_and_translation:
    # maxheight, mintoe, maxtoe, avemintoe, maxmintoe
    _, _, s_maxtoe, _, _, s_median, s_tiptoes, s_tiptoe_to_height = get_keypoints_stats(
        source_keypoints, shape2, spread_s, startname=startname, max_frames=10000
    )

    _, _, t_maxtoe, _, _, t_median, t_tiptoes, t_tiptoe_to_height = get_keypoints_stats(
        target_keypoints, shape1, spread_t, startname=startname, max_frames=10000
    )

    ## Find the far positions through clustering. Far position is spatially above in frame
    t_horizon = find_horizon(t_maxtoe, t_median, t_tiptoes, t_mid_frac)
    s_horizon = find_horizon(s_maxtoe, s_median, s_tiptoes, s_mid_frac)

    print("Target median: %.3f" % t_median)
    print("Target horizon: %.3f" % t_horizon)
    print("Source median: %.3f" % s_median)
    print("Source horizon: %.3f" % t_horizon)

    scale = 1
    # [[target_close, target_far], [source_close, source_far]]
    translation = [(t_maxtoe, t_horizon), (s_maxtoe, s_horizon)]

    if s_maxtoe - s_horizon < t_maxtoe - t_horizon:
        print("Source has smaller range than target")
        t_middle = 0.5 * (t_maxtoe + t_horizon)
        s_half = 0.5 * (s_maxtoe - s_horizon)
        new_t_horizon = t_middle - s_half
        new_t_maxtoe = t_middle + s_half
        translation = [(new_t_maxtoe, new_t_horizon), (s_maxtoe, s_horizon)]

    # We get the scale using the computed translation and the mappings from ankle position to height
    scale = get_minmax_scales(t_tiptoe_to_height, s_tiptoe_to_height, translation, 0.05)

    print("Scale: {}".format(scale))
    print("Translation: {}".format(translation))


#     """ SAVE FACE TEXTS HERE """
#     myfile = savedir + "/norm_params.txt"
#     F = open(myfile, "w")
#     F.truncate(0)
#     F.write(str(scale[0]) + " " + str(scale[1]) + "\n")
#     F.write(
#         str(translation[0][0])
#         + " "
#         + str(translation[0][1])
#         + " "
#         + str(translation[1][0])
#         + " "
#         + str(translation[1][1])
#     )
#     F.close()
# else:
#     norm_file = savedir + "/norm_params.txt"
#     if os.path.exists(norm_file):
#         with open(norm_file, "rb") as f:
#             try:
#                 line = f.readline()
#                 print(line)
#                 params = line.split(" ")
#                 scale = (float(params[0]), float(params[1]))
#                 line = f.readline()
#                 print(line)
#                 params = line.split(" ")
#                 print(params)
#                 translation = [
#                     (float(params[0]), float(params[1])),
#                     (float(params[2]), float(params[3])),
#                 ]
#             except:
#                 print(("unable to extract scale, translation from " + norm_file))
#                 sys.exit(0)

# print("transformation:")
# print(scale, translation)

# transform_interp(
#     source_keypoints,
#     scale,
#     translation,
#     shape1,
#     savedir,
#     spread_t,
#     spread_s,
#     "",
#     framesdir,
#     numkeypoints,
#     startname,
# )

