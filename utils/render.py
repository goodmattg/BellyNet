import cv2 as cv
import numpy as np

import math
import time
import copy
import matplotlib

#%matplotlib inline
import pylab as plt
import json
from PIL import Image
from shutil import copyfile

# from skimage import img_as_float
from math import sqrt
from functools import reduce

# from pose_object import Pose
import os

# canvas = renderpose(ave_posepts, 255 * np.ones(myshape, dtype='uint8'))


def readkeypointsfile(myfile):
    """ supports only .yml and .json files """
    # print(myfile)
    filename, file_extension = os.path.splitext(myfile)
    # print myfile
    if len(file_extension) == 0:
        if os.path.exists(myfile + ".yml"):
            return readkeypointsfile_yml(myfile + ".yml")
        elif os.path.exists(myfile + ".json"):
            return readkeypointsfile_json(myfile + ".json")
        elif os.path.exists(myfile + ".JSON"):
            return readkeypointsfile_json(myfile + ".JSON")
        # else:
        # 	print("filename extension is not .yml or .json")
        # 	import sys
        # 	sys.exit(1)
        return None
    else:
        if file_extension == ".yml":
            return readkeypointsfile_yml(myfile)
        elif file_extension == ".json":
            return readkeypointsfile_json(myfile)
        elif file_extension == ".JSON":
            return readkeypointsfile_json(myfile)
        # else:
        # 	print("filename extension is not .yml or .json")
        # 	import sys
        # 	sys.exit(1)
        return None


def readkeypointsfile_yml(myfile):
    thefile = open(myfile, "r")
    a = thefile.readlines()
    # print a
    # if len(a) == 2:
    # 	return []
    leftovers = []
    for l in a:
        leftovers += [l.rstrip().lstrip()]

    find_data = [x.startswith("data") for x in leftovers]

    data_ind = np.where(np.array(find_data))[0][0]

    leftovers = leftovers[data_ind:]

    if len(leftovers) == 0:
        return []

    datastr = reduce(lambda x, y: x + y, leftovers)
    datastr = datastr.replace("\n", "")
    bigstring = datastr[7 : len(datastr) - 2]
    coords = [float(x.strip()) for x in bigstring.split(",")]
    return coords


def readkeypointsfile_json(myfile):
    import json

    f = open(myfile, "r")
    json_dict = json.load(f)
    people = json_dict["people"]
    posepts = []
    facepts = []
    r_handpts = []
    l_handpts = []
    for p in people:
        posepts += p["pose_keypoints_2d"]
        facepts += p["face_keypoints_2d"]
        r_handpts += p["hand_right_keypoints_2d"]
        l_handpts += p["hand_left_keypoints_2d"]

    return posepts, facepts, r_handpts, l_handpts


def map_25_to_23(posepts):
    if len(posepts) != 75:
        return posepts
    posepts = np.array(posepts)
    posepts23 = np.zeros(69)
    mapping = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        22,
        23,
        12,
        13,
        14,
        19,
        20,
        0,
        15,
        17,
        16,
        18,
    ]
    posepts23[0::3] = posepts[0::3][mapping]
    posepts23[1::3] = posepts[1::3][mapping]
    posepts23[2::3] = posepts[2::3][mapping]
    return posepts23


def scale_resize(curshape, myshape=(1080, 1920, 3), mean_height=0.0):

    if curshape == myshape:
        return None

    x_mult = myshape[0] / float(curshape[0])
    y_mult = myshape[1] / float(curshape[1])

    if x_mult == y_mult:
        # just need to scale
        return x_mult, (0.0, 0.0)
    elif y_mult > x_mult:
        ### scale x and center y
        y_new = x_mult * float(curshape[1])
        translate_y = (myshape[1] - y_new) / 2.0
        return x_mult, (translate_y, 0.0)
    ### x_mult > y_mult
    ### already in landscape mode scale y, center x (rows)

    x_new = y_mult * float(curshape[0])
    translate_x = (myshape[0] - x_new) / 2.0

    return y_mult, (0.0, translate_x)


def fix_scale_image(image, scale, translate, myshape):
    M = np.float32([[scale, 0, translate[0]], [0, scale, translate[1]]])
    dst = cv.warpAffine(image, M, (myshape[1], myshape[0]))
    return dst


def fix_scale_coords(points, scale, translate):
    points = np.array(points)

    points[0::3] = scale * points[0::3] + translate[0]
    points[1::3] = scale * points[1::3] + translate[1]

    return list(points)


def makebox128(miny, maxy, minx, maxx, dimy=128, dimx=128):
    diffy = maxy - miny
    diffx = maxx - minx
    # print "diffyb", maxy - miny
    # print "diffxb", maxx - minx
    if diffy != dimy:
        howmuch = dimy - diffy

        maxy = maxy + (howmuch // 2)
        miny = maxy - dimy

        if maxy > 512:
            maxy = 512
            miny = 512 - dimy
        roomtoedge = miny
        if miny < 0:
            miny = 0
            maxy = dimy
    if diffx != dimx:
        howmuch = dimx - diffx

        maxx = maxx + (howmuch // 2)
        minx = maxx - dimx

        if maxx > 1024:
            maxx = 1024
            minx = 1024 - dimx
        roomtoedge = minx
        if minx < 0:
            minx = 0
            maxx = dimx

    # print "diffy", maxy - miny
    # print "diffx", maxx - minx
    return miny, maxy, minx, maxx


def renderposeCOCO(posepts, canvas, keyname=""):
    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]
    orgImgshape = canvas.shape

    i = 0
    while i < 54:
        confidence = posepts[i + 2]
        if confidence > 0:
            cv.circle(
                canvas,
                (int(posepts[i]), int(posepts[i + 1])),
                8,
                tuple(colors[i // 3]),
                thickness=-1,
            )
        i += 3

    limbSeq = [
        [1, 2],
        [1, 5],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
        [1, 8],
        [8, 9],
        [9, 10],
        [1, 11],
        [11, 12],
        [12, 13],
        [1, 0],
        [0, 14],
        [14, 16],
        [0, 15],
        [15, 17],
    ]  # , [2, 16], [5, 17]]

    stickwidth = 4

    for k in range(len(limbSeq)):
        firstlimb_ind = limbSeq[k][0]
        secondlimb_ind = limbSeq[k][1]

        if (posepts[3 * firstlimb_ind + 2] > 0) and (
            posepts[3 * secondlimb_ind + 2] > 0
        ):
            cur_canvas = canvas.copy()
            Y = [posepts[3 * firstlimb_ind], posepts[3 * secondlimb_ind]]
            X = [posepts[3 * firstlimb_ind + 1], posepts[3 * secondlimb_ind + 1]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv.fillConvexPoly(cur_canvas, polygon, colors[firstlimb_ind])
            canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas


def renderpose25(posepts, canvas):
    """ FILL THIS IN """
    colors = [
        [255, 0, 85],
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [255, 0, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [255, 0, 170],
        [170, 0, 255],
        [255, 0, 255],
        [85, 0, 255],
        [0, 0, 255],
        [0, 0, 255],
        [0, 0, 255],
        [0, 255, 255],
        [0, 255, 255],
        [0, 255, 255],
    ]

    i = 0
    while i < 25 * 3:
        confidence = posepts[i + 2]
        if confidence > 0:
            cv.circle(
                canvas,
                (int(posepts[i]), int(posepts[i + 1])),
                8,
                tuple(colors[i // 3]),
                thickness=-1,
            )
        i += 3

    limbSeq = [
        [0, 1],
        [0, 15],
        [0, 16],
        [1, 2],
        [1, 5],
        [1, 8],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
        [8, 9],
        [8, 12],
        [9, 10],
        [10, 11],
        [11, 22],
        [11, 24],
        [12, 13],
        [13, 14],
        [14, 19],
        [14, 21],
        [15, 17],
        [16, 18],
        [19, 20],
        [22, 23],
    ]

    stickwidth = 4

    for k in range(len(limbSeq)):
        firstlimb_ind = limbSeq[k][0]
        secondlimb_ind = limbSeq[k][1]

        if (posepts[3 * firstlimb_ind + 2] > 0) and (
            posepts[3 * secondlimb_ind + 2] > 0
        ):
            cur_canvas = canvas.copy()
            Y = [posepts[3 * firstlimb_ind], posepts[3 * secondlimb_ind]]
            X = [posepts[3 * firstlimb_ind + 1], posepts[3 * secondlimb_ind + 1]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv.fillConvexPoly(cur_canvas, polygon, colors[firstlimb_ind])
            canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas


def renderpose(posepts, canvas):
    poselen = len(posepts) // 3
    if poselen == 18:
        return renderposeCOCO(posepts, canvas)
    elif poselen == 23:
        return renderpose23(posepts, canvas)
    elif poselen == 25:
        return renderpose25(posepts, canvas)
    print(("Pose Length of " + str(poselen) + " is not supported"))
    import sys

    sys.exit(1)


def renderface(facepts, canvas, disp=False, threshold=0.2, smalldot=2):
    if disp:
        color = tuple([255, 255, 255])
    else:
        color = tuple([0, 0, 0])

    avecons = sum(facepts[2 : len(facepts) : 3]) / 70.0

    if avecons < threshold:
        return canvas
    i = 0

    while i < 210:
        confidence = facepts[i + 2]
        if confidence > 0:
            cv.circle(
                canvas,
                (int(facepts[i]), int(facepts[i + 1])),
                smalldot,
                color,
                thickness=-1,
            )
        i += 3

    if disp:  # graph the lines between points
        stickwidth = 1
        linearSeq = [
            list(range(0, 16 + 1)),
            list(range(17, 21 + 1)),
            list(range(22, 26 + 1)),
            list(range(27, 30 + 1)),
            list(range(31, 35 + 1)),
        ]
        circularSeq = [
            list(range(36, 41 + 1)),
            list(range(42, 47 + 1)),
            list(range(48, 59 + 1)),
            list(range(60, 67)),
        ]

        for line in linearSeq:
            for step in line:
                if step != line[len(line) - 1]:
                    firstlimb_ind = step
                    secondlimb_ind = step + 1
                    if (facepts[3 * firstlimb_ind + 2] > 0) and (
                        facepts[3 * secondlimb_ind + 2] > 0
                    ):
                        cur_canvas = canvas.copy()
                        Y = [facepts[3 * firstlimb_ind], facepts[3 * secondlimb_ind]]
                        X = [
                            facepts[3 * firstlimb_ind + 1],
                            facepts[3 * secondlimb_ind + 1],
                        ]
                        mX = np.mean(X)
                        mY = np.mean(Y)
                        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                        polygon = cv.ellipse2Poly(
                            (int(mY), int(mX)),
                            (int(length / 2), stickwidth),
                            int(angle),
                            0,
                            360,
                            1,
                        )
                        cv.fillConvexPoly(cur_canvas, polygon, color)
                        canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        for circle in circularSeq:
            for step in circle:
                if step == circle[len(circle) - 1]:
                    firstlimb_ind = step
                    secondlimb_ind = circle[0]
                else:
                    firstlimb_ind = step
                    secondlimb_ind = step + 1

                if (facepts[3 * firstlimb_ind + 2] > 0) and (
                    facepts[3 * secondlimb_ind + 2] > 0
                ):
                    cur_canvas = canvas.copy()
                    Y = [facepts[3 * firstlimb_ind], facepts[3 * secondlimb_ind]]
                    X = [
                        facepts[3 * firstlimb_ind + 1],
                        facepts[3 * secondlimb_ind + 1],
                    ]
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                    polygon = cv.ellipse2Poly(
                        (int(mY), int(mX)),
                        (int(length / 2), stickwidth),
                        int(angle),
                        0,
                        360,
                        1,
                    )
                    cv.fillConvexPoly(cur_canvas, polygon, color)
                    canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas


def renderface_sparse(
    facepts, canvas, numkeypoints, disp=False, threshold=0.2, smalldot=4
):
    if numkeypoints == 0:
        return renderface(facepts, myshape, canvas, disp, threshold, getave)
    if disp:
        color = tuple([255, 255, 255])
    else:
        color = tuple([0, 0, 0])

    avecons = sum(facepts[2 : len(facepts) : 3]) / 70.0
    if avecons < threshold:
        return canvas

    pointlist = [27, 30, 8, 0, 16, 33, 68, 69]  # sparse 8 default
    if numkeypoints == 22:
        pointlist = [
            27,
            30,
            8,
            0,
            16,
            31,
            33,
            35,
            68,
            69,
            36,
            39,
            42,
            45,
            17,
            21,
            22,
            26,
            48,
            51,
            54,
            57,
        ]  # endpoints
    elif numkeypoints == 9:
        pointlist += [62]

    for i in pointlist:
        point = 3 * i
        confidence = facepts[point + 2]
        if confidence > 0:
            cv.circle(
                canvas,
                (int(facepts[point]), int(facepts[point + 1])),
                smalldot,
                color,
                thickness=-1,
            )

    return canvas


def renderhand(handpts, canvas, threshold=0.05):
    colors = [
        [230, 53, 40],
        [231, 115, 64],
        [233, 136, 31],
        [213, 160, 13],
        [217, 200, 19],
        [170, 210, 35],
        [139, 228, 48],
        [83, 214, 45],
        [77, 192, 46],
        [83, 213, 133],
        [82, 223, 190],
        [80, 184, 197],
        [78, 140, 189],
        [86, 112, 208],
        [83, 73, 217],
        [123, 46, 183],
        [189, 102, 255],
        [218, 83, 232],
        [229, 65, 189],
        [236, 61, 141],
        [255, 102, 145],
    ]

    i = 0
    while i < 63:
        confidence = handpts[i + 2]
        if confidence > threshold:
            cv.circle(
                canvas,
                (int(handpts[i]), int(handpts[i + 1])),
                3,
                tuple(colors[i // 3]),
                thickness=-1,
            )
        i += 3

    stickwidth = 2
    linearSeq = [
        list(range(1, 4 + 1)),
        list(range(5, 8 + 1)),
        list(range(9, 12 + 1)),
        list(range(13, 16 + 1)),
        list(range(17, 20 + 1)),
    ]
    for line in linearSeq:
        for step in line:
            if step != line[len(line) - 1]:
                firstlimb_ind = step
                secondlimb_ind = step + 1
            else:
                firstlimb_ind = 0
                secondlimb_ind = line[0]
            if (handpts[3 * firstlimb_ind + 2] > threshold) and (
                handpts[3 * secondlimb_ind + 2] > threshold
            ):
                cur_canvas = canvas.copy()
                Y = [handpts[3 * firstlimb_ind], handpts[3 * secondlimb_ind]]
                X = [handpts[3 * firstlimb_ind + 1], handpts[3 * secondlimb_ind + 1]]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv.ellipse2Poly(
                    (int(mY), int(mX)),
                    (int(length / 2), stickwidth),
                    int(angle),
                    0,
                    360,
                    1,
                )
                cv.fillConvexPoly(cur_canvas, polygon, colors[secondlimb_ind])
                canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas

