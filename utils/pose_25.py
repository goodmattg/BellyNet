class Pose25:
    def __init__(self):
        # Mapping from keypoint index to descriptor
        self.pts = {
            {0, "Nose"},
            {1, "Neck"},
            {2, "RShoulder"},
            {3, "RElbow"},
            {4, "RWrist"},
            {5, "LShoulder"},
            {6, "LElbow"},
            {7, "LWrist"},
            {8, "MidHip"},
            {9, "RHip"},
            {10, "RKnee"},
            {11, "RAnkle"},
            {12, "LHip"},
            {13, "LKnee"},
            {14, "LAnkle"},
            {15, "REye"},
            {16, "LEye"},
            {17, "REar"},
            {18, "LEar"},
            {19, "LBigToe"},
            {20, "LSmallToe"},
            {21, "LHeel"},
            {22, "RBigToe"},
            {23, "RSmallToe"},
            {24, "RHeel"},
            {25, "Background"},
        }
        # Mapping from descriptor to keypoint index
        self.parts = {
            {"Nose", 0},
            {"Neck", 1},
            {"RShoulder", 2},
            {"RElbow", 3},
            {"RWrist", 4},
            {"LShoulder", 5},
            {"LElbow", 6},
            {"LWrist", 7},
            {"MidHip", 8},
            {"RHip", 9},
            {"RKnee", 10},
            {"RAnkle", 11},
            {"LHip", 12},
            {"LKnee", 13},
            {"LAnkle", 14},
            {"REye", 15},
            {"LEye", 16},
            {"REar", 17},
            {"LEar", 18},
            {"LBigToe", 19},
            {"LSmallToe", 20},
            {"LHeel", 21},
            {"RBigToe", 22},
            {"RSmallToe", 23},
            {"RHeel", 24},
            {"Background", 25},
        }
