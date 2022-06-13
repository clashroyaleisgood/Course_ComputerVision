import cv2
import numpy as np

def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ", img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def kNN(featureImage1, featureImage2, threshold=2):
    '''
    calc 2-NN from f1 to f2
    do Lowe's Ratio test
        to eliminate bad pairs
    dist1 < 0.75 * dist2 -> good
    if dist2 < 1.33 * dist1 -> bad
    
    feature1: [
        [point1 feature],
        [point2 feature],...
    ]
    feature2: same as feature1
    return good matches [[p1 in f1, p2 in f2], []]
    '''
    good_matches = []
    for idx1, feat1 in enumerate(featureImage1):
        min_dist = 100000000
        min_idx2 = None
        distances = dist(feat1, featureImage2)
        for idx2, distance in enumerate(distances):
            if distance < min_dist:
                min_dist = distance
                min_idx2 = idx2
        # already find min distance

        # dist1 < 0.75 * dist2 -> good
        # if dist2 < 1.33 * dist1 -> bad
        find_close_dist2 = False
        thresold_dist2 = threshold * min_dist
        for idx2, distance in enumerate(distances):
            if distance != min_dist and \
               distance < thresold_dist2:
                find_close_dist2 = True
                break

        # for idx2, feat2 in enumerate(featureImage2):
        #     distance = dist(feat1, feat2)
        #     if distance < min_dist:
        #         min_dist = distance
        #         min_idx2 = idx2
        # already find min distance

        # # dist1 < 0.75 * dist2 -> good
        # # if dist2 < 1.33 * dist1 -> bad
        # find_close_dist2 = False
        # thresold_dist2 = threshold * min_dist
        # for idx2, feat2 in enumerate(featureImage2):
        #     distance = dist(feat1, feat2)
        #     if distance != min_dist and \
        #        distance < thresold_dist2:
        #         find_close_dist2 = True
        #         break

        if not find_close_dist2:
            good_matches += [[idx1, min_idx2]]
            print(f'add{idx1, min_idx2}')
    return good_matches

def dist(feature1, featureImage2):
    '''
    return distance from f1 to each featureImage2
    '''
    distance1to2 = feature1 - featureImage2
    return np.linalg.norm(distance1to2, axis=1)


# new version of RANSAC
# return good matches
def RANSAC(matches, kp1, kp2, threshold=3, repeat=200):
    '''
    random choose 4 from matches
    do PrespectiveTransform(all matches)
        and counting supports
    return best Transform Mat
    '''
    max_support = 0
    max_H = None
    
    for r in range(repeat):
        rand4number = np.random.choice(len(matches), 4)
        # print(f'try {r}: rand select {rand4number}')
        match4 = []
        for i in range(4):
            idx = rand4number[i]
            m = [
                kp1[matches[idx][0]].pt,
                kp2[matches[idx][1]].pt
            ]  # [[x1, y1], [x1h, y1h]]
            match4 += [m]

        H = get_HomographyMatrix(match4)

        # correctness test
        # p1 = np.array([e for e in kp1[matches[rand4number[3]][0]].pt] + [1]).reshape((3, 1))
        # p2 = np.array([e for e in kp2[matches[rand4number[3]][1]].pt] + [1]).reshape((3, 1))
        # print(p1)
        # print(p2)
        # p3 = H @ p1
        # p3 /= p3[2]
        # print(p3)

        support = 0
        for idx1, idx2 in matches:
            p1 = np.array([*(kp1[idx1].pt), 1])  # [x, y, 1]
            p2_proj = H @ p1
            p2_proj = p2_proj[:2] / p2_proj[2]  # [wx, wy, w] -> [x, y]

            p2 = np.array(kp2[idx2].pt)

            distance = np.linalg.norm(p2 - p2_proj)

            if distance < threshold:
                support += 1
        # end calc suport
        if support > max_support:
            max_support = support
            max_H = H
        print(f'try {r}: support = {support}')
    print("support for H", max_support)

    # DIFFERENCES in rectification
    support_matches = []
    for idx1, idx2 in matches:
        p1 = np.array([*(kp1[idx1].pt), 1])  # [x, y, 1]
        p2_proj = max_H @ p1
        p2_proj = p2_proj[:2] / p2_proj[2]  # [wx, wy, w] -> [x, y]

        p2 = np.array(kp2[idx2].pt)

        distance = np.linalg.norm(p2 - p2_proj)

        if distance < threshold:
            support_matches.append(
                [kp1[idx1].pt, kp2[idx2].pt]  # [p1, p2]
            )

    return support_matches

def get_HomographyMatrix(matches4):
    '''
    M: [m1, m2, m3, m4]
    mi_1~4: [[x1, y1], [x1h, y1h]]
    return homography matrix
    '''

    mat_A = []
    for match in matches4:
        (x1, y1), (x1_p, y1_p) = match
        mat_A += [
            [x1, y1,  1,  0,  0,  0, -x1_p * x1, -x1_p * y1, -x1_p],
            [ 0,  0,  0, x1, y1,  1, -y1_p * x1, -y1_p * y1, -y1_p]
        ]

    u, s, vh = np.linalg.svd(mat_A)

    H = vh[8]  # The "rows" of vh are the eigenvectors of ATA
    H = H / H[8]  # normalize A33 to 1
    H = H.reshape((3, 3))
    return H

