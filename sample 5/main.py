import cv2 as cv
import numpy as np


def is_valid_test(tests):
    # multi answers => WRONG
    answers = list(filter(lambda x: x > 0.3, tests))
    return len(answers) == 1



def sort_contours(contours):
    dic = {}
    error = 25
    for contour in contours:
        
        x_points = contour[:, :, 0]
        y_points = contour[:, :, 1]

        key = min(y_points)[0]
        min_key = list(filter(lambda x: -error < x - key < error, dic.keys()))
        
        if min_key:
            key = min_key[0]
            dic[key].append((min(x_points)[0], contour))
        else:
            dic[key] = [(min(x_points)[0], contour)]

    dic = dict(sorted(dic.items()))
    print(dic.keys())
        
    for key in dic.keys():
        dic[key] = list(sorted(dic[key]))

    return [tup[1] for sublist in dic.values() for tup in sublist]



def fix_rotation(img):
    rows, cols = img.shape[:2]
    img_copy = img.copy()

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 11)
    
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    roi = max(contours, key=cv.contourArea)

    perimeter = cv.arcLength(roi, True)
    corners = cv.approxPolyDP(roi, perimeter // 10, True)
    corners = np.vstack(corners)
    corners = sorted(corners, key=lambda corner: corner[0])
    
    if corners[0][1] > corners[1][1]:
        corners[0], corners[1] = corners[1], corners[0]
    if corners[2][1] > corners[3][1]:
        corners[2], corners[3] = corners[3], corners[2]
    corners[1], corners[2] = corners[2], corners[1]

    src = np.float32(corners)
    dst = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

    projective_matrix = cv.getPerspectiveTransform(src, dst)
    rotated_img = cv.warpPerspective(img_copy, projective_matrix, (cols, rows))
    
    return rotated_img[5: rows - 5, 5: cols - 5]



if __name__ == "__main__":
    img = cv.imread('sample5_2.jpg')
    img = cv.resize(img, None, fx=.5, fy=.5)

    roi = fix_rotation(img)
    output = roi.copy()

    roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    roi = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 3)

    contours, hierarchy = cv.findContours(roi, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda x: 8e3 < cv.contourArea(x) < 1e4, contours))
    contours = sort_contours(contours)
    
    coord = []
    answers = []
    for i in range(0, len(contours), 5):
        filled_percentages = []

        for j in range(i, i + 5):
            x, y, w, h = cv.boundingRect(contours[j])
            choice = roi[y: y+h, x: x+w]
            coord.append((x, y, w, h))

            area = w * h
            filled_area = np.count_nonzero(choice)
            filled_percentages.append(filled_area / area)

        if is_valid_test(filled_percentages.copy()):
            choice = filled_percentages.index(max(filled_percentages)) + 1
            answers.append(choice)

            x, y, w, h = coord[i + choice - 1]
            cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 5)
            
        else:
            answers.append(-1)


    print(answers)

    cv.imwrite('output.jpg', output)
    cv.imshow('Detected Choices', output)
    cv.waitKey()
    cv.destroyAllWindows()