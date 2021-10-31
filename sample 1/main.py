import cv2 as cv
import numpy as np
import pandas as pd
import heapq


def sort_contours_horizontally(contours):
    dic = {}
    for contour in contours:
        x_points = contour[:, :, 0]
        dic[min(x_points)[0]] = contour
    
    dic = dict(sorted(dic.items()))
    return dic.values()

def sort_contours(contours):
    dic = {}
    error = 5
    for contour in contours:
        
        x_points = contour[:, :, 0]
        y_points = contour[:, :, 1]
        key = min(x_points)[0]
        min_key = list(filter(lambda x: -error < x - key < error, dic.keys()))
        
        if min_key:
            key = min_key[0]
            dic[key].append((min(y_points)[0], contour))
        else:
            dic[key] = [(min(y_points)[0], contour)]

    dic = dict(sorted(dic.items()))
        
    for key in dic.keys():
        dic[key] = list(sorted(dic[key]))

    return [tup[1] for sublist in dic.values() for tup in sublist]



def is_valid_test(tests):
    # multi answers => WRONG
    answers = list(filter(lambda x: x > 0.7, tests))
    if len(answers) != 1:
        return False

    return True



if __name__ == "__main__":

    img = cv.imread('sample1.jpg')
    cop = img.copy()
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 10)
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 1))
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

    
    contours, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    contours = list(filter(lambda x: 300 < cv.contourArea(x) < 450, contours))
    contours = sort_contours(contours)

    answers = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv.boundingRect(contour)
        roi = cv.cvtColor(cop[y: y+h, x: x + w], cv.COLOR_BGR2GRAY)
        roi_cop = roi.copy()
        roi = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 10)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 3))
        roi = cv.morphologyEx(roi, cv.MORPH_CLOSE, kernel)
        
        cnts, hierarchy = cv.findContours(roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cnts = list(filter(cv.contourArea, cnts))
        cnts = sort_contours_horizontally(cnts)

        tests = list(map(cv.boundingRect, cnts))
        coord = [(x, y)]
        for j, test in enumerate(tests):  # each test is a contour
            coord.append(test)
            x, y, w, h = test
            area = w * h
            filled_area = np.count_nonzero(roi[y: y+h, x: x+w])
            tests[j] = filled_area / area

        if is_valid_test(tests):
            choice = tests.index(max(tests)) + 1
            answers.append(choice)

            X, Y = coord[0]
            x, y, w, h = coord[choice]
            
            pt1 = (X + x, Y + y)
            pt2 = (X + x + w, Y + y + h)
            cv.rectangle(cop, pt1, pt2, (0, 255, 0), 2)

        else:
            answers.append(-1)

    for i in range(len(answers)):
        print(i + 1, ":", answers[i])

    data = {
        "Q": [i for i in range(1, len(answers) + 1)],
        "A": answers
    }

    data = pd.DataFrame(data)
    data.to_excel('./sample1.xlsx', 'Answer Sheet 1')

    cv.imwrite('output.jpg', cop)
    cv.imshow('Detected Choices', cop)
    cv.waitKey()