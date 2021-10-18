import cv2 as cv
import numpy as np
import pandas as pd


def sort_contours_vertically(contours):
    dic = {}
    for contour in contours:
        y_points = contour[:, :, 1]
        dic[min(y_points)[0]] = contour
    
    dic = dict(sorted(dic.items()))
    return dic.values()

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

    img = cv.imread('sample2.jpg', 0)
    cop = img.copy()
    
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 11)
    
    contours, hierarchy = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    contours = list(filter(lambda x: 13e3 < cv.contourArea(x) < 14e3, contours))
    contours = sort_contours(contours)
    sections = []
    bound = 5
    for i in range(len(contours)):
        if i % 2:
            x, y, w, h = cv.boundingRect(contours[i])
            sections.append(((x, y), img[y+bound: y+h-bound, x+bound: x+w-bound]))
        
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (11, 1))
    test_groups = []
    for part in sections:
        (X, Y), section = part
        section = cv.dilate(section, kernel)
        contours, hierarchy = cv.findContours(section, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        contours = list(filter(lambda x: 400 < cv.contourArea(x) < 800, contours))
        contours = sort_contours_vertically(contours)

        for c in contours:
            x, y, w, h = cv.boundingRect(c)
            test = cop[y + Y + bound: y + h + Y + bound , x + X: x + w + X + bound]
            test_groups.append(test)

    tests = []
    for group in test_groups:
        group_copy = group.copy()
        group = cv.adaptiveThreshold(group, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 11)
        contours, hierarchy = cv.findContours(group, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        choices = []
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            choices.append((w * h, contour))

        choices = list(sorted(choices, key=lambda x: x[0]))
        choices = choices[-4:]
        choices = list(map(lambda x: x[1], choices))
        choices = list(sort_contours_horizontally(choices))
        
        # we got only 4 tests
        for i in range(4):
            x, y, w, h = cv.boundingRect(choices[i])
            choices[i] = cv.adaptiveThreshold(group_copy[y:y+h, x:x+w], 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 11)

        tests.append(choices)

    for i, tests_group in enumerate(tests):
        for j, test in enumerate(tests_group):  # each test is a contour
            w, h = test.shape
            area = w * h
            filled_area = np.count_nonzero(test)
            tests_group[j] = filled_area / area

        if is_valid_test(tests_group):
            tests[i] = tests_group.index(max(tests_group)) + 1
        else:
            tests[i] = -1

    data = {
        "Q": [i for i in range(1, len(tests) + 1)],
        "A": tests
    }

    data = pd.DataFrame(data)
    data.to_excel('./sample2.xlsx', 'Answer Sheet 2')