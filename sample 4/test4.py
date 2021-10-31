import cv2 as cv
import numpy as np
import pandas as pd


def is_valid_test(tests):
    # multi answers => WRONG
    answers = list(filter(lambda x: x > 0.55, tests))
    if len(answers) != 1:
        return False

    return True


def sort_contours_vertically(contours):
    dic = {}
    for contour in contours:
        y_points = contour[:, :, 1]
        dic[min(y_points)[0]] = contour
    
    dic = dict(sorted(dic.items()))
    return dic.values()


# sorting right align
def sort_contours_horizontally(contours):
    dic = {}
    for contour in contours:
        x_points = contour[:, :, 0]
        dic[min(x_points)[0]] = contour
    
    dic = dict(sorted(dic.items()))
    return list(reversed(list(dic.values())))


if __name__ == "__main__":

    img = cv.imread('test4.jpg', 0)
    cop = img.copy()

    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 11)
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 4))
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

    contours, hierarchy = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda x: 3e4 < cv.contourArea(x) < 4e4, contours))
    contours = sort_contours_horizontally(contours)

    test_groups = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        section = cop[y: y+h, x: x+w]
        part = section.copy()
        section = cv.adaptiveThreshold(section, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 11)

        
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
        section = cv.morphologyEx(section, cv.MORPH_OPEN, kernel)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (19, 1))
        section = cv.dilate(section, kernel)

        cnts, hierarchy = cv.findContours(section, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        cnts = list(filter(lambda x: 250 < cv.contourArea(x) < 650, cnts))
        cnts = sort_contours_vertically(cnts)
        for cnt in cnts:
            x, y, w, h = cv.boundingRect(cnt)
            group = part[y: y+h, x: x+w]
            group = cv.adaptiveThreshold(group, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 11)
            
            x_axis = [55, 38, 20, 3]
            w, h = 16, 10

            group = list(map(lambda x: group[0: h, x: x + w], x_axis))
            test_groups.append(group)

        
    answers = []
    for i, group in enumerate(test_groups):
        filled_percentages = []

        for choice in group:
            w, h = choice.shape
            area = w * h
            filled_area = np.count_nonzero(choice)
            filled_percentages.append(filled_area / area)

        if is_valid_test(filled_percentages):
            choice = filled_percentages.index(max(filled_percentages)) + 1
            answers.append(choice)
        else:
            answers.append(-1)

    data = {
        "Q": [i for i in range(1, len(answers) + 1)],
        "A": answers
    }

    data = pd.DataFrame(data)
    data.to_excel('./sample4.xlsx', 'Answer Sheet 4')