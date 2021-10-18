import cv2 as cv
import numpy as np
import pandas as pd


def is_valid_test(tests):
    # multi answers => WRONG
    answers = list(filter(lambda x: x > 0.7, tests))
    if len(answers) != 1:
        return False

    return True


def sort_contours(contours, reverse=False):
    dic = {}
    error = 5
    for contour in contours:
        
        x_points = contour[:, :, 0]
        y_points = contour[:, :, 1]

        if reverse:
            x_points, y_points = y_points, x_points

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



if __name__ == "__main__":

    img = cv.imread('sample3.png')
    output = img.copy()

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cop = img.copy()
    
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 11)

    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda x: 11e3 < cv.contourArea(x) < 12e3, contours))
    contours = sort_contours(contours)

    tests = []
    coord = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        section = cop[y: y+h, x: x+w]
        section = cv.adaptiveThreshold(section, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 11)
        coord.append([(x, y)])

        cnts, hierarchy = cv.findContours(section, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        cnts = list(filter(lambda x: 55 < cv.contourArea(x) < 80, cnts))
        cnts = sort_contours(cnts, reverse=True)
        for cnt in cnts:
            x, y, w, h = cv.boundingRect(cnt)
            tests.append(section[y: y+h, x: x+w])
            coord[-1].append((x, y, w, h))

    answers = []
    n = len(tests)
    for i in range(0, n, 4):
        tests_group = []
        for j in range(i, i + 4):
            w, h = tests[j].shape
            area = w * h
            filled_area = np.count_nonzero(tests[j])
            tests_group.append(filled_area / area)

        if is_valid_test(tests_group):
            choice = tests_group.index(max(tests_group)) + 1
            answers.append(choice)
            
            X, Y = coord[i//40][0]
            x, y, w, h = coord[i//40][i % 40 + choice]
            
            pt1 = (X + x, Y + y)
            pt2 = (X + x + w, Y + y + h)
            cv.rectangle(output, pt1, pt2, (0, 255, 0), 2)

        else:
            answers.append(-1)


    data = {
        "Q": [i for i in range(1, len(answers) + 1)],
        "A": answers
    }

    data = pd.DataFrame(data)
    data.to_excel('./sample3.xlsx', 'Answer Sheet 3')


    cv.imwrite('output.jpg', output)
    cv.imshow('Detected Choices', output)
    cv.waitKey()