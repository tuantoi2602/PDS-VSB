import multiprocessing

import cv2
import numpy as np
import math
import time
from multiprocessing import Process
import threading


def show(img, name='img'):
    maxHeight = 540
    maxWidth = 960
    scaleX = maxWidth / img.shape[1]
    scaleY = maxHeight / img.shape[0]
    scale = min(scaleX, scaleY)
    if scale < 1:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    # cv2.imshow(name, img)
    # cv2.waitKey(0)
    # cv2.destroyWindow(name)


def convert_img_to_binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    return binary_img


def getContours(img):
    binary_img = convert_img_to_binary(img)
    thresholdImage = cv2.Canny(binary_img, 100, 200)
    contours, hierarchy = cv2.findContours(
        thresholdImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return thresholdImage, contours, hierarchy


def checkRatioOfContours(index, contours, hierarchy):
    firstChildIndex = hierarchy[0][index][2]
    secondChildIndex = hierarchy[0][firstChildIndex][2]
    firstArea = cv2.contourArea(contours[index]) / (
            cv2.contourArea(contours[firstChildIndex]) + 1e-5)
    secondArea = cv2.contourArea(contours[firstChildIndex]) / (
            cv2.contourArea(contours[secondChildIndex]) + 1e-5)
    return ((firstArea / (secondArea + 1e-5)) > 1 and
            ((firstArea / (secondArea + 1e-5)) < 10))


def isPossibleCorner(contourIndex, levelsNum, contours, hierarchy):
    chirldIdx = hierarchy[0][contourIndex][2]
    level = 0
    while chirldIdx != -1:
        level += 1
        chirldIdx = hierarchy[0][chirldIdx][2]
    if level >= levelsNum:
        return checkRatioOfContours(contourIndex, contours, hierarchy)
    return False


def getContourWithinLevel(levelsNum, contours, hierarchy):
    patterns = []
    patternsIndices = []
    for contourIndex in range(len(contours)):
        if isPossibleCorner(contourIndex, levelsNum, contours, hierarchy):
            patterns.append(contours[contourIndex])
            patternsIndices.append(contourIndex)
    return patterns, patternsIndices


def isParentInList(intrestingPatternIdList, index, hierarchy):
    parentIdx = hierarchy[0][index][3]
    while (parentIdx != -1) and (parentIdx not in intrestingPatternIdList):
        parentIdx = hierarchy[0][parentIdx][3]
    return parentIdx != -1


def getOrientation(contours, centerOfMassList):
    distance_AB = np.linalg.norm(centerOfMassList[0].flatten() - centerOfMassList[1].flatten(), axis=0)
    distance_BC = np.linalg.norm(centerOfMassList[1].flatten() - centerOfMassList[2].flatten(), axis=0)
    distance_AC = np.linalg.norm(centerOfMassList[0].flatten() - centerOfMassList[2].flatten(), axis=0)

    largestLine = np.argmax(
        np.array([distance_AB, distance_BC, distance_AC]))
    bottomLeftIdx = 0
    topLeftIdx = 1
    topRightIdx = 2
    if largestLine == 0:
        bottomLeftIdx, topLeftIdx, topRightIdx = 0, 2, 1
    if largestLine == 1:
        bottomLeftIdx, topLeftIdx, topRightIdx = 1, 0, 2
    if largestLine == 2:
        bottomLeftIdx, topLeftIdx, topRightIdx = 0, 1, 2

    slope = (centerOfMassList[bottomLeftIdx][1] - centerOfMassList[topRightIdx][1]) / (
            centerOfMassList[bottomLeftIdx][0] - centerOfMassList[topRightIdx][0] + 1e-5)
    coefficientA = -slope
    coefficientB = 1
    constant = slope * centerOfMassList[bottomLeftIdx][0] - centerOfMassList[bottomLeftIdx][1]
    distance = (coefficientA * centerOfMassList[topLeftIdx][0] + coefficientB * centerOfMassList[topLeftIdx][
        1] + constant) / (
                   np.sqrt(coefficientA ** 2 + coefficientB ** 2))

    pointList = np.zeros(shape=(3, 2))
    # O    O
    if (slope >= 0) and (distance >= 0):
        if (centerOfMassList[bottomLeftIdx][0] > centerOfMassList[topRightIdx][0]):
            pointList[1] = centerOfMassList[bottomLeftIdx]
            pointList[2] = centerOfMassList[topRightIdx]
        else:
            pointList[1] = centerOfMassList[topRightIdx]
            pointList[2] = centerOfMassList[bottomLeftIdx]
        ORIENTATION = "SouthWest"

    # O   O
    #
    #     O
    elif (slope > 0) and (distance < 0):

        if (centerOfMassList[bottomLeftIdx][1] > centerOfMassList[topRightIdx][1]):
            pointList[2] = centerOfMassList[bottomLeftIdx]
            pointList[1] = centerOfMassList[topRightIdx]
        else:
            pointList[2] = centerOfMassList[topRightIdx]
            pointList[1] = centerOfMassList[bottomLeftIdx]
        ORIENTATION = "NorthEast"


    #       O
    #
    # O     O
    elif (slope < 0) and (distance > 0):
        if (centerOfMassList[bottomLeftIdx][0] > centerOfMassList[topRightIdx][0]):
            pointList[1] = centerOfMassList[bottomLeftIdx]
            pointList[2] = centerOfMassList[topRightIdx]
        else:
            pointList[1] = centerOfMassList[topRightIdx]
            pointList[2] = centerOfMassList[bottomLeftIdx]
        ORIENTATION = "SouthEast"
    # O    O
    #
    # O
    elif (slope < 0) and (distance < 0):

        if (centerOfMassList[bottomLeftIdx][0] > centerOfMassList[topRightIdx][0]):
            pointList[2] = centerOfMassList[bottomLeftIdx]
            pointList[1] = centerOfMassList[topRightIdx]
        else:
            pointList[2] = centerOfMassList[topRightIdx]
            pointList[1] = centerOfMassList[bottomLeftIdx]
    pointList[0] = centerOfMassList[topLeftIdx]
    return pointList


def getCenterOfMass(contours):
    pointList = []
    for i in range(len(contours)):
        moment = cv2.moments(contours[i])
        centreOfMassX = int(moment['m10'] / moment['m00'])
        centreOfMassY = int(moment['m01'] / moment['m00'])
        pointList.append([centreOfMassX, centreOfMassY])
    return pointList


def lineAngle(line1, line2):
    return math.acos((line1[0] * line2[0] + line1[1] * line2[1]) /
                     (np.linalg.norm(line1) * np.linalg.norm(line2, axis=0)))


def selectPatterns(pointList):
    lineList = []
    for i in range(len(pointList)):
        for j in range(i, len(pointList)):
            lineList.append([i, j])
    finalResult = None
    minLengthDiff = -1
    for i in range(len(lineList)):
        for j in range(i, len(lineList)):
            line1 = np.array([pointList[lineList[i][0]][0] - pointList[lineList[i][1]][0],
                              pointList[lineList[i][0]][1] - pointList[lineList[i][1]][1]])
            line2 = np.array([pointList[lineList[j][0]][0] - pointList[lineList[j][1]][0],
                              pointList[lineList[j][0]][1] - pointList[lineList[j][1]][1]])
            pointIdxList = np.array([lineList[i][0], lineList[i][1], lineList[j][0], lineList[j][1]])
            pointIdxList = np.unique(pointIdxList)
            if len(pointIdxList) == 3:
                theta = lineAngle(line1, line2)
                if abs(math.pi / 2 - theta) < math.pi / 6:
                    lengthDiff = abs(np.linalg.norm(line1, axis=0) - np.linalg.norm(line2, axis=0))
                    if lengthDiff < minLengthDiff or minLengthDiff < 0:
                        minLengthDiff = abs(np.linalg.norm(line1, axis=0) - np.linalg.norm(line2, axis=0))
                        finalResult = pointIdxList

    return finalResult


def process(input_file_path, output_file_path):
    tstart = time.time()
    name = input_file_path.split('/')[0].replace(".png", "")
    path = input_file_path
    img = cv2.imread(path)
    # show(img)
    thresholdImage, contours, hierarchy = getContours(img)

    levelsNum = 3
    patterns, patternsIndices = getContourWithinLevel(levelsNum, contours, hierarchy)
    img_show = cv2.cvtColor(thresholdImage, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_show, patterns, -1, (0, 255, 0), 3)
    file2 = 'data/' + name + '_4.png'
    cv2.imwrite(file2, img_show)

    # show(img_show)

    while len(patterns) < 3 and levelsNum > 0:
        levelsNum -= 1
        patterns, patternsIndices = getContourWithinLevel(levelsNum, contours, hierarchy)

    interstingPatternList = []
    if len(patterns) < 3:
        print('no enough pattern')
        return False, []


    elif len(patterns) == 3:
        for patternIndex in range(len(patterns)):
            x, y, w, h = cv2.boundingRect(patterns[patternIndex])
            interstingPatternList.append(patterns[patternIndex])

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        show(img, 'qrcode')


    elif len(patterns) > 3:
        patternAreaList = np.array(
            [cv2.contourArea(pattern) for pattern in patterns])
        areaIdList = np.argsort(patternAreaList)
        intrestingPatternIdList = []
        for i in range(len(areaIdList) - 1, 0, -1):
            index = patternsIndices[areaIdList[i]]
            if hierarchy[0][index][3] == -1:
                intrestingPatternIdList.append(index)
            else:
                if not isParentInList(intrestingPatternIdList, index, hierarchy):
                    intrestingPatternIdList.append(index)
        img_show = cv2.cvtColor(thresholdImage, cv2.COLOR_GRAY2BGR)
        for intrestingPatternId in intrestingPatternIdList:
            x, y, w, h = cv2.boundingRect(contours[intrestingPatternId])

            cv2.rectangle(img_show, (x, y), (x + w, y + h), (0, 255, 0), 2)
            interstingPatternList.append(contours[intrestingPatternId])
        file = 'data/' + name + '_5.png'
        cv2.imwrite(file, img_show)
        # show(img_show, 'qrcode')

    img_show = cv2.cvtColor(thresholdImage, cv2.COLOR_GRAY2BGR)
    centerOfMassList = getCenterOfMass(interstingPatternList)

    for centerOfMass in centerOfMassList:
        cv2.circle(img_show, tuple(centerOfMass), 3, (0, 255, 0))
    file = 'data/' + name + '_6.png'
    cv2.imwrite(file, img_show)
    # show(img_show, 'qrcode')

    id1, id2, id3 = 0, 1, 2
    if len(patterns) > 3:
        result = selectPatterns(centerOfMassList)
        if result is None:
            print('no correct pattern')
            return False, []
        id1, id2, id3 = result

    interstingPatternList = np.array(interstingPatternList)[[id1, id2, id3]]
    centerOfMassList = np.array(centerOfMassList)[[id1, id2, id3]]
    pointList = getOrientation(interstingPatternList, centerOfMassList)

    img_show = img.copy()
    for point in pointList:
        cv2.circle(img_show, tuple([int(point[0]), int(point[1])]), 10, (0, 255, 0), -1)
    point = pointList[0]
    cv2.circle(img_show, tuple([int(point[0]), int(point[1])]), 10, (0, 0, 255), -1)
    file1 = output_file_path
    cv2.imwrite(file1, img_show)
    # show(img_show,'20')
    tend = time.time()
    print("Time for running %s %f seconds" % (input_file_path, tend - tstart))


def main():
    tstart = time.time()

    input_list = ["data/1.png", "data/2.png", "data/3.png", "data/4.png", "data/8.png", "data/9.png", "data/10.png"]

    all_process = []

    for (i, j) in enumerate(input_list):
        input_file_path = j
        output_file_path = j.replace(".png", "_result.png")
        t = multiprocessing.Process(target=process, name=input_file_path, args=(input_file_path, output_file_path))
        all_process.append(t)

    for t in all_process:
        t.start()

    for t in all_process:
        t.join()

    tend = time.time()
    print("Time for running all file %f seconds" % (tend - tstart))


if __name__ == "__main__":
    main()
