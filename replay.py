import cv2
import glob
import time

currentIndex = 0

# def Laplacian(imagePath):
#     img = cv2.imread(imagePath, 0)
#     var = cv2.Laplacian(img, cv2.CV_64F).var()
#     return var

def readCoord(filename):
    file = open(filename, 'r')
    content = file.read().split('\n')[:-1]
    coordinates = [list(map(int, coord.split(','))) for coord in content]
    return coordinates

coordinates = readCoord('coord.txt')
maxImg = len(coordinates)

while True:
    filename = str(currentIndex).zfill(4)
    visual = cv2.imread('visual/' + filename + ".jpg")
    g = cv2.imread('cam2_img/' + filename + '.jpg')

    groundTruth = cv2.flip(g, 1)
    groundTruth = cv2.flip(groundTruth, 0)
    novel = cv2.imread('cam1_img/' + filename + '.jpg')

# Illustrate the position of the robot
    # center = tuple(coordinates[currentIndex][:2])
    # cv2.circle(groundTruth, center, 5, (0,255,0), -1)

    # Illustrate the orientation
    # or_point = tuple(coordinates[currentIndex][2:])
    # cv2.arrowedLine(groundTruth, center, or_point, (0,255,0), 3)
    # blurFactor = Laplacian('cam1_img/' + filename + '.jpg')
    # cv2.putText(visual, str(blurFactor), (500, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    cv2.imshow('Ground Truth', groundTruth)
    cv2.imshow('Visual Representation', visual)
    cv2.imshow('Novel', novel)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('d') and currentIndex <= maxImg-2:
        currentIndex += 1
    if k == ord('a') and currentIndex >= 1:
        currentIndex -= 1
    if k == 27:
        break

    # time.sleep(0.5)

cv2.destroyAllWindows()