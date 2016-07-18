import cv2
import glob
import time
import numpy as np 

currentIndex = 0

# def Laplacian(imagePath):
#     img = cv2.imread(imagePath, 0)
#     var = cv2.Laplacian(img, cv2.CV_64F).var()
#     return var
extension = '.jpg'
def readCoord(filename):
    file = open(filename, 'r')
    content = file.read().split('\n')[:-1]
    coordinates = [list(map(int, coord.split(','))) for coord in content]
    return coordinates

def copyTo(arr1, arr2, y, x):
    '''
    Copies arr2 into arr1, with the top left corner of arr2 at the
    specified location.
    '''
    if y + arr2.shape[0] > arr1.shape[0] or x + arr2.shape[1] > arr1.shape[1]:
        raise ValueError('Dimensions of image are exceeded.')

    # for i in range(arr2.shape[0]):
    #     for j in range(arr2.shape[1]):
    #         arr1[y+i][j+x] = arr2[i][j]
    arr1[y:y+arr2.shape[0], x:x+arr2.shape[1]] = arr2

def readBestGuess(filename):
    '''this function reads the list of best guesses of the robot's position at every position'''
    file = open(filename, 'r')
    content = file.read().split('\n')[:-1]
    content = list(map(int, content))
    bestGuesses = [[content[x], content[x+1]] for x in range(len(content) - 1 ) [::2]    ]
    return bestGuesses

def Laplacian(img):
    ''' this function calculates the blurriness factor'''
    # img = cv2.imread(imagePath, 0)
    var = cv2.Laplacian(img, cv2.CV_64F).var()
    return var

def readCommand(filename):
    '''this function reads the command list from the robot'''
    file = open(filename, 'r')
    content = file.read().split('\n')[:-1]
    commandDict = {}
    for data in content:
        commandDict[data[:4]] = str(data[-1])
    return commandDict


coordinates = readCoord('coord.txt')
bestGuess = readBestGuess('bestGuess.txt')
commands = readCommand('commands.txt')
maxImg = len(commands)

img = np.zeros((960,1280, 3), np.uint8)

while True:
    filename = str(currentIndex).zfill(4)
    visual = cv2.imread('visual/' + filename + extension)
    cv2.putText(visual, 'Algorithm Representation', (30,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
    g = cv2.imread('cam2_img/' + filename + '.jpg')

    groundTruth = cv2.flip(g, 1)
    groundTruth = cv2.flip(groundTruth, 0)
    cv2.putText(groundTruth, 'Ground Truth', (30,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))

    novel = cv2.imread('cam1_img/' + filename + extension)
    novel = cv2.resize(novel, (640, 480))
    blurFactor = Laplacian(novel)
    cv2.putText(novel, 'Robot Camera', (30,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
    cv2.putText(novel, 'Blur: ' + str(blurFactor), (30, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))

    bestCircle = bestGuess[int(filename)][0]
    bestArrow = bestGuess[int(filename)][1]

    guess = cv2.imread(('map/%s/angle%s' + extension) % (str(bestCircle), str(15*bestArrow).zfill(3)))
    guess = cv2.resize(guess, (640,480))
    cv2.putText(guess, 'Image Match', (30,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
    # Illustrate the position of the robot
    # center = tuple(coordinates[currentIndex][:2])
    # cv2.circle(groundTruth, center, 5, (0,255,0), -1)

    # Illustrate the orientation
    # or_point = tuple(coordinates[currentIndex][2:])
    # cv2.arrowedLine(groundTruth, center, or_point, (0,255,0), 3)
    # blurFactor = Laplacian('cam1_img/' + filename + '.jpg')
    # cv2.putText(visual, str(blurFactor), (500, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    # cv2.imshow('Ground Truth', groundTruth)
    # cv2.imshow('Visual Representation', visual)
    # cv2.imshow('Novel', novel)
    copyTo(img, visual, 0, 0)
    copyTo(img, novel, 480, 640)
    copyTo(img, groundTruth, 0, 640)
    copyTo(img, guess, 480, 0)
    cv2.imshow('Monte Carlo Localization', img)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('d') and currentIndex <= maxImg-2:
        currentIndex += 1
    if k == ord('a') and currentIndex >= 1:
        currentIndex -= 1
    if k == 27:
        break

    # time.sleep(0.5)

cv2.destroyAllWindows()