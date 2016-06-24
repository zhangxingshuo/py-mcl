import cv2
import numpy as np 
import math
import glob
import time

from Matcher import Matcher 

class analyzer(object):

    def __init__(self, method, resolution1, resolution2):
        self.index1, self.index2, self.index3 = [], [], []
        self.method = method
        self.res1 = resolution1
        self.res2 = resolution2
        self.rawP = []
        self.blurP = []
        self.commands = self.readCommand('commands.txt')
        self.bestGuess = []
        # self.pGen = []
        #res1 = 320
        #res2 = 240

    def readProb(self, filename):
        '''this function reads the probabilities from a txt file'''
        file = open(filename,'r')
        content = file.read().split('\n')[:-1]
        newContent = []
        for i in range(len(content))[::2]:
            L = list(map(float, content[i+1].replace('[','').replace(']','').split(',')))
            newContent.append((int(content[i]), L))
        return newContent

    def probUpdate(self, previousP, currentP, blurFactor):
        '''this function weighted the probability list according to the blurriness factor'''
        currentWeight = 0
        if blurFactor > 200:
            currentWeight = 0.85
        else:
            currentWeight = (blurFactor / 200) * 0.85
        previousWeight = 1 - currentWeight

        # Assigning the weight to each list
        truePosition = [[0, []], [0,[]] , [0,[]]]


        for circleIndex in range(len(truePosition)):
            currentCircle = currentP[circleIndex]
            previousCircle = previousP[circleIndex]

            # Number of matches 
            current_num_matches = currentCircle[0]
            previous_num_matches = previousCircle[0]
            
            # Each probability list
            current_probList = currentCircle[1]
            previous_probList = previousCircle[1]


            truePosition[circleIndex][0] = (currentWeight * current_num_matches + previousWeight * previous_num_matches)
            for probIndex in range(len(currentP[circleIndex][1])): 

                current_prob = current_probList[probIndex]
                previous_prob = previous_probList[probIndex]

                truePosition[circleIndex][1].append(currentWeight * current_prob + previousWeight * previous_prob)

        return truePosition

    def prevWeight(self, previousP, currentP):
        '''this function weighted the probability list according to the blurriness factor'''
        currentWeight = 0.9
        previousWeight = 1- currentWeight
        # if blurFactor > 200:
        #     currentWeight = 0.85
        # else:
        #     currentWeight = (blurFactor / 200) * 0.85
        # previousWeight = 1 - currentWeight

        # Assigning the weight to each list
        truePosition = [[0, []], [0,[]] , [0,[]]]


        for circleIndex in range(len(truePosition)):
            currentCircle = currentP[circleIndex]
            previousCircle = previousP[circleIndex]

            # Number of matches 
            current_num_matches = currentCircle[0]
            previous_num_matches = previousCircle[0]
            
            # Each probability list
            current_probList = currentCircle[1]
            previous_probList = previousCircle[1]


            truePosition[circleIndex][0] = (currentWeight * current_num_matches + previousWeight * previous_num_matches)
            for probIndex in range(len(currentP[circleIndex][1])): 

                current_prob = current_probList[probIndex]
                previous_prob = previous_probList[probIndex]

                truePosition[circleIndex][1].append(currentWeight * current_prob + previousWeight * previous_prob)

        return truePosition

    def createIndex(self):
        ''' This function creates indexes of feature '''
        matcher = Matcher(self.method, width=self.res1, height=self.res2)
        matcher.setDirectory('spot_one')
        self.index1 = matcher.createFeatureIndex()
        matcher.setDirectory('spot_two')
        self.index2 = matcher.createFeatureIndex()
        matcher.setDirectory('spot_three')
        self.index3 = matcher.createFeatureIndex()
        # self.index1 = Matcher('query.jpg','spot_one', self.method, None, self.res1, self.res2).createFeatureIndex()
        # self.index2 = Matcher('query.jpg','spot_two', self.method, None, self.res1, self.res2).createFeatureIndex()
        # self.index3 = Matcher('query.jpg','spot_three', self.method, None, self.res1, self.res2).createFeatureIndex()

    def readCommand(self, filename):
        '''this function reads the command list from the robot'''
        file = open(filename, 'r')
        content = file.read().split('\n')[:-1]
        commandDict = {}
        for data in content:
            commandDict[data[:4]] = str(data[-1])
        return commandDict

    def createRawP(self):
        ''' This function generates a list of raw probabilities directly from image matching'''
        self.createIndex()
        p = []
        matcher = Matcher(self.method, width=self.res1, height=self.res2)
        for imagePath in glob.glob('cam1_img' + '/*.jpg')[:5]:
            matcher.setQuery(imagePath)

            matcher.setDirectory('spot_one')
            matcher.setIndex(self.index1)
            totalMatches1, results1, _ = matcher.run()

            matcher.setDirectory('spot_two')
            matcher.setIndex(self.index2)
            totalMatches2, results2, _ = matcher.run()

            matcher.setDirectory('spot_three')
            matcher.setIndex(self.index3)
            totalMatches3, results3, _ = matcher.run()
            # totalMatches1, results1, __ = Matcher(imagePath, 'spot_one', self.method, self.index1, self.res1, self.res2).run()
            # totalMatches2, results2, __ = Matcher(imagePath, 'spot_two', self.method, self.index2, self.res1, self.res2).run()
            # totalMatches3, results3, __ = Matcher(imagePath, 'spot_three', self.method, self.index3, self.res1, self.res2).run()
            p.extend([[totalMatches1, results1], [totalMatches2, results2], [totalMatches3, results3]])
            # print(p)
            # cv2.waitKey(0)        
        self.rawP = p

    def Laplacian(self, imagePath):
        ''' this function calcualte the blurriness factor'''
        img = cv2.imread(imagePath, 0)
        var = cv2.Laplacian(img, cv2.CV_64F).var()
        return var

    def createBlurP(self):
        ''' This function generates a list of probabilities after accounting for the blurriness factor'''
        start = time.time()
        self.createIndex()
        blurP = []
        previousProbs = [[1, [1/75] * 25 ], [1,[1/75] * 25 ] , [1,[1/75] * 25]]
        matcher = Matcher(self.method, width=self.res1, height=self.res2)
        for imagePath in glob.glob('cam1_img' + '/*.jpg'):
            p = []
            matcher.setQuery(imagePath)

            matcher.setDirectory('spot_one')
            matcher.setIndex(self.index1)
            totalMatches1, results1, _ = matcher.run()

            matcher.setDirectory('spot_two')
            matcher.setIndex(self.index2)
            totalMatches2, results2, _ = matcher.run()

            matcher.setDirectory('spot_three')
            matcher.setIndex(self.index3)
            totalMatches3, results3, _ = matcher.run()
            # totalMatches1, results1, __ = Matcher(imagePath, 'spot_one', self.method, self.index1, self.res1, self.res2).run()
            # totalMatches2, results2, __ = Matcher(imagePath, 'spot_two', self.method, self.index2, self.res1, self.res2).run()
            # totalMatches3, results3, __ = Matcher(imagePath, 'spot_three', self.method, self.index3, self.res1, self.res2).run()
            p.extend([[totalMatches1, results1], [totalMatches2, results2], [totalMatches3, results3]])

            # Reading Blur
            blurFactor = self.Laplacian(imagePath)
            # Reading Command
            command = self.commands[imagePath.replace('cam1_img/', '').replace('.jpg', '')]

            # Account for Action
            actionAccount = self.accountCommand(command, previousProbs)

            # Adjusting for Action
            adjusted = self.prevWeight(actionAccount, p)

            # Adjusting for Blur
            adjusted = self.probUpdate(actionAccount, adjusted, blurFactor)

            # Getting best guess
            # this will get the max of the first variable
            bestCircleIndex = adjusted.index(max(adjusted[0], adjusted[1], adjusted[2]))
            bestAngleIndex = adjusted[bestCircleIndex][1].index(max(adjusted[bestCircleIndex][1]))
            self.bestGuess.extend([[bestCircleIndex, bestAngleIndex]])
            blurP.extend(adjusted)
            previousProbs = adjusted
            self.rawP.extend(p)
            print(imagePath)
        self.blurP = blurP
        self.writeProb(self.blurP, 'out.txt', 'w')
        self.writeProb(self.bestGuess, 'bestGuess.txt', 'w')
        self.writeCoord('coord.txt','w')
        end = time.time()

        print('Time elapsed: %0.3f' % (end-start))

    def accountCommand(self, command, previousP):
        '''this funciton accounts for the command robot is given at the moment'''
        # Left
        copy = previousP[:]
        if command == 'l':
            for circles in copy:
                circles[1] = circles[1][1:] + circles[1][0:1]
        if command == "r":
            for circles in copy:
                circles[1] = circles[1][-1:] + circles[1][0:-1]
        return copy

    def trackRobot(self, imagePath):
        '''this function track the robot and return its coordinates'''
        img = cv2.imread(imagePath)
        img = cv2.flip(img, 1)
        img = cv2.flip(img, 0)

        # convert into hsv 
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Find mask that matches 
        green_mask = cv2.inRange(hsv, np.array((50., 30., 0.)), np.array((100., 255., 255.)))
        green_mask = cv2.erode(green_mask, None, iterations=2)
        green_mask = cv2.dilate(green_mask, None, iterations=2)

        green_cnts = cv2.findContours(green_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        green_c = max(green_cnts, key=cv2.contourArea)

        # fit an ellipse and use its orientation to gain info about the robot
        green_ellipse = cv2.fitEllipse(green_c)

        # This is the position of the robot
        green_center = (int(green_ellipse[0][0]), int(green_ellipse[0][1]))

        red_mask = cv2.inRange(hsv, np.array((0., 100., 100.)), np.array((80., 255., 255.)))
        red_mask = cv2.erode(red_mask, None, iterations=2)
        red_mask = cv2.erode(red_mask, None, iterations=2)

        red_cnts = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        red_c = max(red_cnts, key=cv2.contourArea)

        red_ellipse = cv2.fitEllipse(red_c)
        red_center = (int(red_ellipse[0][0]), int(red_ellipse[0][1]))

        return green_center, red_center

    def writeCoord(self, filename, mode):
        '''this function writes out the coordinate of the robot to a txt file'''
        file = open(filename, mode)
        for imagePath in glob.glob('cam2_img' + '/*.jpg'):
            position, orientation = self.trackRobot(imagePath)
            file.write('%d,%d,%d,%d\n' % (position[0], position[1], orientation[0], orientation[1]))


    def writeProb(self, prob, filename, mode):
        ''' this function write out the probabilistic values to a txt file'''
        file = open(filename, mode)
        for index in prob:
            file.write(str(index[0]) + '\n')
            file.write(str(index[1]) + '\n')





            
