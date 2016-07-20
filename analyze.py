'''
Visual Monte Carlo Localization
===============================

Given a dataset of images and map of surroundings,
can successfully localize the camera within the
surroundings.

Supports multiple image-matching algorithms, including 
Scale-Invariant Feature Transform (SIFT), Speeded-Up
Robust Features (SURF), and color histogram matching. 

Corrects for motion blur by weighing each update 
proportional to the variance of the Laplacian. Also
has a motion model to shift the particles according to
the motion of the camera. 
'''

import cv2
import numpy as np 
import math
import glob
import time

from Matcher import Matcher 

extension = '.png'

class analyzer(object):

    def __init__(self, method, width, height):
        self.numLocations = 7
        self.indices = [None] * self.numLocations
        self.method = method
        self.w = width
        self.h = height
        self.rawP = []
        self.blurP = []
        self.commands = self.readCommand('commands.txt')
        self.bestGuess = []

    def createIndex(self):
        ''' This function creates indexes of feature '''
        matcher = Matcher(self.method, width=self.w, height=self.h)
        for i in range(self.numLocations):
            matcher.setDirectory('map/' + str(i))
            if self.method != 'Color':
                self.indices[i] = matcher.createFeatureIndex()
            else:
                self.indices[i] = matcher.createColorIndex()

    ####################
    ### Main Methods ###
    ####################

    def createRawP(self):
        ''' This function generates a list of raw probabilities directly from image matching'''
        self.createIndex()
        start = time.time()
        p = []
        matcher = Matcher(self.method, width=self.w, height=self.h)

        print('Matching...')
        for imagePath in glob.glob('cam1_img' + '/*' + extension):
            matcher.setQuery(imagePath)
            results = []
            for i in range(self.numLocations):
                matcher.setDirectory('map/' + str(i))
                if self.method != 'Color':
                    matcher.setIndex(self.indices[i])
                else:
                    matcher.setColorIndex(self.indices[i])
                totalMatches, probL, _ = matcher.run()
                results.append([totalMatches, probL])

            p.extend(results)  
            print('\t' + imagePath)
        self.rawP = p
        self.writeProb(p, 'rawP.txt', 'w')

        end = time.time()
        print('Time elapsed: %0.1f' % (end-start))

    def processRaw(self):
        '''this function processed the raw function'''
        previousProbs = []
        for i in range(self.numLocations):
            previousProbs.append([1, [1/75] * 25])

        start = time.time()
        probDict = self.readProb('rawP.txt')
        blurP = []

        for imagePath in glob.glob('cam1_img' + '/*' + extension):
            p = probDict[imagePath.replace('cam1_img/', '').replace(extension, '')]

            # Reading Blur
            blurFactor = self.Laplacian(imagePath)
            # Reading Command
            command = self.commands[imagePath.replace('cam1_img/', '').replace(extension, '')]
            # Account for Command
            actionAccount = self.accountCommand(command, previousProbs)
            # Adjusting for Command
            adjusted = self.prevWeight(actionAccount, p)
            # Adjusting for Blur
            adjusted = self.probUpdate(actionAccount, adjusted, blurFactor)

            # Getting best guess
            # this will get the max of the first variable
            bestCircleIndex = adjusted.index(max(adjusted))
            bestAngleIndex = adjusted[bestCircleIndex][1].index(max(adjusted[bestCircleIndex][1]))
            self.bestGuess.extend([[bestCircleIndex, bestAngleIndex]])
            blurP.extend(adjusted)
            previousProbs = adjusted
            print('\t' + imagePath)

        self.blurP = blurP
        self.writeProb(self.blurP, 'out.txt', 'w')
        self.writeProb(self.bestGuess, 'bestGuess.txt', 'w')

        end = time.time()
        print('Time elapsed: %0.1f' % (end-start))

    def optP(self):
        print('Creating indices...')
        self.createIndex()

        blurP = []
        previousProbs = []
        bestAngleIndex = None
        bestCircleIndex = None
        for i in range(self.numLocations):
            previousProbs.append([1, [1/75] * 25])

        matcher = Matcher(self.method, width=self.w, height=self.h)
        start = time.time()

        print('Matching...')
        for imagePath in glob.glob('cam1_img' + '/*' + extension):
            p = []
            matcher.setQuery(imagePath)
            results = []
            if bestCircleIndex == None:
                for i in range(self.numLocations):
                    matcher.setDirectory('map/' + str(i))
                    if self.method != 'Color':
                        matcher.setIndex(self.indices[i])
                    else:
                        matcher.setColorIndex(self.indices[i])
                    totalMatches, probL, _ = matcher.optRun(bestAngleIndex)
                    results.append([totalMatches, probL])
            else:
                lower = bestCircleIndex - 2
                upper = bestCircleIndex + 2
                for i in range(self.numLocations):
                    if i >= lower and i <= upper:
                        matcher.setDirectory('map/' + str(i))
                        if self.method != 'Color':
                            matcher.setIndex(self.indices[i])
                        else:
                            matcher.setColorIndex(self.indices[i])
                        totalMatches, probL, _ = matcher.optRun(bestAngleIndex)
                        results.append([totalMatches, probL])
                    else:
                        results.append([1, [1/75] * 25])


            p.extend(results)  
            print('\t' + imagePath)
            
            blurFactor = self.Laplacian(imagePath)
            # Reading Command
            command = self.commands[imagePath.replace('cam1_img/', '').replace(extension, '')]
            # Account for Command
            actionAccount = self.accountCommand(command, previousProbs)
            # Adjusting for Command
            adjusted = self.prevWeight(actionAccount, p)
            # Adjusting for Blur
            adjusted = self.probUpdate(actionAccount, adjusted, blurFactor)

            # Getting best guess
            # this will get the max of the first variable
            bestCircleIndex = adjusted.index(max(adjusted))
            bestAngleIndex = adjusted[bestCircleIndex][1].index(max(adjusted[bestCircleIndex][1]))
            self.bestGuess.extend([[bestCircleIndex, bestAngleIndex]])
            blurP.extend(adjusted)
            previousProbs = adjusted

        self.blurP = blurP
        self.writeProb(self.blurP, 'out.txt', 'w')
        self.writeProb(self.bestGuess, 'bestGuess.txt', 'w')

        end = time.time()
        print('Time elapsed: %0.1f' % (end-start))


    ############################
    ### Probability Updating ###
    ############################

    def probUpdate(self, previousP, currentP, blurFactor):
        '''this function weighted the probability list according to the blurriness factor'''
        currentWeight = 0
        if blurFactor > 200:
            currentWeight = 0.85
        else:
            currentWeight = (blurFactor / 200) * 0.85
        previousWeight = 1 - currentWeight

        # Assigning the weight to each list
        truePosition = []
        for i in range(self.numLocations):
            truePosition.append([0, []])

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
        currentWeight = 0.5
        previousWeight = 1- currentWeight

        # Assigning the weight to each list
        truePosition = []
        for i in range(self.numLocations):
            truePosition.append([0, []])

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

    def accountCommand(self, command, previousP):
        '''this funciton accounts for the command robot is given at the moment'''
        # Left
        copy = previousP[:]
        if command == 'l':
            for circles in copy:
                circles[1] = circles[1][1:] + circles[1][0:1]
        elif command == 'r':
            for circles in copy:
                circles[1] = circles[1][-1:] + circles[1][0:-1]
        elif command == 'f':
            bestCircleIndex = previousP.index(max(previousP))
            bestAngleIndex = previousP[bestCircleIndex][1].index(max(previousP[bestCircleIndex][1]))
            factor = 0.2 * abs(math.sin(bestAngleIndex*15 * 180/math.pi))
            if bestCircleIndex < self.numLocations - 1 and bestAngleIndex*15 < 180 and bestAngleIndex > 0:
                copy[bestCircleIndex+1][0] *= (1 + factor)
            elif bestCircleIndex > 0 and bestAngleIndex*15 > 180 and bestAngleIndex*15 < 360: 
                copy[bestCircleIndex-1][0] *= (1 + factor)
        return copy


    ###################################
    ### Reading and Writing Methods ###
    ###################################

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

    def readCommand(self, filename):
        '''this function reads the command list from the robot'''
        file = open(filename, 'r')
        content = file.read().split('\n')[:-1]
        commandDict = {}
        for data in content:
            commandDict[data[:4]] = str(data[-1])
        return commandDict

    def readBestGuess(self, filename):
        '''this function reads the list of best guesses of the robot's position at every position'''
        file = open(filename, 'r')
        content = file.read().split('\n')[:-1]
        content = list(map(int, content))
        bestGuesses = [[content[x], content[x+1]] for x in range(len(content) - 1 ) [::2]    ]
        return bestGuesses

    def readCoord(self, filename):
        file = open(filename, 'r')
        content = file.read().split('\n')[:-1]
        coordinates = [list(map(int, coord.split(','))) for coord in content]
        return coordinates

    def readProb(self, filename):
        '''this function reads the content of a txt file, turn the data into  dictionaries of 
        circles'''
        file = open(filename, 'r')
        raw_content = file.read().split('\n')[:-1]
        raw_chunks = [raw_content[i:i+2] for i in range(0, len(raw_content), 2)]
        raw_probL = [raw_chunks[i:i+self.numLocations] for i in range(0, len(raw_chunks), self.numLocations)]
        probD = {}
        counter = 0
        for prob in raw_probL:
            content = []
            for location in prob:
                totalMatches = float(location[0])
                probabilities = list(map(float, location[1].replace('[','').replace(']','').split(',')))
                content.append([totalMatches, probabilities])
            probD[str(counter).zfill(4)] = content
            counter += 1
        return probD    


    #############################
    ### Miscellaneous Methods ###
    #############################  

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

    def Laplacian(self, imagePath):
        ''' this function calcualte the blurriness factor'''
        img = cv2.imread(imagePath, 0)
        var = cv2.Laplacian(img, cv2.CV_64F).var()
        return var     
