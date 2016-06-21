import cv2
import numpy as np 
import math
import glob

from Matcher import Matcher 

class analyzer(object):

    def __init__(self, imagePath, probsPath, method, resolution1, resolution2):
        # self.probsList = readProb(probsPath)
        # self.image = cv2.imread(imagePath)
        self.index1, self.index2, self.index3 = [], [], []
        self.method = method
        self.res1 = resolution1
        self.res2 = resolution2
        self.rawP = []
        self.blurP = []


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

    def createIndex(self):
        ''' This function creates indexes of feature '''
        self.index1 = Matcher('query.jpg','spot_one', self.method, None, self.res1, self.res2).createFeatureIndex('one_index.p')
        self.index2 = Matcher('query.jpg','spot_two', self.method, None, self.res1, self.res2).createFeatureIndex('two_index.p')
        self.index3 = Matcher('query.jpg','spot_three', self.method, None, self.res1, self.res2).createFeatureIndex('three_index.p')


    def createRawP(self):
        ''' This function generates a list of raw probabilities directly from image matching'''
        self.createIndex()
        p = []
        for imagePath in glob.glob('cam1_img' + '/*.jpg')[:5]:
             totalMatches1, results1, __ = Matcher(imagePath, 'spot_one', self.method, self.index1, self.res1, self.res2).run()
             totalMatches2, results2, __ = Matcher(imagePath, 'spot_two', self.method, self.index2, self.res1, self.res2).run()
             totalMatches3, results3, __ = Matcher(imagePath, 'spot_three', self.method, self.index3, self.res1, self.res2).run()
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
        self.createIndex()
        blurP = []
        previousProbs = [[0, [0] * 25 ], [0,[0] * 25 ] , [0,[0] * 25]]
        for imagePath in glob.glob('cam1_img' + '/*.jpg'):
             p = []
             totalMatches1, results1, __ = Matcher(imagePath, 'spot_one', self.method, self.index1, self.res1, self.res2).run()
             totalMatches2, results2, __ = Matcher(imagePath, 'spot_two', self.method, self.index2, self.res1, self.res2).run()
             totalMatches3, results3, __ = Matcher(imagePath, 'spot_three', self.method, self.index3, self.res1, self.res2).run()
             p.extend([[totalMatches1, results1], [totalMatches2, results2], [totalMatches3, results3]])
             blurFactor = self.Laplacian(imagePath)
             adjusted = self.probUpdate(previousProbs, p, blurFactor)
             blurP.extend(adjusted)
             previousProbs = adjusted
             self.rawP.extend(p)
             print(imagePath)
        self.blurP = blurP
        self.writeOut('out.txt', 'w')

    def writeOut(self, filename, mode):
        file = open(filename + '', mode)
        for index in self.blurP:
            file.write(str(index[0]) + '\n')
            file.write(str(index[1]) + '\n')





            
