'''
General-Purpose Image Matching
==============================

Class that can match an image against a dataset with
options for parameters such as image size and the type 
of image matching, e.g. color-based, SIFT, SURF, ORB,
etc.

Usage:
------
    python Matcher.py -q [<query image>] -d [<directory>] -a [<algorithm>]

    Viable algorithms are ORB, SIFT, and SURF.
'''

import cv2
import numpy as np
import argparse
import glob
import time
from matplotlib import pyplot as plt

from search import Searcher

extension = '.png'

class Matcher(object):

    ######################
    ### Initialization ###
    ######################

    def __init__(self, algorithm, index=None, width=800, height=600):
        self.w = width
        self.h = height
        self.alg = algorithm
        self.index = index

    def setQuery(self, imagePath):
        self.image = cv2.imread(imagePath)
        self.image = cv2.bilateralFilter(self.image, 9, 75, 75)
        self.image = cv2.resize(self.image, (self.w, self.h))

    def setDirectory(self, directory):
        self.data = directory

    def setIndex(self, index):
        self.index = index

    def setColorIndex(self, colorIndex):
        self.colorIndex = colorIndex


    ############################
    ### Index Initialization ###
    ############################

    def createColorIndex(self):
        '''
        Creates dictionary with keys as image names and histograms as values.
        '''
        index = {}

        for imagePath in glob.glob(self.data + "/*" + extension):
            filename = imagePath[imagePath.rfind("/") + 1:]
            image = cv2.imread(imagePath)
            features = self.createHistogram(image)
            index[filename] = features

        return index

    def createFeatureIndex(self):
        '''
        Creates a dictionary with keys as image paths and values as keypoints and descriptors
        '''
        if self.alg == 'SURF':
            desc = cv2.xfeatures2d.SURF_create()
        elif self.alg == 'SIFT':
            desc = cv2.xfeatures2d.SIFT_create()
        else:
            desc = cv2.ORB_create()

        index = {}
        for imagePath in glob.glob(self.data + '/*' + extension):
            image = cv2.imread(imagePath)
            # image = cv2.bilateralFilter(image, 9, 75, 75)
            kp, des = desc.detectAndCompute(image, None)
            index[imagePath] = (kp, des)

        return index


    ############################
    ### Color-Based Matching ###
    ############################

    def createHistogram(self, image, bins=[8, 8, 8]):
        '''
        Creates a flattened 3D histogram.
        '''
        hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist.flatten()

    def colorSearch(self):
        '''
        Searches query image against index and returns the specified number of matches.
        Results are in the format (chi-squared distance, image name).
        '''
        searcher = Searcher(self.colorIndex)
        queryFeatures = self.createHistogram(self.image)

        results = searcher.search(queryFeatures)

        return results


    #################################
    ### Image Matching Algorithms ###
    #################################

    def ORBMatch(self, imagePath, display_results=False):
        '''
        Matches query against specified image using the Oriented FAST and Rotated BRIEF algorithm.
        Matching is done through Brute-Force.
        '''

        orb = cv2.ORB_create()

        kp1, des1 = orb.detectAndCompute(self.image, None)
        # kp2, des2 = orb.detectAndCompute(training, None)
        kp2, des2 = self.index[imagePath]

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)

        if display_results:
            draw_params = dict(matchColor=(0,255,0), 
                singlePointColor=None, 
                flags=2)

            image = cv2.drawMatches(self.image, kp1, training, kp2, matches, None, **draw_params)
            plt.imshow(image), plt.show()

        return len(good)

    def SURFMatch(self, imagePath, display_results=False):
        '''
        Performs a match using Speeded-Up Robust Features algorithm.
        Matching is done with Fast Library for Approximate Nearest Neighbors.
        Lowe's ratio test is applied.
        '''

        surf = cv2.xfeatures2d.SURF_create()

        kp1, des1 = surf.detectAndCompute(self.image, None)
        if not self.index:
            training = cv2.imread(imagePath)
            kp2, des2 = surf.detectAndCompute(training, None)
        else:
            kp2, des2 = self.index[imagePath]

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=25)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)
        filtered = list(filter(lambda x:x[0].distance < 0.7*x[1].distance, matches))
        good = list(map(lambda x: x[0], filtered))

        if display_results:
            draw_params = dict(matchColor=(0,255,0), 
                singlePointColor=None, 
                flags=2)

            result = cv2.drawMatches(self.image, kp1, training, kp2, good, None, **draw_params)
            plt.imshow(result), plt.show()

        return len(good)

    def SIFTMatch(self, imagePath, display_results=False):
        '''
        Performs a match using Scale-Invariant Feature Transform algorithm.
        Matching is done with Fast Library for Approximate Nearest Neighbors.
        Lowe's ratio test is applied.
        '''

        sift = cv2.xfeatures2d.SIFT_create()

        kp1, des1 = sift.detectAndCompute(self.image, None)
        # kp2, des2 = sift.detectAndCompute(training, None)
        kp2, des2 = self.index[imagePath]

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)
        filtered = list(filter(lambda x:x[0].distance < 0.7*x[1].distance, matches))
        good = list(map(lambda x: x[0], filtered))

        if display_results:
            draw_params = dict(matchColor=(0,255,0), 
                singlePointColor=None, 
                flags=2)

            result = cv2.drawMatches(self.image, kp1, training, kp2, good, None, **draw_params)
            plt.imshow(result), plt.show()

        return len(good)

    def run(self):
        if self.alg != 'Color':
            matches = []
            for i in range(0, 375, 15):
                imagePath = self.data + '/angle' + str(i).zfill(3) + extension
                # print('\tMatching %s ...' % imagePath)
                if self.alg == 'SIFT':
                    numMatches = self.SIFTMatch(imagePath)
                elif self.alg == 'SURF':
                    numMatches = self.SURFMatch(imagePath)
                else:
                    numMatches = self.ORBMatch(imagePath)
                # print("\tFound %s matches" % numMatches)
                matches.append((imagePath, numMatches))

            totalMatches = sum(list(map(lambda x: x[1], matches)))
            if totalMatches == 0:
                totalMatches = 1

        else:
            results = self.colorSearch()
            totalChiSquared = sum(list(map(lambda x: x[0], results)))
            totalMatches = 300000./totalChiSquared
            rawProbs = list(map(lambda x: (self.data + '/' + x[1], 200./x[0]), results)) # invert chi-squared
            totalProb = sum(list(map(lambda x: x[1], rawProbs)))
            rawMatches = list(map(lambda x: (x[0], x[1]/totalProb * totalMatches), rawProbs)) # normalize probabilities
            matches = sorted(rawMatches, key=lambda x: int(x[0].replace(extension,'').replace(self.data+'/angle','')))

        sorted_matches = sorted(matches, key=lambda x: x[1])
        
        return totalMatches, list(map(lambda x:x[1]/totalMatches, matches)), sorted_matches[-1]

    def optRun(self, bestAngleIndex):
        if bestAngleIndex is not None:
            bestAngle = bestAngleIndex * 15
            lower = bestAngle - 30
            upper = bestAngle + 30
            # optimized run
            if self.alg != 'Color':
                matches = []
                for i in range(0, 375, 15):
                    imagePath = self.data + '/angle' + str(i).zfill(3) + extension
                    if lower >= 0 and upper <= 360:
                        if i >= lower and i <= upper:
                            # print('\tMatching %s ...' % imagePath)
                            if self.alg == 'SIFT':
                                numMatches = self.SIFTMatch(imagePath)
                            elif self.alg == 'SURF':
                                numMatches = self.SURFMatch(imagePath)
                            else:
                                numMatches = self.ORBMatch(imagePath)
                        else:
                            numMatches = 1

                    else:
                        if i >= lower % 360 or i <= upper % 360:
                            if self.alg == 'SIFT':
                                numMatches = self.SIFTMatch(imagePath)
                            elif self.alg == 'SURF':
                                numMatches = self.SURFMatch(imagePath)
                            else:
                                numMatches = self.ORBMatch(imagePath)
                        else:
                            numMatches = 1
                    matches.append((imagePath, numMatches))

                totalMatches = sum(list(map(lambda x: x[1], matches)))
                if totalMatches == 0:
                    totalMatches = 1

            else:

                results = self.colorSearch()
                totalChiSquared = sum(list(map(lambda x: x[0], results)))
                totalMatches = 300000./totalChiSquared
                rawProbs = list(map(lambda x: (self.data + '/' + x[1], 200./x[0]), results)) # invert chi-squared
                totalProb = sum(list(map(lambda x: x[1], rawProbs)))
                rawMatches = list(map(lambda x: (x[0], x[1]/totalProb * totalMatches), rawProbs)) # normalize probabilities
                matches = sorted(rawMatches, key=lambda x: int(x[0].replace(extension,'').replace(self.data+'/angle','')))

            sorted_matches = sorted(matches, key=lambda x: x[1])
            return totalMatches, list(map(lambda x:x[1]/totalMatches, matches)), sorted_matches[-1]
        else:
            return self.run()

if __name__ == '__main__':

    print(__doc__)
