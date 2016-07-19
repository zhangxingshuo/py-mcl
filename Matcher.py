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
from pano import Panorama

extension = '.png'

class Matcher(object):

    ######################
    ### Initialization ###
    ######################

    def __init__(self, algorithm, index=None, width=800, height=600):
        # self.image = cv2.imread(queryPath)
        # self.image = cv2.bilateralFilter(self.image, 9, 75, 75)
        self.w = width
        self.h = height
        # self.image = cv2.resize(self.image, (self.w, self.h))
        # self.data = directory
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
    ### Color-Based Matching ###
    ############################

    def createHistogram(self, image, bins=[8, 8, 8]):
        '''
        Creates a flattened 3D histogram.
        '''

        hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist.flatten()

    def createColorIndex(self):
        '''
        Creates dictionary with keys as image names and histograms as values.
        '''
        # print("Indexing: " + self.data + "...")
        index = {}

        for imagePath in glob.glob(self.data + "/*" + extension):
            filename = imagePath[imagePath.rfind("/") + 1:]
            image = cv2.imread(imagePath)
            # print('\t%s' % imagePath)
            features = self.createHistogram(image)
            index[filename] = features

        return index

    def colorSearch(self):
        '''
        Searches query image against index and returns the specified number of matches.
        Results are in the format (chi-squared distance, image name).
        '''

        # image = cv2.imread(self.image)
        # print("Querying: " + self.image + " ...")
        searcher = Searcher(self.colorIndex)
        queryFeatures = self.createHistogram(self.image)

        results = searcher.search(queryFeatures)

        # print("Matches found:")
        # for j in range(len(results)):
        #     (score, imageName) = results[j]
        #     print("\t%d. %s : %.3f" % (j+1, imageName, score))

        return results



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



    #################################
    ### Image Matching Algorithms ###
    #################################

    def ORBMatch(self, imagePath, display_results=False):
        '''
        Matches query against specified image using the Oriented FAST and Rotated BRIEF algorithm.
        Matching is done through Brute-Force.
        '''
        # training = cv2.imread(imagePath)
        # training = cv2.resize(training, (self.w, self.h))

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
        # training = cv2.imread(imagePath)
        # training = cv2.resize(training, (self.w, self.h))

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

    # def writeMatches(self, method):
    #     '''outputs the list of matched features in to txt files given the method'''
    #     file = open('matched_features.txt', 'w')
    #     if self.alg != 'Color':
    #         matched = []
    #         for i in range(0, 375, 15):
    #             imagePath = self.data + '/angle' + str(i).zfill(3) + '.jpg'
    #             # print('\tMatching %s ...' % imagePath)
    #             if self.alg == 'SIFT':
    #                 matched = self.SIFTMatch(imagePath)
    #             elif self.alg == 'SURF':
    #                 matched = self.SURFMatch(imagePath)
    #             else:
    #                 matched = self.ORBMatch(imagePath)
    #             # print("\tFound %s matches" % numMatches)
    #     for matchedPoint in matched:
    #         file.write(str(matchedPoint) + )


    def SIFTMatch(self, imagePath, display_results=False):
        '''
        Performs a match using Scale-Invariant Feature Transform algorithm.
        Matching is done with Fast Library for Approximate Nearest Neighbors.
        Lowe's ratio test is applied.
        '''
        # training = cv2.imread(imagePath)
        # training = cv2.resize(training, (self.w, self.h))

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

    def write(self, filename, mode):
        file = open(filename, mode)
        totalMatches, results, bestMatch = self.run()
        file.write(str(totalMatches) + '\n')
        file.write('[')
        for prob in results[:len(results)-1]:
            file.write(str(prob) + ', ')
        file.write(str(results[-1]) + ']\n')

        angle = int(bestMatch[0].replace(self.data,'').replace('/angle','').replace('.jpg',''))
        Panorama(self.data, 100, 100, angle).write(self.data + '_panorama.jpg')

    def run(self):
        # start = time.time()
        # print('%s matching...' % self.alg)
        # self.index = self.createFeatureIndex()
        
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

        # for j in range(1,6):
        #     (imageName, score) = sorted_matches[-j]
        #     print("%d. %s : %0.3f" % (j, imageName, score / totalMatches))

        # print("Found %d total matches" % totalMatches)

        # end = time.time()

        # print('Time elapsed: %0.1f s' % (end-start))
        
        return totalMatches, list(map(lambda x:x[1]/totalMatches, matches)), sorted_matches[-1]

    # def optRun(self, bestGuess):
    #     '''optimizing run that only tragets a few panoramas'''
    #     probsL = []
    #     if self.alg != 'Color':
    #         matches = []
    #         # consider only 5 angles around bestGuess the rest sum up to 0.2 
    #         bestAngle = bestGuess * 15
    #         upThreshold = bestAngle + 30
    #         lowThreshold = bestAngle - 30
    #         # angleOfInterests = [bestAngle - 30, bestAngle -15, bestAngle, bestAngle + 15, bestAngle + 30 ]
    #         for i in range(0, 375, 15):
    #             if i >= lowThreshold or i <= upThreshold: 
    #                 imagePath = self.data + '/angle' + str(i).zfill(3) + '.jpg'
    #                 if i > 360:
    #                     i = i % 360
    #                 elif i < 0:
    #                     i = i % 360
    #                 if self.alg == 'SIFT':
    #                     numMatches = self.SIFTMatch(imagePath)
    #                 elif self.alg == 'SURF':
    #                     numMatches = self.SURFMatch(imagePath)
    #                 else:
    #                     numMatches = self.ORBMatch(imagePath)
    #             else:
    #                 numMatches = 0
    #             # print("\tFound %s matches" % numMatches)
    #             matches.append((imagePath, numMatches))

    #         totalMatches = sum(list(map(lambda x: x[1], matches)))
            
    #         if totalMatches == 0:
    #             totalMatches = 1

    #     else:

    #         results = self.colorSearch()
    #         totalChiSquared = sum(list(map(lambda x: x[0], results)))
    #         totalMatches = 300000./totalChiSquared
    #         rawProbs = list(map(lambda x: (self.data + '/' + x[1], 200./x[0]), results)) # invert chi-squared
    #         totalProb = sum(list(map(lambda x: x[1], rawProbs)))
    #         rawMatches = list(map(lambda x: (x[0], x[1]/totalProb * totalMatches), rawProbs)) # normalize probabilities
    #         matches = sorted(rawMatches, key=lambda x: int(x[0].replace('.jpg','').replace(self.data+'/angle','')))

    #     sorted_matches = sorted(matches, key=lambda x: x[1])
    #     return totalMatches, list(map(lambda x:x[1]/totalMatches, matches)), sorted_matches[-1]

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
                        # print("\tFound %s matches" % numMatches)
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
    # ap = argparse.ArgumentParser()
    # ap.add_argument('-q', '--query', required=True,
    #     help='Path to query image')
    # ap.add_argument('-d', '--dataset', required=True,
    #     help='Path to directory of training images')
    # ap.add_argument('-a', '--algorithm', required=True,
    #     help='Algorithm to use for matching')
    # args = vars(ap.parse_args())

    print(__doc__)

    # Matcher(args['query'], args['dataset'], args['algorithm']).run()
