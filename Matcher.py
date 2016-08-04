'''
General-Purpose Image Matching
==============================

Class that can match an image against a dataset with
options for parameters such as image size and the type 
of image matching, e.g. color-based, SIFT, SURF, ORB,
etc.

Also included are optimization techniques such as Bag-of-
Words and k-means clustering, as well as DOR, a dynamically
optimized retrieval algorithm for visual particle filters.

Usage:
------
    python Matcher.py -q [<query image>] -d [<directory>] -a [<algorithm>]

    Viable algorithms are ORB, SIFT, SURF, and BOW.
'''

import cv2
import numpy as np
import glob
import time
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from search import Searcher

from scipy.cluster.vq import *

from sklearn import preprocessing

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
        self.numWords = 10000

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
        searcher = Searcher(self.colorIndex)
        queryFeatures = self.createHistogram(self.image)

        results = searcher.search(queryFeatures)
        return results


    #############################
    ### Bag-of-Words Matching ###
    #############################

    def createIndex(self, trainingPath):
        desc = cv2.xfeatures2d.SIFT_create()
        train_des_list = []
        numWords = self.numWords
        image_paths = glob.glob(trainingPath + '/*' + '.png')
        # Extract the descriptors from the maps and store them 
        for imagePath in glob.glob(trainingPath + '/*' + '.png'):
            print(imagePath)
            image = cv2.imread(imagePath)
            kp, des = desc.detectAndCompute(image, None)
            train_des_list.append((imagePath, des))

        # Stack them all in a numpy array
        train_descriptors = train_des_list[0][1]
        for image_path, descriptor in train_des_list[1:]:
            train_descriptors = np.vstack((train_descriptors, descriptor))

        # Perform K-means clustering?
        voc, variance = kmeans(train_descriptors, numWords, 1)

        # Calculate histogram of features for the Training SET
        im_features = np.zeros((len(image_paths), numWords), "float32")
        for i in range(len(glob.glob(trainingPath + '/*' + '.png'))):
            words, distance = vq(train_des_list[i][1], voc)
            for w in words:
                im_features[i][w] += 1

        # Perform Tf-idf vectorization for Training set
        nbr_occurences = np.sum((im_features > 0) * 1, axis = 0)
        idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

        # Perform L2 normalization
        im_features = im_features*idf
        im_features = preprocessing.normalize(im_features, norm='l2')

        joblib.dump((im_features, image_paths, idf, numWords, voc), trainingPath + ".pkl", compress=3)

    def writeIndices(self):
        for mapp in glob.glob('map/*/'):
            self.createIndex(mapp[:-1])

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

        orb = cv2.ORB_create()

        kp1, des1 = orb.detectAndCompute(self.image, None)
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

    def BOWMatch(self, indexPath):
        '''the query's score against an individual index'''
        # start = time.time()
        query_des_list = []
        im_features, image_paths, idf, numWords, voc = joblib.load(indexPath)
        numWords = self.numWords

        desc = cv2.xfeatures2d.SIFT_create()
        # Extract the descriptors from the query 
        query = self.image
        kp, des = desc.detectAndCompute(query, None)
        query_des_list.append((query, des))

        # Stack query descriptors in a numpy array
        query_descriptors = query_des_list[0][1]

        # Calculate histogram of Features for the Query 
        test_features = np.zeros((1, numWords), "float32")
        words, distance = vq(query_descriptors, voc)
        for w in words:
            test_features[0][w] += 1 

        # Perform Tf-idf vectorization for the Query
        test_features = test_features * idf
        test_features = preprocessing.normalize(test_features, norm='l2')

        score = np.dot(test_features, im_features.T)
        return score


    def SIFTMatch(self, imagePath, display_results=False):
        '''
        Performs a match using Scale-Invariant Feature Transform algorithm.
        Matching is done with Fast Library for Approximate Nearest Neighbors.
        Lowe's ratio test is applied.
        '''
        sift = cv2.xfeatures2d.SIFT_create()

        kp1, des1 = sift.detectAndCompute(self.image, None)
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
        
        if self.alg != 'Color' and self.alg != 'BOW':
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
                matches.append((imagePath, numMatches))

            totalMatches = sum(list(map(lambda x: x[1], matches)))
            if totalMatches == 0:
                totalMatches = 1

        elif self.alg == 'BOW':
            for i in range(0, 375, 15):
                score = self.BOWMatch(self.data + '.pkl')
                return 10*np.max(score), score[0].tolist()
        else:

            results = self.colorSearch()
            totalChiSquared = sum(list(map(lambda x: x[0], results)))
            totalMatches = 300000./totalChiSquared
            rawProbs = list(map(lambda x: (self.data + '/' + x[1], 200./x[0]), results)) # invert chi-squared
            totalProb = sum(list(map(lambda x: x[1], rawProbs)))
            rawMatches = list(map(lambda x: (x[0], x[1]/totalProb * totalMatches), rawProbs)) # normalize probabilities
            matches = sorted(rawMatches, key=lambda x: int(x[0].replace(extension,'').replace(self.data+'/angle','')))
        
        return totalMatches, list(map(lambda x:x[1]/totalMatches, matches))

    def optRun(self, bestAngleIndex):
        if bestAngleIndex is not None:
            bestAngle = bestAngleIndex * 15
            lower = bestAngle - 30
            upper = bestAngle + 30
            # optimized run
            if self.alg != 'Color' and self.alg != 'BOW':
                matches = []
                for i in range(0, 375, 15):
                    imagePath = self.data + '/angle' + str(i).zfill(3) + extension
                    if lower >= 0 and upper <= 360:
                        if i >= lower and i <= upper:
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

            return totalMatches, list(map(lambda x:x[1]/totalMatches, matches))
        else:
            return self.run()

if __name__ == '__main__':

    print(__doc__)
