import cv2
import numpy as np 
import argparse

class Panorama(object):

    def __init__(self, data, height, width, matchAngle):
        self.dataset = data
        self.h = height
        self.w = width
        self.match = matchAngle
        self.images = self.readImages()

    def copyTo(self, arr1, arr2, y, x):
        '''
        Copies arr2 into arr1, with the top left corner of arr2 at the
        specified location.
        '''
        if y + arr2.shape[0] > arr1.shape[0] or x + arr2.shape[1] > arr1.shape[1]:
            raise ValueError('Dimensions of image are exceeded.')

        for i in range(arr2.shape[0]):
            for j in range(arr2.shape[1]):
                arr1[y+i][j+x] = arr2[i][j]

    def readImages(self):
        '''
        Read images from dataset. Images must be named as 'angle[<measure>].jpg', e.g.
        'angle225.jpg'. 
        '''
        imgArr = []
        for angle in range(0, 375, 15):
            currentImg = cv2.imread('%s/angle%s.jpg' % (self.dataset, str(angle).zfill(3)))
            smallImg = cv2.resize(currentImg, (self.w, self.h))
            cv2.putText(smallImg, str(angle), (int(0.55*self.w), self.h-10), cv2.FONT_HERSHEY_PLAIN, 1, 255)
            cv2.line(smallImg, (int(self.w*0.5), 0), (int(self.w*0.5), self.h), (255, 0, 0), 1)
            imgArr.append(smallImg)
        return imgArr

    def drawRect(self, image):
        pt1 = (int(self.match/360 * self.w * 24), 0)
        pt2 = (int(self.match/360 * self.w * 24 + self.w), self.h)
        cv2.rectangle(image, pt1, pt2, (0, 0, 255), 5)

    def write(self, filename):
        img = np.zeros((self.h, self.w*len(self.images), 3), np.uint8)
        for x in range(len(self.images)):
            try:
                self.copyTo(img, self.images[x], 0, x*self.w)
            except:
                pass
        self.drawRect(img)
        cv2.imwrite(filename, img   )

    def run(self):
        img = np.zeros((self.h, self.w*len(self.images), 3), np.uint8)
        for x in range(len(self.images)):
            try:
                self.copyTo(img, self.images[x], 0, x*self.w)
            except:
                pass
        self.drawRect(img)
        cv2.imshow('Panorama', img)
        cv2.waitKey(0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True,
        help='Path to the directory containing images')
    args = vars(ap.parse_args())

    Panorama(args['dataset'], 100, 100, 210).run()