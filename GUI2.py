import cv2
import numpy as np 
import math
import glob

from Matcher import Matcher

class Circle(object):
    def __init__(self, radius, x, y, folder, color):
        self.r = radius
        self.x = x
        self.y = y
        self.folder = folder
        self.color = color
        self.panoWindow = self.folder + " panorama"
        self.pano = cv2.imread(self.folder + "_panorama.jpg")

    def draw(self, image):
        cv2.circle(image, (self.x, self.y), self.r, self.color, -1)

    def setColor(self, co):
        self.color = co

    def showPanorama(self):
        cv2.imshow(self.panoWindow, self.pano)
        cv2.waitKey(0)
        cv2.destroyWindow(self.panoWindow)

    def inCircle(self, point):
        if (point[0]-self.x)**2 + (point[1]-self.y)**2 < self.r**2:
            return True
        return False

class Arrow(object):
    def __init__(self, Circle, length, angle, size, color):
        self.size = size
        self.color = color
        self.circle = Circle
        self.angle = angle
        self.length = length
        self.x = int(Circle.x + length*math.cos(angle + 3*math.pi/2))
        self.y = int(Circle.y + length*math.sin(angle + 3*math.pi/2))


    def setSize(self, s):
        self.size = s

    def setLength(self, l):
        mult_constant = self.length * l
        self.x = int(self.circle.x + mult_constant*math.cos(self.angle + 3*math.pi/2))
        self.y = int(self.circle.y + mult_constant*math.sin(self.angle + 3*math.pi/2))

    def setColor(self, co):
        self.color = co

    def draw(self, image):
        # print (self.color)
        cv2.arrowedLine(image, (self.circle.x, self.circle.y), (self.x, self.y), self.color, self.size)
        # cv2.arrowedLine(image, (self.circle.x,self.circle.y), (100,100), (50, 50, 50), 10)



def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for circle in circles:
            if circle.inCircle((x,y)):
                circle.showPanorama()

def getArrows(Cir, intervals):
    '''return a list of arrows pointing in all direction
    intervals defined how many arrows there are in a full circle'''
    angInterval = 2*math.pi/intervals
    center_x = Cir.x
    center_y = Cir.y
    arrowList = []
    for i in range(intervals):
        arrow = Arrow(Cir, 60, angInterval * i, 1, (200,200,200))
        arrowList.append(arrow)   
    return arrowList

def drawArrows(arrowL):
    ''' this function initialize all the arrows. All grey with length 1'''
    # arrowL = getArrow(Circle, interval)
    for arrow in arrowL:
        arrow.draw(img)

def setArrow(arrowL, index, thickness, color, length):
    ''' this function access an individual arrow and modify its 
    size, color, and magnitude'''
    arrowL[index].setSize(thickness)
    arrowL[index].setColor(color)
    arrowL[index].setLength(length)

def resetArrow(arrowL):
    for arrows in arrowL:
        for ind_arrow in arrows:
            ind_arrow.setColor((200,200,200))
            ind_arrow.setLength(60)

def drawCircle(circleL):
    for circle in circleL:
        circle.draw(img) 

def normalize(prob):
    return [float(i)/sum(prob) for i in prob]

def illustrateProb(circle, arrowsL, probsL):
    '''circleL is the list of circles in one region, and arrowsL are the 
    corresponding circles'''
    
    # Let the color range goes from 50 to 255. The higher it is the more
    # likely the robot is there
    minColor = 0
    maxColor = 255
    diff = maxColor - minColor
    totalMatches = sum(list(map(lambda x: x[0],probsL)))
    # Resetting the arrows
    resetArrow(arrowsL)

    for circle_ind in range(len(probsL)):
        num_matches, list_of_probs = probsL[circle_ind]
        diffProb = max(list_of_probs) - min(list_of_probs)
        num_probs = len(list_of_probs)
        this_circles_arrows = arrowsL[circle_ind]
        circle[circle_ind].setColor(((num_matches/ totalMatches)*255, (num_matches/ totalMatches)*255,
                                    (num_matches/ totalMatches)*255))
        for j in range(num_probs ):
            this_prob = list_of_probs[j]
            color = (this_prob/diffProb * 255, 50, 50)
            mult = (num_matches/ totalMatches) * this_prob * 20
            setArrow(this_circles_arrows, j, 1, color, mult)


def read(filename):
    '''this function reads the content from a txt file'''
    file = open(filename,'r')
    content = file.read().split('\n')[:-1]
    newContent = []
    for i in range(len(content))[::2]:
        L = list(map(float, content[i+1].replace('[','').replace(']','').split(',')))
        newContent.append((int(content[i]), L))
    return newContent

def multipleMatch(folder,method,  res1, res2):
    '''this function establish multiple matches and output everything in out.txt'''
    for imagePath in glob.glob(folder + '/*.jpg'):
        Matcher(imagePath, 'spot_one', method, res1, res2).write("out.txt", 'w')
        Matcher(imagePath, 'spot_two', method, res1, res2).write("out.txt", 'a')
        Matcher(imagePath, 'spot_three', method, res1, res2).write("out.txt", 'a')

def Laplacian(imagePath):
    ''' this function calcualte the blurriness factor'''
    img = cv2.imread(imagePath, 0)
    var = cv2.Laplacian(img, cv2.CV_64F).var()
    return var

def probUpdate(previousP, currentP, blurFactor):
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
        # print (circleIndex)
        # Circles
        currentCircle = currentP[circleIndex]
        previousCircle = previousP[circleIndex]

        # Number of matches 
        current_num_matches = currentCircle[0]
        previous_num_matches = previousCircle[0]
        
        # Each probability list
        current_probList = currentCircle[1]
        previous_probList = previousCircle[1]


        truePosition[circleIndex][0] = (currentWeight * current_num_matches + previousWeight * previous_num_matches)
        # print(truePosition[circleIndex][0])
        for probIndex in range(len(currentP[circleIndex][1])): 

            current_prob = current_probList[probIndex]
            previous_prob = previous_probList[probIndex]
            # true_prob = true_probList[probIndex]

            truePosition[circleIndex][1].append(currentWeight * current_prob + previousWeight * previous_prob)
                       
            # print (truePosition[circleIndex][1])
            # print (truePosition)


    return truePosition



# Initiate Screen
img = np.zeros((540, 960, 3), np.uint8)
cv2.namedWindow('GUI')

res1 = 320
res2 = 240

# Initiating Circles and Matches
circle1 = Circle(50, 200, 200, 'spot_one', [150, 150, 150])
# Matcher('query.jpg','spot_one','SURF',320,240).write('out.txt','w')
spot_one_index = Matcher('query.jpg','spot_one','SURF', None, res1, res2).createFeatureIndex('one_index.p')
circle2 = Circle(50, 400, 200, 'spot_two', [150, 150, 150])
# Matcher('query.jpg','spot_two','SURF',320,240).write('out.txt','a')
spot_two_index = Matcher('query.jpg','spot_two','SURF', None, res1, res2).createFeatureIndex('two_index.p')
circle3 = Circle(50, 600, 200, 'spot_three', [150, 150, 150])
# Matcher('query.jpg','spot_three','SURF',320,240).write('out.txt','a')
spot_three_index = Matcher('query.jpg','spot_three','SURF', None, res1,res2).createFeatureIndex('three_index.p')
circles = [circle1, circle2, circle3]


# Initiate Matches
# multipleMatch('cam2_img', 'SURF', 320, 240)

# Initiating Arrows
arrows1 = getArrows(circle1, 25)
arrows2 = getArrows(circle2, 25)
arrows3 = getArrows(circle3, 25)

Arrows = [arrows1, arrows2, arrows3]

# p = read('out.txt')

# Account for possibility
# illustrateProb(circles, Arrows, p)

method = 'SURF'
previousProbs = [[0, [0] * 25 ], [0,[0] * 25 ] , [0,[0] * 25]]



# Outputting the Probability

for imagePath in glob.glob('cam1_img' + '/*.jpg'):
        # Initiating views
        img = np.zeros((540,960,3), np.uint8)
        novelView = cv2.imread(imagePath)
        groundTruth = cv2.imread(imagePath.replace('cam1_img', 'cam2_img'))

        # Output Matching data
        Matcher(imagePath, 'spot_one', method, spot_one_index, res1, res2).write("out.txt", 'w')
        Matcher(imagePath, 'spot_two', method, spot_two_index, res1, res2).write("out.txt", 'a')
        Matcher(imagePath, 'spot_three', method, spot_three_index, res1, res2).write("out.txt", 'a')

        # Read matching data
        p = read('out.txt')

        # Accounting for Blur factor 
        blurFactor = Laplacian(imagePath)
        p = probUpdate(previousProbs, p, blurFactor)
        # print (len(p[0][1]))
        previousProbs = p
        illustrateProb(circles, Arrows, p)

        # Drawing Circles
        drawCircle(circles)
        drawArrows(arrows1)
        drawArrows(arrows2)
        drawArrows(arrows3)
        cv2.putText(img, imagePath, (500,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)
        cv2.imwrite('visual/' + imagePath.replace('cam1_img/', ''), img)
        cv2.imshow('Visualization', img)
        cv2.imshow('Ground Truth', groundTruth)
        cv2.imshow('Novel', novelView)


# Matcher('cam1_img/0022.jpg', 'spot_one', method, spot_one_index, res1, res2).write("out.txt", 'w')
# Matcher('cam1_img/0022.jpg', 'spot_two', method, spot_two_index, res1, res2).write("out.txt", 'a')
# Matcher('cam1_img/0022.jpg', 'spot_three', method, spot_three_index, res1, res2).write("out.txt", 'a')
# p = read('out.txt')
# illustrateProb(circles, Arrows, p)
# drawCircle(circles)
# drawArrows(arrows1)
# drawArrows(arrows2)
# drawArrows(arrows3)
# cv2.imshow('GUI', img)
# cv2.waitKey(0)



# Constructing Circles and Arrows
# drawCircle(circles)
# drawArrows(arrows1)
# drawArrows(arrows2)
# drawArrows(arrows3)


# cv2.setMouseCallback('GUI', click)

# while True:
#     cv2.imshow('GUI', img)



#     k = cv2.waitKey(0) & 0xFF
#     if k == 27:
#         break

cv2.destroyAllWindows()