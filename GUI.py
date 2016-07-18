import cv2
import numpy as np 
import math
import glob

from Matcher import Matcher

extension = '.jpg'
NUM_LOCATIONS = 3

class Circle(object):
    def __init__(self, radius, x, y, folder, color):
        self.r = radius
        self.x = x
        self.y = y
        self.folder = folder
        self.color = color
        # self.panoWindow = self.folder + " panorama"
        # self.pano = cv2.imread(self.folder + "_panorama.jpg")

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

# def normalize(prob):
#     return [float(i)/sum(prob) for i in prob]

def illustrateProb(circle, arrowsL, probsL):
    '''circleL is the list of circles in one region, and arrowsL are the 
    corresponding circles'''
    
    minColor = 0
    maxColor = 255
    diff = maxColor - minColor
    totalMatches = sum(list(map(lambda x: x[0],probsL)))

    for circle_ind in range(len(probsL)):
        num_matches, list_of_probs = probsL[circle_ind]
        maxProb = max(list_of_probs)
        num_probs = len(list_of_probs)
        this_circles_arrows = arrowsL[circle_ind]
        circle[circle_ind].setColor(((num_matches/ totalMatches)*255, (num_matches/ totalMatches)*255,
                                    (num_matches/ totalMatches)*255))
        for j in range(num_probs ):
            this_prob = list_of_probs[j]
            blue = 0
            green = 0
            red = 0
            if this_prob >= 0.05:
                red = this_prob/maxProb * 255
                green = this_prob/maxProb * 255
                blue = (1-this_prob/maxProb) * 255 
            else:
                red = 0
                green = 0
                blue = 255 * this_prob/maxProb
            color = (blue, green, red)
            mult = (num_matches/ totalMatches) * this_prob * 20
            setArrow(this_circles_arrows, j, 1, color, mult)

def readProb(filename):
    '''this function reads the content of a txt file, turn the data into  dictionaries of 
    circles'''
    # file = open(filename, 'r') 
    # content = file.read().split('\n')[:-1]
    # probDict = {}
    # counter = 0
    # for i in range(len(content))[::6]:
    #     name = str(counter).zfill(4)
    #     L1 = list(map(float, content[i+1].replace('[','').replace(']','').split(',')))
    #     L2 = list(map(float, content[i+3].replace('[','').replace(']','').split(',')))
    #     L3 = list(map(float, content[i+5].replace('[','').replace(']','').split(',')))
    #     probDict[name] = [[float(content[i]), L1], [float(content[i+2]), L2], [float(content[i+4]), L3]]
    #     counter += 1
    # return probDict

    file = open(filename, 'r')
    raw_content = file.read().split('\n')[:-1]
    raw_chunks = [content[i:i+2] for i in range(0, len(content), 2)]
    raw_probL = [raw_chunks[i:i+NUM_LOCATIONS] for i in range(0, len(raw_chunks), NUM_LOCATIONS)]
    content = list(map(lambda x: (float(x[0], list(map(float, x[1].replace('[','').replace(']','').split(','))))), raw_probL))
    probD = {}
    for i in range(len(content)):
        probD[str(i).zfill(4)] = content[i]
    return probD

def readBestGuess(filename):
    '''this function reads the list of best guesses of the robot's position at every position'''
    file = open(filename, 'r')
    content = file.read().split('\n')[:-1]
    content = list(map(int, content))
    bestGuesses = [[content[x], content[x+1]] for x in range(len(content) - 1 ) [::2]    ]
    return bestGuesses

def readCommand(filename):
    '''this function reads the command list from the robot'''
    file = open(filename, 'r')
    content = file.read().split('\n')[:-1]
    commandDict = {}
    for data in content:
        commandDict[data[:4]] = str(data[-1])
    return commandDict

def readCoord(filename):
    file = open(filename, 'r')
    content = file.read().split('\n')[:-1]
    coordinates = [list(map(int, coord.split(','))) for coord in content]
    return coordinates

def Laplacian(imagePath):
    ''' this function calculates the blurriness factor'''
    img = cv2.imread(imagePath, 0)
    var = cv2.Laplacian(img, cv2.CV_64F).var()
    return var


# Initiate Screen
img = np.zeros((540, 1920, 3), np.uint8)
cv2.namedWindow('GUI')  

# Initiating Circles and Matches
circle1 = Circle(50, 141, 221, 'map/0', [150, 150, 150])
circle2 = Circle(50, 304, 207, 'map/1', [150, 150, 150])
circle3 = Circle(50, 498, 196, 'map/2', [150, 150, 150])
circles = [circle1, circle2, circle3]

# Initiating Arrows
# arrows1 = getArrows(circle1, 25)
# arrows2 = getArrows(circle2, 25)
# arrows3 = getArrows(circle3, 25)

# arrows = [arrows1, arrows2, arrows3]

arrows = []
for circle in circles:
    arrows.append(getArrows(circle, 25))

method = 'SURF'

commandList = readCommand('commands.txt')
probDict = readProb('out.txt')
coordinates = readCoord('coord.txt')
bestGuess = readBestGuess('bestGuess.txt')

# Outputting the Probability


for imagePath in glob.glob('cam1_img' + '/*' + extension):
        # Initiating views
        img = np.zeros((480,640,3), np.uint8)
        novelView = cv2.imread(imagePath)
        groundTruth = cv2.imread(imagePath.replace('cam1_img', 'cam2_img'))

        # Read matching data
        p = probDict[imagePath.replace('cam1_img/', '').replace(extension, '')]

        # Accounting for Blur factor 
        blurFactor = Laplacian(imagePath)
        illustrateProb(circles, Arrows, p)

        # Illustrate the position of the robot
        center = tuple(coordinates[int(imagePath.replace('cam1_img/', '').replace(extension, ''))][:2])

        # Illustrate the orientation
        or_point = tuple(coordinates[int(imagePath.replace('cam1_img/', '').replace(extension, ''))][2:])

        bestCircleIndex = bestGuess[int(imagePath.replace('cam1_img/', '').replace(extension, ''))][0]
        bestArrowIndex = bestGuess[int(imagePath.replace('cam1_img/', '').replace(extension, ''))][1]
        # Best arrow:
        bestArrow = Arrows[bestCircleIndex][bestArrowIndex]
        bestArrow.setColor((255,255, 0))
        bestArrow.setLength(2)
        bestArrow.setSize(5)
        # Drawing Circles
        drawCircle(circles)
        for arrow in arrows:
            drawArrows(arrow)
        cv2.putText(img, imagePath, (100,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)
        cv2.putText(img, commandList[imagePath.replace(extension, '').replace('cam1_img/', '')], (500, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
        cv2.putText(img, str(blurFactor), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

        # Draw actual position
        cv2.arrowedLine(img, (center[0], center[1] - 100), (or_point[0], or_point[1] - 100), (0,255,0), 3)
        cv2.circle(img, (center[0], center[1] - 100), 5, (0,0,255), -1)

        cv2.imwrite('visual/' + imagePath.replace('cam1_img/', ''), img)
        cv2.imshow('Visualization', img)
        # cv2.imshow('Ground Truth', groundTruth)
        cv2.imshow('Novel', novelView)

cv2.destroyAllWindows()