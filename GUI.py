import cv2
import numpy as np 
import math
import glob

from Matcher import Matcher

extension = '.png'
NUM_LOCATIONS = 7

class Circle(object):
    def __init__(self, radius, x, y, folder, color):
        self.r = radius
        self.x = x
        self.y = y
        self.folder = folder
        self.color = color

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
        cv2.arrowedLine(image, (self.circle.x, self.circle.y), (self.x, self.y), self.color, self.size)



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
    file = open(filename, 'r')
    raw_content = file.read().split('\n')[:-1]
    raw_chunks = [raw_content[i:i+2] for i in range(0, len(raw_content), 2)]
    raw_probL = [raw_chunks[i:i+NUM_LOCATIONS] for i in range(0, len(raw_chunks), NUM_LOCATIONS)]
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

def initializeCircle():
    circles = [None] * NUM_LOCATIONS
    for i in range(NUM_LOCATIONS):
        circles[i] = Circle(50, 141 + 150 * i, 221, 'map/'+str(i), [150, 150, 150])
    return circles

# Initialize Screen
cv2.namedWindow('GUI')  
circles = initializeCircle()

arrows = []
for circle in circles:
    arrows.append(getArrows(circle, 25))

commandList = readCommand('commands.txt')
probDict = readProb('out.txt')
coordinates = readCoord('coord.txt')
bestGuess = readBestGuess('bestGuess.txt')

# Outputting the Probability

for imagePath in glob.glob('cam1_img' + '/*' + extension):
    # Initiating views
    img = np.zeros((480,200 + 150 * NUM_LOCATIONS,3), np.uint8)
    novelView = cv2.imread(imagePath)
    groundTruth = cv2.imread(imagePath.replace('cam1_img', 'cam2_img'))

    # Read matching data
    p = probDict[imagePath.replace('cam1_img/', '').replace(extension, '')]

    # Accounting for Blur factor 
    blurFactor = Laplacian(imagePath)
    illustrateProb(circles, arrows, p)

    bestCircleIndex = bestGuess[int(imagePath.replace('cam1_img/', '').replace(extension, ''))][0]
    bestArrowIndex = bestGuess[int(imagePath.replace('cam1_img/', '').replace(extension, ''))][1]
    # Best arrow:
    bestArrow = arrows[bestCircleIndex][bestArrowIndex]
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

    cv2.imwrite('visual/' + imagePath.replace('cam1_img/', ''), img)
    print('Writing %s...' %  imagePath.replace('cam1_img/', ''))

cv2.destroyAllWindows()