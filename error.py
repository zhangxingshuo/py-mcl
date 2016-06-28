import math

#####################
### Reading files ###
#####################

def readProb(filename):
    '''this function reads the content of a txt file, turn the data into  dictionaries of 
    circles'''
    file = open(filename, 'r') 
    content = file.read().split('\n')[:-1]
    probDict = {}
    counter = 0
    for i in range(len(content))[::6]:
        name = str(counter).zfill(4)
        L1 = list(map(float, content[i+1].replace('[','').replace(']','').split(',')))
        L2 = list(map(float, content[i+3].replace('[','').replace(']','').split(',')))
        L3 = list(map(float, content[i+5].replace('[','').replace(']','').split(',')))
        probDict[name] = [[float(content[i]), L1], [float(content[i+2]), L2], [float(content[i+4]), L3]]
        counter += 1
    return probDict

def readBestGuess(filename):
    '''this function reads the list of best guesses of the robot's position at every position'''
    file = open(filename, 'r')
    content = file.read().split('\n')[:-1]
    content = list(map(int, content))
    bestGuesses = [[content[x], content[x+1]] for x in range(len(content) - 1 ) [::2]]
    return bestGuesses

def readCoord(filename):
    file = open(filename, 'r')
    content = file.read().split('\n')[:-1]
    coordinates = [list(map(int, coord.split(','))) for coord in content]
    return coordinates

#############################
### Inlier-Outlier Method ###
#############################

def successMetric():
    ''' this function calculate the differences between best guesses and the true values'''
    bestGuess = readBestGuess('bestGuess.txt')
    # best Guess contains a list of lists. First elements is the index of the best cirlce, second element is the index
    # of the best angle
    coordinates = readCoord('coord.txt')
    # coordinates is the "real" position of the robot as analyzed by the image matching algorithm
    # the first two points of coordinates is the position of the robot, the second set of points
    # are direction of the angle 

    success = []
    for i in range(len(bestGuess)):
        bestGuessAngle = bestGuess[i][1]*15
        positions = []
        bestGuessCircle = bestGuess[i][0]
        robotPos = coordinates[i][:2]
        robotDir = coordinates[i][2:]
        angle = math.atan2(robotDir[1] - robotPos[1], robotDir[0] - robotPos[0])
        angle = angle*180./math.pi + 90
        if angle < 0:
            angle += 360
        success.append(abs(angle-bestGuessAngle) < 30)
        
    return success.count(True)/len(success)

######################
### Modal Variance ###
######################

def modalMetric():
    ''' this function calculate the differences between best guesses and the true values'''
    bestGuess = readBestGuess('bestGuess.txt')
    # best Guess contains a list of lists. First elements is the index of the best cirlce, second element is the index
    # of the best angle
    coordinates = readCoord('coord.txt')
    # coordinates is the "real" position of the robot as analyzed by the image matching algorithm
    # the first two points of coordinates is the position of the robot, the second set of points
    # are direction of the angle 
    angleError = 0
    for i in range(len(bestGuess)):
        bestGuessAngle = bestGuess[i][1] * 15
        robotPos = coordinates[i][:2]
        robotDir = coordinates[i][2:]
        angle = math.atan2(robotDir[1] - robotPos[1], robotDir[0] - robotPos[0])
        angle *= 180./math.pi
        angle += 90
        if angle <= 0:
            angle += 360
        # print (angle)
        # print ("best: " + str(bestGuessAngle))
        if bestGuessAngle == 0 and angle > 300:
            bestGuessAngle = 360
        elif bestGuessAngle == 360 and angle < 60:
            bestGuessAngle = 0
        angleError += (bestGuessAngle - angle)**2

    return math.sqrt(angleError)

########################################
### Average Probabilitistic Variance ###
########################################

def errorMetric():
    ''' this function calculate the differences between best guesses and the true values'''
    bestGuess = readBestGuess('bestGuess.txt')
    # best Guess contains a list of lists. First elements is the index of the best cirlce, second element is the index
    # of the best angle
    coordinates = readCoord('coord.txt')
    # coordinates is the "real" position of the robot as analyzed by the image matching algorithm
    # the first two points of coordinates is the position of the robot, the second set of points
    # are direction of the angle 

    probD = readProb('out.txt')

    L= []
    for key, value in probD.items():
        index = int(key)
        bestGuessAngle = bestGuess[index][1]*15
        bestGuessCircle = bestGuess[index][0]
        predictedAngles = value[bestGuessCircle][1]
        angleError = 0
        robotPos = coordinates[index][:2]
        robotDir = coordinates[index][2:]
        angle = math.atan2(robotDir[1] - robotPos[1], robotDir[0] - robotPos[0])
        angle = angle*180./math.pi + 90
        if angle < 0:
            angle += 360
        for i in range(len(predictedAngles)):
            prob = predictedAngles[i]
            predAngle = i*15
            angleError += prob * (predAngle - angle)**2
        L.append(angleError)
        
    return sum(L)/len(L)
            