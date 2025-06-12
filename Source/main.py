import pygame

import math
import random

import copy

#initialize pygame window
pygame.init()
screenWidth = 1200
screenHeight = 900
screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption('Learning Cars')

#fps display
clock = pygame.time.Clock()
def displayFPS(screen, font_size):
    font = pygame.font.SysFont(None, font_size)
    fps = round(clock.get_fps(), 1)
    fps_text = font.render(f"{fps}", True, (255, 255, 255))
    screen.blit(fps_text, (10, 10))

font = pygame.font.SysFont(None, 25)

#CLASS DEFINITION -----------------------------------------------------------------------------------------------------------------------------------------

class NeuralNetwork():
    def __init__(self, inputSize, hiddenSize, outputSize):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.hiddenSize2 = hiddenSize
        self.outputSize = outputSize

        #weights and biases
        self.weights_inputToHidden = [[random.uniform(-1, 1) for _ in range(inputSize)] for _ in range(hiddenSize)]
        self.weights_hiddenToHidden = [[random.uniform(-1, 1) for _ in range(hiddenSize)] for _ in range(hiddenSize)]
        self.weights_hiddenToOutput = [[random.uniform(-1, 1) for _ in range(hiddenSize)] for _ in range(outputSize)]

        self.bias_hidden = [random.uniform(-1, 1) for _ in range(hiddenSize)]
        self.bias_hidden2 = [random.uniform(-1, 1) for _ in range(hiddenSize)]
        self.bias_output = [random.uniform(-1, 1) for _ in range(outputSize)]

    def iterateForward(self, inputs):

        #get hidden layer calculations
        hidden = []
        for i in range(len(self.weights_inputToHidden)):
            activation = sum(w * inp for w, inp in zip(self.weights_inputToHidden[i], inputs)) + self.bias_hidden[i]
            hidden.append(sigmoid(activation))

        #get hidden2 layer calculations
        hidden2 = []
        for i in range(len(self.weights_hiddenToHidden)):
            activation = sum(w * inp for w, inp in zip(self.weights_hiddenToHidden[i], inputs)) + self.bias_hidden2[i]
            hidden2.append(sigmoid(activation))

        #get output layer calculations
        outputs = []
        for i in range(len(self.weights_hiddenToOutput)):
            activation = sum(w * h for w, h in zip(self.weights_hiddenToOutput[i], hidden)) + self.bias_output[i]
            outputs.append(sigmoid(activation))

        return outputs

    def interpretOutputAsAction(self, output):
        # Let's say:
        # output[0] = turn left strength (0 to 1)
        # output[1] = turn right strength (0 to 1)
        # output[2] = throttle (0 = off, 1 = on)

        turn = (output[1] - output[0]) * 2

        throttle = output[2]

        return turn, throttle

class Car():
    def __init__(self, x, y):
        #physics
        self.pos = (x, y)
        self.angle = 0
        self.vel = (0, 0)
        self.friction = 0.98
        self.accelaration = 240/60
        self.maxVel = 240/60
        self.turnRate = 150/60

        #visuals
        self.vertOffsets = [
            (0, -15), 
            (7, 7), 
            (0, 11), 
            (-7, 7)
        ]
        self.verts = [
            (self.pos[0] + 0, self.pos[1] - 15), 
            (self.pos[0] + 7, self.pos[1] + 7), 
            (self.pos[0] + 0, self.pos[1] + 11), 
            (self.pos[0] - 7, self.pos[1] + 7)
        ]

        #road references
        self.closestTweenPoint = None

        #rays
        self.numOfRays = 7
        self.rayLength = 250
        self.angleOffset = 180/(self.numOfRays - 1)

        self.rayVertOffsets = []
        self.rayVerts = []

        for i in range(self.numOfRays):
            vctr = angleToVector(180 + i * self.angleOffset)
            self.rayVertOffsets.append((vctr[0] * self.rayLength, vctr[1] * self.rayLength))

            #initial update
            self.rayVerts.append((self.pos[0] + self.rayVertOffsets[i][0], self.pos[1] + self.rayVertOffsets[i][1]))

        self.rayIntersections = [(self.rayVerts[i][0], self.rayVerts[i][1]) for i in range(self.numOfRays)]

        self.dead = False

        #neuralnetwork stuff
        cars.append(self)

        self.nn = NeuralNetwork(self.numOfRays + 1, 6, 3)
        self.fitness = 0

        self.life = 20*60

    def updateVerts(self):
        for i in range(len(self.verts)):
            #update pos
            self.verts[i] = (self.pos[0] + self.vertOffsets[i][0], self.pos[1] + self.vertOffsets[i][1])

            #rotate reinitialized verts
            self.verts[i] = rotatePoint(self.verts[i], self.pos, self.angle)

        for i in range(self.numOfRays):
            #update ray end pos and reinitialize verts for relative transform
            self.rayVerts[i] = (self.pos[0] + self.rayVertOffsets[i][0], self.pos[1] + self.rayVertOffsets[i][1])
            self.rayVerts[i] = rotatePoint(self.rayVerts[i], self.pos, self.angle)
            
    def draw(self):
        self.updateVerts()

        #draw ray lines
        """for i in range(self.numOfRays):
            pygame.draw.line(screen, (80, 0, 0), self.pos, self.rayVerts[i], 1)"""

        if not self.dead:
            carColor = (255, 255, 255)
        else:
            carColor = (100, 100, 100)

        pygame.draw.polygon(screen, carColor, self.verts, 2)

        #draw line towards closest tween point
        """if self.closestTweenPoint != None:
            pygame.draw.line(screen, (0, 0, 255), self.pos, self.closestTweenPoint)"""

    def move(self, dTs, road):
        #apply friction
        self.vel = (self.vel[0] * self.friction, self.vel[1] * self.friction)

        #apply velocity
        self.pos = (self.pos[0] + self.vel[0], self.pos[1] + self.vel[1])

        #get updated fitness
        if self.closestTweenPoint != None:
            self.fitness = road.cornerTweenPoints.index(self.closestTweenPoint)

        #update life 
        self.life -= 1
        if self.life <= 0:
            self.dead = True

    def throttle(self, strengthFactor, dTs):
        #apply accelaration forward
        angleRad = math.radians(self.angle - 90)
        accelaration = (math.cos(angleRad) * self.accelaration, math.sin(angleRad) * self.accelaration)
        self.vel = (self.vel[0] + accelaration[0]* strengthFactor, self.vel[1] + accelaration[1] * strengthFactor)

        #clamp to max speed if neccesary
        mag = math.sqrt(self.vel[0] ** 2 + self.vel[1] ** 2)

        if mag > self.maxVel:
            nVel = (self.vel[0] / mag, self.vel[1] / mag)
            self.vel = [nVel[0] * self.maxVel, nVel[1] * self.maxVel]

    def turn(self, direction, dTs):
        #apply turn to angle
        self.angle += direction * self.turnRate

        #normalize angle
        self.angle = self.angle % 360

    def getClosestTweenPoint(self, points):
        closestPoint = None
        closestDist = math.inf
        for i in range(len(points)):
            vec = (self.pos[0] - points[i][0], self.pos[1] - points[i][1])
            dist = vec[0] ** 2 + vec[1] ** 2

            if dist < closestDist:
                closestDist = dist
                closestPoint = points[i]
        self.closestTweenPoint = closestPoint

    def getIntersections(self, road):
        #get road edges as simpler array
        roadsides = road.leftSidePoints + road.rightSidePoints

        #for every ray
        for i in range(self.numOfRays):
            #get all intersections
            intersections = []

            #for every road edge
            for j in range(len(roadsides)):
                #check for intersections
                nextRoadsideIndex = (j + 1) % len(roadsides)

                #check if the line passes to the opposite side, (blocking road when t4ransitioning from left to right array)
                isJinLeft = roadsides[j] in road.leftSidePoints
                isJNextinLeft = roadsides[nextRoadsideIndex] in road.leftSidePoints

                if isJinLeft == isJNextinLeft:

                    intersection = linenItersection(self.pos, self.rayVerts[i], roadsides[j], roadsides[nextRoadsideIndex], True)
                    if intersection != None:
                        intersections.append(intersection)

            #get closest intersection
            closestIndex = None
            closestDist = math.inf
            for i in range(len(intersections)):
                vec = (self.pos[0] - intersections[i][0], self.pos[1] - intersections[i][1])
                squaredDist = vec[0] ** 2 + vec[1] ** 2

                if squaredDist < closestDist:
                    closestDist = squaredDist
                    closestIndex = i

            #draw dot
            if closestIndex != None:
                """pygame.draw.circle(screen, (255, 255, 0), intersections[closestIndex], 3)"""
                self.rayIntersections[i] = intersections[closestIndex]

        #check if car is on the road
        numOfIntersections = 0
        for j in range(len(roadsides)):
            #check for intersections
            nextRoadsideIndex = (j + 1) % len(roadsides)

            #check if the line passes to the opposite side, (blocking road when t4ransitioning from left to right array)
            isJinLeft = roadsides[j] in road.leftSidePoints
            isJNextinLeft = roadsides[nextRoadsideIndex] in road.leftSidePoints

            if isJinLeft == isJNextinLeft:

                intersection = linenItersection(self.pos, (self.pos[0] - 2000, self.pos[1]), roadsides[j], roadsides[nextRoadsideIndex], True)
                if intersection != None:
                    numOfIntersections += 1
        
        #do some even-odd trick
        if numOfIntersections % 2 != 1:
            self.dead = True

    def performAction(self, dTs):
        #get inputs
        inputs = []

        #add distance to intersections
        for i in range(self.numOfRays):
            vec = (self.pos[0] - self.rayIntersections[i][0], self.pos[1] - self.rayIntersections[i][1])
            mag = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
            inputs.append(mag)

        #add velocity magnitude
        inputs.append(math.sqrt(self.vel[0] ** 2 + self.vel[1] ** 2))

        #normalize inputs
        for i in range(self.numOfRays):
            inputs[i] = inputs[i] / self.rayLength

        inputs[-1] = inputs[-1] / self.maxVel

        #get outputs
        outputs = self.nn.iterateForward(inputs)

        #interpet outputs
        outputCalls = self.nn.interpretOutputAsAction(outputs)

        #perform action
        self.turn(outputCalls[0], dTs)
        self.throttle(outputCalls[1], dTs)

class Road():
    def __init__(self):
        self.cornerPoints = [
            (309, 766),
            (184, 732),
            (130, 631),
            (198, 528),
            (127, 349),
            (262, 213),
            (347, 72),
            (557, 143),
            (731, 91),
            (853, 177),
            (952, 333),
            (954, 481),
            (949, 670),
            (824, 750),
            (576, 796)
        ]

        self.cornerTweenPoints = []
        self.tweenAccuracy = 10

        #initialize tween points
        for i in range(len(self.cornerPoints)):
            for j in range(self.tweenAccuracy):
                #get vector to next road
                nextIndex = (i + 1) % len(self.cornerPoints)
                v = (self.cornerPoints[nextIndex][0] - self.cornerPoints[i][0], self.cornerPoints[nextIndex][1] - self.cornerPoints[i][1])

                #create new tween point
                self.cornerTweenPoints.append((self.cornerPoints[i][0] + v[0] * (j / self.tweenAccuracy), self.cornerPoints[i][1] + v[1] * (j / self.tweenAccuracy)))

        #initialize track sides
        self.roadThickness = 70

        #generate left and right side seperatly
        self.leftSidePoints = []
        self.rightSidePoints = []

        #generate left side points
        for i in range (len(self.cornerPoints)):
            #get center line direction
            nextIndex = (i + 1) % len(self.cornerPoints)
            dir = (self.cornerPoints[nextIndex][0] - self.cornerPoints[i][0], self.cornerPoints[nextIndex][1] - self.cornerPoints[i][1])

            #get normalized direction
            dirMag = math.sqrt(dir[0] ** 2 + dir[1] ** 2)
            dirNorm = (dir[0] / dirMag, dir[1] / dirMag)

            #rotate it by 90 left for left side
            dirLeft = (-dirNorm[1], dirNorm[0])

            #generate center line copy (as premica) offset by vector * roadthickness
            copyStart = (self.cornerPoints[i][0] + dirLeft[0] * self.roadThickness, self.cornerPoints[i][1] + dirLeft[1] * self.roadThickness)
            copyEnd = (self.cornerPoints[nextIndex][0] + dirLeft[0] * self.roadThickness, self.cornerPoints[nextIndex][1] + dirLeft[1] * self.roadThickness)

            #append to array
            self.leftSidePoints.append(copyStart)
            self.leftSidePoints.append(copyEnd)

        #generate rigfht side points
        for i in range (len(self.cornerPoints)):
            #get center line direction
            nextIndex = (i + 1) % len(self.cornerPoints)
            dir = (self.cornerPoints[nextIndex][0] - self.cornerPoints[i][0], self.cornerPoints[nextIndex][1] - self.cornerPoints[i][1])

            #get normalized direction
            dirMag = math.sqrt(dir[0] ** 2 + dir[1] ** 2)
            dirNorm = (dir[0] / dirMag, dir[1] / dirMag)

            #rotate it by 90 right for right side
            dirLeft = (dirNorm[1], -dirNorm[0])

            #generate center line copy (as premica) offset by vector * roadthickness
            copyStart = (self.cornerPoints[i][0] + dirLeft[0] * self.roadThickness, self.cornerPoints[i][1] + dirLeft[1] * self.roadThickness)
            copyEnd = (self.cornerPoints[nextIndex][0] + dirLeft[0] * self.roadThickness, self.cornerPoints[nextIndex][1] + dirLeft[1] * self.roadThickness)

            #append to array
            self.rightSidePoints.append(copyStart)
            self.rightSidePoints.append(copyEnd)

        #go through line loops and remove intersection (this leaves nonintersection corners squared of instead of pointy)
        for i in range(len(self.cornerPoints)):
            nextIndex = (i + 1) % len(self.cornerPoints)

            firstStart = self.leftSidePoints[i * 2]
            firstEnd = self.leftSidePoints[i * 2 + 1]

            secondStart = self.leftSidePoints[nextIndex * 2]
            secodnEnd = self.leftSidePoints[nextIndex * 2 + 1]

            #get line intersection
            intersection = linenItersection(firstStart, firstEnd, secondStart, secodnEnd, False)

            if intersection != None:
                self.leftSidePoints[i * 2 + 1] = intersection
                self.leftSidePoints[nextIndex * 2] = intersection

        #repeat for right side
        for i in range(len(self.cornerPoints)):
            nextIndex = (i + 1) % len(self.cornerPoints)

            firstStart = self.rightSidePoints[i * 2]
            firstEnd = self.rightSidePoints[i * 2 + 1]

            secondStart = self.rightSidePoints[nextIndex * 2]
            secodnEnd = self.rightSidePoints[nextIndex * 2 + 1]

            #get line intersection
            intersection = linenItersection(firstStart, firstEnd, secondStart, secodnEnd, False)

            if intersection != None:
                self.rightSidePoints[i * 2 + 1] = intersection
                self.rightSidePoints[nextIndex * 2] = intersection

    def drawRoad(self):
        #draw road lines
        for i in range(len(self.cornerPoints)):
            nextIndex = (i + 1) % len(self.cornerPoints)
            pygame.draw.line(screen, (120, 120, 120), self.cornerPoints[i], self.cornerPoints[nextIndex])
            
        #draw tween points
        for i in range(len(self.cornerTweenPoints)):
            pygame.draw.circle(screen, (255, 255, 255), self.cornerTweenPoints[i], 2)

        #draw corner points
        """for i in range(len(self.cornerPoints)):
            pygame.draw.circle(screen, (255, 0, 0), self.cornerPoints[i], 3)"""

        #draw sidelines
        for i in range(len(self.rightSidePoints)):
            nextIndex = (i + 1) % len(self.rightSidePoints)
            pygame.draw.line(screen, (0, 120, 0), self.leftSidePoints[i], self.leftSidePoints[nextIndex])
            pygame.draw.line(screen, (0, 120, 0), self.rightSidePoints[i], self.rightSidePoints[nextIndex])

        #draw side points
        """for i in range(len(self.rightSidePoints)):
            pygame.draw.circle(screen, (0, 255, 0), self.leftSidePoints[i], 3)
            pygame.draw.circle(screen, (0, 255, 0), self.rightSidePoints[i], 3)"""
        
        #debug draw
        """#get road edges as simpler array
        roadsides = self.leftSidePoints + self.rightSidePoints
        #for every road edge
        for j in range(len(roadsides) - 1):
            #check for intersections
            nextRoadsideIndex = (j + 1) % len(roadsides)

            isJinLeft = roadsides[j] in self.leftSidePoints
            isJNextinLeft = roadsides[nextRoadsideIndex] in self.leftSidePoints

            if isJinLeft == isJNextinLeft:
                ratio = j / len(roadsides)
                color = (255 * ratio, 0, 0)
                pygame.draw.line(screen, color, roadsides[j], roadsides[nextRoadsideIndex], 3)"""

#FUNCTION DEFINITION - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def rotatePoint(point, center, angleDeg):
    #get start vars
    angleRad = math.radians(angleDeg)
    x, y = point
    cx, cy = center

    #transalte point to origin
    tx = x - cx
    ty = y - cy

    #do some math
    rx = tx * math.cos(angleRad) - ty * math.sin(angleRad)
    ry = tx * math.sin(angleRad) + ty * math.cos(angleRad)

    #translate back to finalize
    fx = rx + cx
    fy = ry + cy

    return (fx, fy)

def linenItersection(p1, p2, p3, p4, inSegments):
    #math generated by ChatGPT

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Calculate the denominators
    denom = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1)
    if denom == 0:
        # Lines are parallel or coincident
        return None

    # Calculate the numerators
    ua = ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3)) / denom

    # Get final point coordinates
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)

    # Check if point is on line segments
    if inSegments:
        ub = ((x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3)) / denom
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            return (x, y)
        else:
            return None
    
    return (x, y)

def angleToVector(deg):
    rad = math.radians(deg)
    x = math.cos(rad)
    y = math.sin(rad)
    return (x, y)

# NEURAL NETWORK STUFF ----------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def mutate2D(weights, mutationRate = 0.2, mutationStrength = 0.03):
    new_weights = weights

    for i in range(len(new_weights)):
        for j in range(len(new_weights[i])):
            if random.random() < mutationRate:
                new_weights[i][j] += random.gauss(0, mutationStrength)

    return new_weights

def mutate1D(weights, mutationRate = 0.2, mutationStrength = 0.03):
    new_weights = weights

    for i in range(len(new_weights)):
        if random.random() < mutationRate:
            new_weights[i] += random.gauss(0, mutationStrength)

    return new_weights

#VARIABLE INITIALIZATION -----------------------------------------------------------------------------------------------------------------------------------------

road = Road()

cars = []

for i in range(40):
    car = Car(151, 673)

generation = 1
alltopcars = []

#get initial ticks
prevT = pygame.time.get_ticks()

#WHILE LOOP - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

running = True
while running:

    #update delta time
    currT = pygame.time.get_ticks()
    dTms = currT - prevT
    dTs = dTms / 1000.0

    #fill screen
    screen.fill((20, 20, 20))

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pass

        #track builder for road verts
        if event.type == pygame.MOUSEBUTTONDOWN:
            print(pygame.mouse.get_pos())

    """keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        car.throttle(dTs)
    if keys[pygame.K_a]:
        car.turn(-1, dTs)
    if keys[pygame.K_d]:
        car.turn(1, dTs)"""
    
    #update physics
    for car in cars:
        if car.dead == False:
            car.performAction(dTs)
            car.move(dTs, road)

    #other
    for car in cars:
        if car.dead == False:
            car.getClosestTweenPoint(road.cornerTweenPoints)
            car.getIntersections(road)

    #draw
    road.drawRoad()

    for car in cars:
        car.draw()

    #draw dot on elite
    pygame.draw.circle(screen, (255, 0, 0), cars[0].pos, 7, 4)

    #check if gen has finished
    allDied = False
    numOfDead = 0
    for car in cars:
        if car.dead == True:
            numOfDead += 1

    if numOfDead == len(cars):
        allDied = True

    if allDied:
        #get top of current gen
        sortedCars = sorted(cars, key=lambda car: car.fitness)
        topCars = copy.deepcopy(sortedCars[-5:])

        #add to all time top
        #alltopcars.extend(topCars)                                                            #straight copy copies objects which get altered, deep copy copies object traits which persist
        alltopcars.extend([copy.deepcopy(car) for car in topCars])

        #choose parent randomly from top of all time top for only best traits so far
        alltopcars = sorted(alltopcars, key=lambda car: car.fitness)
        chosenNext = copy.deepcopy(alltopcars[-5:])

        print("------ Top 5 best performing all time: ------")
        for car in chosenNext:
            print(car.fitness)
        print(str(sortedCars[-1].fitness) + " (best this gen)" )

        #introduce elitism
        cars[0] = copy.deepcopy(alltopcars[-1])

        #get generations performance comparison to all time, generate mutation parameters
        bestCurr = sortedCars[-1].fitness
        bestAllTime = alltopcars[-1].fitness

        mutationRate = 0.5
        mutationStrength = 0.5

        if bestCurr > bestAllTime:      #better, stay consistent with slight improvements
            mutationRate = 0.01
            mutationStrength = 0.1
            print("perfect")
        elif bestCurr == bestAllTime:   #same, induce exploration
            mutationRate = 0.2
            mutationStrength = 0.2
            print("explore")
        else:                           #worse, induce large mutations
            mutationRate = 0.2
            mutationStrength = 0.4
            print("mutate")

        #copy weights of best car to all other and mutate
        for car in cars[1:]:    #skip first car, elite)
            randomParent = chosenNext[random.randint(0, len(chosenNext)-1)]

            car.nn.weights_inputToHidden = mutate2D(copy.deepcopy(randomParent.nn.weights_inputToHidden), mutationRate, mutationStrength)
            car.nn.weights_hiddenToHidden = mutate2D(copy.deepcopy(randomParent.nn.weights_hiddenToHidden), mutationRate, mutationStrength)
            car.nn.weights_hiddenToOutput = mutate2D(copy.deepcopy(randomParent.nn.weights_hiddenToOutput), mutationRate, mutationStrength)

            car.nn.bias_hidden = mutate1D(copy.deepcopy(randomParent.nn.bias_hidden), mutationRate, mutationStrength)
            car.nn.bias_hidden2 = mutate1D(copy.deepcopy(randomParent.nn.bias_hidden2), mutationRate, mutationStrength)
            car.nn.bias_output = mutate1D(copy.deepcopy(randomParent.nn.bias_output), mutationRate, mutationStrength)

        #reinit cars
        for car in cars:
            car.dead = False
            car.life = 20 * 60
            car.pos = (151, 673)
            car.vel = (0, 0)
            car.angle = 0

        generation += 1

    gen_text = font.render(f"{generation}", True, (255, 255, 255))
    screen.blit(gen_text, (10, 40))

    # Update the display (buffer flip)
    displayFPS(screen, 25)
    pygame.display.flip()
    clock.tick(60)

    #update delta time
    prevT = currT

# Quit Pygame
pygame.quit()
