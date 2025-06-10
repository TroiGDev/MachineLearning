import pygame

import math
import random

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

#CLASS DEFINITION -----------------------------------------------------------------------------------------------------------------------------------------

class Car():
    def __init__(self, x, y):
        #physics
        self.pos = (x, y)
        self.angle = 0
        self.vel = (0, 0)
        self.friction = 0.98
        self.accelaration = 520
        self.maxVel = 520
        self.turnRate = 150

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

    def updateVerts(self):
        for i in range(len(self.verts)):
            #update pos
            self.verts[i] = (self.pos[0] + self.vertOffsets[i][0], self.pos[1] + self.vertOffsets[i][1])

            #rotate reinitialized verts
            self.verts[i] = rotatePoint(self.verts[i], self.pos, self.angle)

    def draw(self):
        self.updateVerts()
        pygame.draw.polygon(screen, (255, 255, 255), self.verts, 2)

        #draw line towards closest tween point
        if self.closestTweenPoint != None:
            pygame.draw.line(screen, (0, 0, 255), self.pos, self.closestTweenPoint)

    def move(self, dTs):
        #apply friction
        self.vel = (self.vel[0] * self.friction, self.vel[1] * self.friction)

        #apply velocity
        self.pos = (self.pos[0] + self.vel[0] * dTs, self.pos[1] + self.vel[1] * dTs)

    def throttle(self, dTs):
        #apply accelaration forward
        angleRad = math.radians(self.angle - 90)
        accelaration = (math.cos(angleRad) * self.accelaration, math.sin(angleRad) * self.accelaration)
        self.vel = (self.vel[0] + accelaration[0] * dTs, self.vel[1] + accelaration[1] * dTs)

        #clamp to max speed if neccesary
        mag = math.sqrt(self.vel[0] ** 2 + self.vel[1] ** 2)

        if mag > self.maxVel:
            nVel = (self.vel[0] / mag, self.vel[1] / mag)
            self.vel = [nVel[0] * self.maxVel, nVel[1] * self.maxVel]

    def turn(self, direction, dTs):
        #apply turn to angle
        self.angle += direction * self.turnRate * dTs

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

class Road():
    def __init__(self):
        self.cornerPoints = [
            (963, 485),
            (726, 694),
            (533, 532),
            (271, 666),
            (140, 571),
            (149, 331),
            (249, 145),
            (353, 209),
            (546, 285),
            (885, 152),
            (1064, 333)
        ]

        self.cornerTweenPoints = []
        self.tweenAccuracy = 5

        #initialize tween points
        for i in range(len(self.cornerPoints)):
            for j in range(self.tweenAccuracy):
                #get vector to next road
                nextIndex = (i + 1) % len(self.cornerPoints)
                v = (self.cornerPoints[nextIndex][0] - self.cornerPoints[i][0], self.cornerPoints[nextIndex][1] - self.cornerPoints[i][1])

                #create new tween point
                self.cornerTweenPoints.append((self.cornerPoints[i][0] + v[0] * (j / self.tweenAccuracy), self.cornerPoints[i][1] + v[1] * (j / self.tweenAccuracy)))

        #initialize track sides
        self.roadThickness = 60

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
            intersection = linenItersection(firstStart, firstEnd, secondStart, secodnEnd)

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
            intersection = linenItersection(firstStart, firstEnd, secondStart, secodnEnd)

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

def linenItersection(p1, p2, p3, p4):   #for infinetly long lines
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
    
    return (x, y)

#VARIABLE INITIALIZATION -----------------------------------------------------------------------------------------------------------------------------------------

car = Car(screenWidth/2, screenHeight/2)
road = Road()

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

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        car.throttle(dTs)
    if keys[pygame.K_a]:
        car.turn(-1, dTs)
    if keys[pygame.K_d]:
        car.turn(1, dTs)
    
    #update physics
    car.move(dTs)

    #other
    car.getClosestTweenPoint(road.cornerTweenPoints)

    #draw
    road.drawRoad()
    car.draw()

    # Update the display (buffer flip)
    displayFPS(screen, 25)
    pygame.display.flip()
    clock.tick(60)

    #update delta time
    prevT = currT

# Quit Pygame
pygame.quit()
