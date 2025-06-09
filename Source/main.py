import pygame

import math
import random

#initialize pygame window
pygame.init()
screenWidth = 800
screenHeight = 600
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
            (0, 2), 
            (-7, 7)
        ]
        self.verts = [
            (self.pos[0] + 0, self.pos[1] - 15), 
            (self.pos[0] + 7, self.pos[1] + 7), 
            (self.pos[0] + 0, self.pos[1] + 2), 
            (self.pos[0] - 7, self.pos[1] + 7)
        ]

    def updateVerts(self):
        for i in range(len(self.verts)):
            #update pos
            self.verts[i] = (self.pos[0] + self.vertOffsets[i][0], self.pos[1] + self.vertOffsets[i][1])

            #rotate reinitialized verts
            self.verts[i] = rotatePoint(self.verts[i], self.pos, self.angle)

    def draw(self):
        self.updateVerts()
        pygame.draw.polygon(screen, (255, 255, 255), self.verts, 2)

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

#VARIABLE INITIALIZATION -----------------------------------------------------------------------------------------------------------------------------------------

car = Car(screenWidth/2, screenHeight/2)

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

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        car.throttle(dTs)
    if keys[pygame.K_a]:
        car.turn(-1, dTs)
    if keys[pygame.K_d]:
        car.turn(1, dTs)
    
    #update physics
    car.move(dTs)

    #draw
    car.draw()

    # Update the display (buffer flip)
    displayFPS(screen, 25)
    pygame.display.flip()
    clock.tick(60)

    #update delta time
    prevT = currT

# Quit Pygame
pygame.quit()
