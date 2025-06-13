import pygame

import math
import random

#initialize pygame window
pygame.init()
screenWidth = 700
screenHeight = 700

resolution = 28
tileSize = screenWidth /resolution

screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption('MNIST digit recognition')

#fps display
clock = pygame.time.Clock()
def displayFPS(screen, font_size):
    font = pygame.font.SysFont(None, font_size)
    fps = round(clock.get_fps(), 1)
    fps_text = font.render(f"{fps}", True, (255, 255, 255))
    screen.blit(fps_text, (10, 10))

#CLASS DEFINITION -----------------------------------------------------------------------------------------------------------------------------------------

class Grid():
    def __init__(self):
        self.tileSize = tileSize
        self.grid = [[0 for _ in range(resolution)] for _ in range(resolution)]

    def draw(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                
                if self.grid[i][j] > 1:
                    self.grid[i][j] = 1
                
                pygame.draw.rect(screen, (255 * self.grid[i][j], 255 * self.grid[i][j], 255 * self.grid[i][j]), pygame.Rect(i * self.tileSize, j * self.tileSize, self.tileSize, self.tileSize))

#FUNCTION DEFINITION - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

#VARIABLE INITIALIZATION -----------------------------------------------------------------------------------------------------------------------------------------

#get grid
grid = Grid()

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

        """if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pass"""

    """keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        pass"""
    
    buttons = pygame.mouse.get_pressed()
    if buttons[0]:
        screenPos = pygame.mouse.get_pos()
        gridPos = (screenPos[0] // grid.tileSize, screenPos[1] // grid.tileSize)

        gridPos = (int(clamp(gridPos[0], 0, resolution)), int(clamp(gridPos[1], 0, resolution)))

        grid.grid[gridPos[0]][gridPos[1]] += 0.4

    #drwa grid
    grid.draw()

    # Update the display (buffer flip)
    displayFPS(screen, 25)
    pygame.display.flip()
    clock.tick(60)

    #update delta time
    prevT = currT

# Quit Pygame
pygame.quit()
