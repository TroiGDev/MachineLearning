import pygame

import math
import random

#initialize pygame window
pygame.init()
screenWidth = 800
screenHeight = 600
screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption('<Title>')

#fps display
clock = pygame.time.Clock()
def displayFPS(screen, font_size):
    font = pygame.font.SysFont(None, font_size)
    fps = round(clock.get_fps(), 1)
    fps_text = font.render(f"{fps}", True, (255, 255, 255))
    screen.blit(fps_text, (10, 10))

#CLASS DEFINITION -----------------------------------------------------------------------------------------------------------------------------------------

#FUNCTION DEFINITION - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import matplotlib.pyplot as plt

class RoadOutlineGenerator:
    def __init__(self, points):
        """Initialize with centerline points."""
        self.points = np.array(points)
        self.num_points = len(points)
        
    def interpolate_points(self, num_interpolation=100):
        """Interpolate points along the centerline."""
        t = np.linspace(0, 1, num_interpolation)
        x = np.interp(t * (self.num_points - 1), 
                     range(self.num_points),
                     self.points[:, 0])
        y = np.interp(t * (self.num_points - 1), 
                     range(self.num_points),
                     self.points[:, 1])
        return np.column_stack((x, y))
    
    def calculate_normals(self, interpolated_points):
        """Calculate normal vectors at each point."""
        dx = np.gradient(interpolated_points[:, 0])
        dy = np.gradient(interpolated_points[:, 1])
        normals_x = -dy / np.sqrt(dx**2 + dy**2)
        normals_y = dx / np.sqrt(dx**2 + dy**2)
        return np.column_stack((normals_x, normals_y))
    
    def create_outline(self, thickness_func=lambda t: 1.0):
        """Create road outline with varying thickness."""
        interpolated_points = self.interpolate_points()
        normals = self.calculate_normals(interpolated_points)
        
        # Calculate varying thickness
        t = np.linspace(0, 1, len(interpolated_points))
        thickness = np.array([thickness_func(ti) for ti in t])
        
        # Create left and right edges
        left_edge = interpolated_points + normals * thickness[:, None]
        right_edge = interpolated_points - normals * thickness[:, None]
        
        return left_edge, right_edge
    
    def plot(self, thickness_func=lambda t: 1.0):
        """Plot the road outline."""
        centerline = self.points
        left_edge, right_edge = self.create_outline(thickness_func)
        
        plt.figure(figsize=(10, 6))
        
        # Plot centerline
        plt.plot(centerline[:, 0], centerline[:, 1], 'r--', label='Centerline')
        
        # Plot edges
        plt.plot(left_edge[:, 0], left_edge[:, 1], 'b-', label='Left Edge')
        plt.plot(right_edge[:, 0], right_edge[:, 1], 'b-', label='Right Edge')
        
        # Fill road area
        plt.fill(np.concatenate([left_edge[::-1], right_edge]),
                color='gray', alpha=0.2)
        
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.title('Road Outline with Varying Thickness')
        plt.savefig('road_outline.png')
        plt.close()

# Example points forming a curved road
points = np.array([
    [0, 0], [1, 0.5], [2, 1], [3, 1.5], [4, 2],
    [5, 1.5], [6, 1], [7, 0.5], [8, 0]
])

# Create road outline generator
road = RoadOutlineGenerator(points)

# Define varying thickness function
def varying_thickness(t):
    return 0.5 + 0.5 * np.sin(2 * np.pi * t)  # Oscillating thickness

# Plot with constant thickness
road.plot(lambda t: 1.0)

# Plot with varying thickness
road.plot(varying_thickness)

#VARIABLE INITIALIZATION -----------------------------------------------------------------------------------------------------------------------------------------

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
    
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(screenWidth/2 - 25, screenHeight/2 - 25, 50, 50), 2)

    # Update the display (buffer flip)
    displayFPS(screen, 25)
    pygame.display.flip()
    clock.tick(60)

    #update delta time
    prevT = currT

# Quit Pygame
pygame.quit()
