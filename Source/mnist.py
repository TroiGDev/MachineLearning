
#todo:
#shuffle data instead of randomly sampling
#log otal accuracy
#different backpropogation implementation (possibly external libs)

#many more epochs

import pygame

import math
import random

import time

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'       #the os import and this call are to remove an annoying terminal massage about floating point accuracy

#get dataset
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

images = []
labels = []
for i in range(len(train_dataset)):
    image, label = train_dataset[i]
    images.append(image.squeeze().tolist())
    labels.append(label)

#initialize pygame window
pygame.init()
screenWidth = 700
screenHeight = 700

resolution = 28
tileSize = screenWidth /resolution

screen = None
def initializeWindowDisplay():
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    pygame.display.set_caption('MNIST digit recognition')
    return screen

#fps display
clock = pygame.time.Clock()
def displayFPS(screen, font_size):
    font = pygame.font.SysFont(None, font_size)
    fps = round(clock.get_fps(), 1)
    fps_text = font.render(f"{fps}", True, (255, 255, 255))
    screen.blit(fps_text, (10, 10))

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
            hidden.append(self.ReLU(activation))

        #get hidden2 layer calculations
        hidden2 = []
        for i in range(len(self.weights_hiddenToHidden)):
            activation = sum(w * inp for w, inp in zip(self.weights_hiddenToHidden[i], hidden)) + self.bias_hidden2[i]
            hidden2.append(self.ReLU(activation))

        #get output layer calculations
        outputs = []
        activations = []
        for i in range(len(self.weights_hiddenToOutput)):
            activation = sum(w * h for w, h in zip(self.weights_hiddenToOutput[i], hidden2)) + self.bias_output[i]
            activations.append(activation)

            #replaced by softmax
            """outputs.append(self.sigmoid(activation))"""

        #use softmax instead of sigmoid for last layer
        exp_outputs = [math.exp(o) for o in activations]
        sum_exp = sum(exp_outputs)
        outputs = [eo / sum_exp for eo in exp_outputs]

        #return all layer outputs for correct graident calculations in elarning
        return hidden, hidden2, outputs
    
    """def sigmoid(self, x):
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x)
            return z / (1 + z)
    
    def sigmoidDerivative(self, x):
        return x * (1 - x)"""
    
    def ReLU(self, x):
        if isinstance(x, list):
            return [self.ReLU(i) for i in x]
        else:
            return max(0, x)
        
    def ReLUDerivative(self, x):
        if isinstance(x, list):
            return [self.ReLUDerivative(i) for i in x]
        else:
            return 1.0 if x > 0 else 0.0
    
    def loadWeights(self, filename):
        with open(filename, "r") as file:
            content = file.readlines()

            #split first line into rows by comma
            content[0] = content[0].rstrip()    #remove newline char
            arr = content[0].split(",")
            pArr = []
            for row in arr[:-1]:
                #split each row into array and add it to parent array
                pArr.append([float(x) for x in row.split(" ")])

            #apply parent array to nn var weights
            self.weights_inputToHidden = pArr

            #split first line into rows by comma
            content[1] = content[1].rstrip()    #remove newline char
            arr = content[1].split(",")
            pArr = []
            for row in arr[:-1]:
                #split each row into array and add it to parent array
                pArr.append([float(x) for x in row.split(" ")])

            #apply parent array to nn var weights
            self.weights_hiddenToHidden = pArr

            #split first line into rows by comma
            content[2] = content[2].rstrip()    #remove newline char
            arr = content[2].split(",")
            pArr = []
            for row in arr[:-1]:
                #split each row into array and add it to parent array
                pArr.append([float(x) for x in row.split(" ")])

            #apply parent array to nn var weights
            self.weights_hiddenToOutput = pArr

            #load biases
            self.bias_hidden = [float(x) for x in content[3].split(" ")]
            self.bias_hidden2 = [float(x) for x in content[4].split(" ")]
            self.bias_output = [float(x) for x in content[5].split(" ")]

            print("Loaded saved weights!")

    def saveWeights(self, filename):
        with open(filename, "w") as file:
            
            #save weights1
            fString1 = ""
            for row in self.weights_inputToHidden:
                #convert row into string, values seperated by spaces
                rowString = ' '.join(str(f) for f in row)

                #add it to fstring with a comma at the end
                fString1 += rowString + ","

            #add new line and write
            file.write(fString1 + "\n")

            #save weights2
            fString2 = ""
            for row in self.weights_hiddenToHidden:
                #convert row into string, values seperated by spaces
                rowString = ' '.join(str(f) for f in row)

                #add it to fstring with a comma at the end
                fString2 += rowString + ","

            #add new line and write
            file.write(fString2 + "\n")

            #save weigths3
            fString3 = ""
            for row in self.weights_hiddenToOutput:
                #convert row into string, values seperated by spaces
                rowString = ' '.join(str(f) for f in row)

                #add it to fstring with a comma at the end
                fString3 += rowString + ","

            #add new line and write
            file.write(fString3 + "\n")

            #save biases
            file.write(' '.join(str(f) for f in self.bias_hidden) + "\n")
            file.write(' '.join(str(f) for f in self.bias_hidden2) + "\n")
            file.write(' '.join(str(f) for f in self.bias_output) + "\n")


class Grid():
    def __init__(self):
        self.tileSize = tileSize
        self.grid = [[0 for _ in range(resolution)] for _ in range(resolution)]

        self.nn = NeuralNetwork(28*28, 128, 10)

    def draw(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):

                color = 255 * self.grid[i][j]
                if color > 255:
                    color = 255
                if color < 0:
                    color = 0
                
                pygame.draw.rect(screen, (color, color, color), pygame.Rect(j * self.tileSize, i * self.tileSize, self.tileSize, self.tileSize))

    def performRecognition(self, doPrint):
        #get 28by28 1d array of 0 to 1 floats from image
        inputs = []
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                inputs.append(self.grid[i][j])

        #get output array
        hidden1, hidden2, outputs = self.nn.iterateForward(inputs)

        #interpret output to get final number
        output = outputs.index(max(outputs))

        if doPrint:
            print("----------------------")
            print(outputs)
            print("Recognized digit " + str(output))
            print("----------------------")

        return outputs, hidden1, hidden2, output

    def learnFromExample(self):
        #get random example
        randomExampleIndex = random.randint(0, len(images) - 1)
        self.grid, label = loadTrainExample(randomExampleIndex)
        self.grid1d = [pixel for row in self.grid for pixel in row]

        #run a recognition to get outputs
        outputs, hidden1, hidden2, _ = self.performRecognition(False)

        #get accuracy:
        #get onehot encoding of label (from label digit to array of correct expected output)
        oneHot = [1 if i == label else 0 for i in range(10)]

        #cross entropy loss calculation 
        epsilon = 1e-15
        loss = 0.0
        for i in range(10):
            clipped = min(max(outputs[i], epsilon), 1 - epsilon)
            loss += -oneHot[i] * math.log(clipped)

        #backpropagation gradient calculation for each layers output individualy

        #note: from this point on the pressence of godly math magic i simply dont understand is apparent---------------------------------------------------------------------------------------------------------
        
        # Output layer delta (gradient)
        outputGradient = [outputs[i] - oneHot[i] for i in range(10)]

        # Gradients for hidden2 to output weights and output biases
        d_weights_hiddenToOutput = [[outputGradient[o] * hidden2[h] for h in range(len(hidden2))] for o in range(len(outputGradient))]
        d_bias_output = outputGradient[:]

        # Backprop to hidden layer 2
        delta_hidden2 = []
        for h in range(len(hidden2)):
            error = sum(outputGradient[o] * self.nn.weights_hiddenToOutput[o][h] for o in range(len(outputGradient)))
            delta_hidden2.append(error * self.nn.ReLUDerivative(hidden2[h]))

        # Gradients for hidden1 to hidden2 weights and hidden2 biases
        d_weights_hiddenToHidden = [[delta_hidden2[h2] * hidden1[h1] for h1 in range(len(hidden1))] for h2 in range(len(delta_hidden2))]
        d_bias_hidden2 = delta_hidden2[:]

        # Backprop to hidden layer 1
        delta_hidden1 = []
        for h in range(len(hidden1)):
            error = sum(delta_hidden2[h2] * self.nn.weights_hiddenToHidden[h2][h] for h2 in range(len(delta_hidden2)))
            delta_hidden1.append(error * self.nn.ReLUDerivative(hidden1[h]))

        # Gradients for input to hidden1 weights and hidden1 biases
        d_weights_inputToHidden = [[delta_hidden1[h] * self.grid1d[i] for i in range(len(self.grid1d))]for h in range(len(delta_hidden1))]
        d_bias_hidden = delta_hidden1[:]

        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        #weight update using gradient descent
        learningRate = 0.01       #for 10k-50k training examples

        # Update input-to-hidden weights
        for i in range(len(self.nn.weights_inputToHidden)):
            for j in range(len(self.nn.weights_inputToHidden[i])):
                self.nn.weights_inputToHidden[i][j] -= learningRate * d_weights_inputToHidden[i][j]

        # Update hidden-to-hidden weights
        for i in range(len(self.nn.weights_hiddenToHidden)):
            for j in range(len(self.nn.weights_hiddenToHidden[i])):
                self.nn.weights_hiddenToHidden[i][j] -= learningRate * d_weights_hiddenToHidden[i][j]

        # Update hidden-to-output weights
        for i in range(len(self.nn.weights_hiddenToOutput)):
            for j in range(len(self.nn.weights_hiddenToOutput[i])):
                self.nn.weights_hiddenToOutput[i][j] -= learningRate * d_weights_hiddenToOutput[i][j]

        # Update biases
        for i in range(len(self.nn.bias_hidden)):
            self.nn.bias_hidden[i] -= learningRate * d_bias_hidden[i]

        for i in range(len(self.nn.bias_hidden2)):
            self.nn.bias_hidden2[i] -= learningRate * d_bias_hidden2[i]

        for i in range(len(self.nn.bias_output)):
            self.nn.bias_output[i] -= learningRate * d_bias_output[i]


#FUNCTION DEFINITION - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def loadTrainExample(i):
    example = images[i]
    label = int(labels[i])

    return example, label

def formatTime(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

#VARIABLE INITIALIZATION -----------------------------------------------------------------------------------------------------------------------------------------

#get grid
grid = Grid()

#filesacving weights after learning
filename = "weights6.txt"
reLearn = False

#initial learn from examples
if reLearn:

    numOfEpochs = 2.5      #r1 epoch - 1 full dataset
    numOfExamplesToTrainFrom = int(numOfEpochs * len(images))

    startTime = time.time()

    for i in range(numOfExamplesToTrainFrom):
        try:
            grid.learnFromExample()

            #some loading printing
            if i % 50 == 0:
                #print precantage finished
                perc = str(round((i / numOfExamplesToTrainFrom * 100), 2)) + "%"
                bar = "■" * round(((i / numOfExamplesToTrainFrom) * 50))
                emptyBar = "□" * (50 - len(bar))

                #get  remaining time estimate
                elapsedTime = time.time() - startTime
                avgTimePerIteration = elapsedTime / (i+1)
                remainingSeconds = avgTimePerIteration * (numOfExamplesToTrainFrom - (i+1))

                print(f"\r" + perc + " " + bar + emptyBar + " " + formatTime(remainingSeconds), end='')

        except Exception as e:
            grid.nn.saveWeights(filename)

            print(e)

    print("Finished!")

    #save resulted weights
    grid.nn.saveWeights(filename)
else:
    grid.nn.loadWeights(filename)

#clear grid after last train example
grid.grid = [[0 for _ in range(resolution)] for _ in range(resolution)]

#create window after learning
screen = initializeWindowDisplay()

"""grid.grid, p = loadTrainExample(random.randint(0, len(images) - 1))
print(p)"""

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
            if event.key == pygame.K_c:
                grid.grid = [[0 for _ in range(resolution)] for _ in range(resolution)]
        
        if event.type == pygame.MOUSEBUTTONUP:
            #print recognised digit
            grid.performRecognition(True)

    """keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        pass"""
    
    buttons = pygame.mouse.get_pressed()
    if buttons[0]:
        screenPos = pygame.mouse.get_pos()
        gridPos = (screenPos[1] // grid.tileSize, screenPos[0] // grid.tileSize)

        gridPos = (int(clamp(gridPos[0], 0, resolution)), int(clamp(gridPos[1], 0, resolution)))

        if(gridPos[0] >= 0 and gridPos[0] < resolution) and (gridPos[1] >= 0 and gridPos[1] < resolution):
            grid.grid[gridPos[0]][gridPos[1]] += 0.6

            dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            for dir in dirs:
                if (gridPos[0] + dir[0] >= 0 and gridPos[0] + dir[0] < resolution) and (gridPos[1] + dir[1] >= 0 and gridPos[1] + dir[1] < resolution):
                    grid.grid[gridPos[0] + dir[0]][gridPos[1] + dir[1]] += 0.3

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
