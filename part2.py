import numpy as np
from PIL import Image

def convertToInt(strings):
    return list(map(int, strings))

def computeResult(yin, theta):
    if yin > theta:
        return 1
    if yin < -theta:
        return -1
    return 0

def perceptronModel(inputs, targets, weights, alpha, theta):
    flag = True
    apoches = 0
    wrongResult = 0
    error = []
    while flag:
        flag = False
        apoches += 1
        t = 0
        wrongResult = 0
        for input in inputs:
            for w in range(len(weights)):
                yin = np.array(input).dot(np.array(weights[w]).transpose())
                y = computeResult(yin,theta)
                if y != targets[t][w]:
                    wrongResult += 1
                    flag = True
                    for j in range(len(weights[w])):
                        weights[w][j] += alpha*input[j]*targets[t][w]
            #print(t,y,targets[t])
            t += 1
        print("Error in Epoch no. " + str(apoches) + ":",wrongResult/(len(inputs)*3)*100)
    print("Model Trained!!!")
    return weights

def testing(im, weights,theta):
    pix = convertToPixel(im)
    #bias input
    pix.append(1)
    y = []
    for w in range(len(weights)):
        yin = np.array(input).dot(np.array(weights[w]).transpose())
        y.append(computeResult(yin,theta))
    if y[0] == 1 and y[1] == -1 and y[2] == -1:
        print("Image of K")
    elif y[0] == -1 and y[1] == 1 and y[2] == -1:
        print("Image of L")
    elif y[0] == -1 and y[1] == -1 and y[2] == 1:
        print("Image of M")
    else:
        print("Not K, L or M")

def convertToPixel(im):
    newsize = (100, 100) 
    img_file = im.resize(newsize) 
    img_grey = img_file.convert('L')
    
    rawData = img_grey.load()
    #converting img to byte array: 1px = 1 byte
    data = []
    for y in range(100):
        for x in range(100):
            data.append(rawData[x,y])
    #converting each byte to 8 bits
    pixels = []
    for d in data:
        strings = list(bin(d)[2:].zfill(8))
        pixels.extend(convertToInt(strings))
    return pixels


def main():    
    #input samples 
    inputs = []
    for i in range(30):
        im = Image.open("D:\\University\\7th Semester\\NN\\Assignment4\\dataset\\images\\" + str(i) + ".png")
        pix = convertToPixel(im)
        #bias input
        pix.append(1)
        inputs.append(pix)
    #targets
    targets = []
    with open("D:\\University\\7th Semester\\NN\\Assignment4\\dataset\\targets\\multiClassificationTargets.txt") as targetFile:
        while True:
            line = targetFile.readline()
            if not line:
                break
            line = line[:-1]
            line = line.split(',')
            line = convertToInt(line)
            targets.append(line)
    #weights
    rows, cols = (3, 80001) #col size 80,000 weights + b value 
    weights = [[0]*cols]*rows 
    alpha = 0.5
    theta = 0.1
    #training
    weights = perceptronModel(inputs, targets, weights, alpha, theta)
    #Testing
    print("Testing....")
    im = Image.open("D:\\University\\7th Semester\\NN\\Assignment4\\dataset\\images\\9.png")
    print("image of K?")
    testing(im,weights,theta)
    im = Image.open("D:\\University\\7th Semester\\NN\\Assignment4\\dataset\\images\\12.png")
    print("image of L?")
    testing(im,weights,theta)
    im = Image.open("D:\\University\\7th Semester\\NN\\Assignment4\\dataset\\images\\25.png")
    print("image of M?")
    testing(im,weights,theta)
    im = Image.open("D:\\University\\7th Semester\\NN\\Assignment4\\dataset\\img.png")
    print("image of any other character?")
    testing(im,weights,theta)
main()