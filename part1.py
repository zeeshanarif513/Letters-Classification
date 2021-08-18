import numpy as np
from PIL import Image

def convertToInt(strings):
    return list(map(int, strings))

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

def computeResult(yin, theta):
    if yin > theta:
        return 1
    if yin < -theta:
        return -1
    return 0

def perceptronModel(inputs, targets, weights, alpha, theta):
    flag = True
    apoches = 0
    error = []
    while flag:
        flag = False
        apoches += 1
        t = 0
        wrongResult = 0
        for input in inputs:
            yin = np.array(input).dot(np.array(weights).transpose())
            y = computeResult(yin,theta)
            if y != targets[t]:
                flag = True
                for w in range(len(weights)):
                    weights[w] += alpha*input[w]*targets[t]
            t += 1
    print("Model Trained!!!")
    return weights

def testing(im, weights,theta):
    pix = convertToPixel(im)
    #bias input
    pix.append(1)
    yin = np.array(pix).dot(np.array(weights).transpose())
    y = computeResult(yin,theta)
    if y == 1:
        print("positive instance")
    else:
        print("negative instance")

def trainingAndTestingForK(inputs):
    #targets
    targets = []
    with open("D:\\University\\7th Semester\\NN\\Assignment4\\dataset\\targets\\K.txt") as targetFile:
        while True:
            line = targetFile.readline()
            if not line:
                break
            targets.append(line[:-1])
    targets = convertToInt(targets)    

    #weights
    weights = [0]*80000
    #adding bias input weight
    weights.append(0)

    alpha = 1
    theta = 0.1

    #training
    weights = perceptronModel(inputs, targets, weights, alpha, theta)
    #testing
    print("Testing....")
    im = Image.open("D:\\University\\7th Semester\\NN\\Assignment4\\dataset\\images\\6.png")
    print("image of K?")
    testing(im,weights,theta)
    im = Image.open("D:\\University\\7th Semester\\NN\\Assignment4\\dataset\\images\\12.png")
    print("image of L?")
    testing(im,weights,theta)
    im = Image.open("D:\\University\\7th Semester\\NN\\Assignment4\\dataset\\images\\25.png")
    print("image of M?")
    testing(im,weights,theta)

    
def trainingAndTestingForL(inputs):
    #targets
    targets = []
    with open("D:\\University\\7th Semester\\NN\\Assignment4\\dataset\\targets\\L.txt") as targetFile:
        while True:
            line = targetFile.readline()
            if not line:
                break
            targets.append(line[:-1])
    targets = convertToInt(targets)    

    #weights
    weights = [0]*80000
    #adding bias input weight
    weights.append(0)

    alpha = 1
    theta = 0.1

    #training
    weights = perceptronModel(inputs, targets, weights, alpha, theta)
    #testing
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

def trainingAndTestingForM(inputs):
    #targets
    targets = []
    with open("D:\\University\\7th Semester\\NN\\Assignment4\\dataset\\targets\\M.txt") as targetFile:
        while True:
            line = targetFile.readline()
            if not line:
                break
            targets.append(line[:-1])
    targets = convertToInt(targets)    

    #weights
    weights = [0]*80000
    #adding bias input weight
    weights.append(0)

    alpha = 1
    theta = 0.1

    #training
    weights = perceptronModel(inputs, targets, weights, alpha, theta)
    #testing
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

def main():    
    #input samples 
    inputs = []
    for i in range(30):
        im = Image.open("D:\\University\\7th Semester\\NN\\Assignment4\\dataset\\images\\" + str(i) + ".png")
        pix = convertToPixel(im)
        #bias input
        pix.append(1)
        inputs.append(pix)
    print("Separate(one output class) training for K!!!")
    #separate(one output class) training for K
    trainingAndTestingForK(inputs)
    
    print("Separate(one output class) training for L!!!")
    #separate(one output class) training for L
    trainingAndTestingForL(inputs)
    
    print("Separate(one output class) training for M!!!")
    #separate(one output class) training for M
    trainingAndTestingForM(inputs)

main()
