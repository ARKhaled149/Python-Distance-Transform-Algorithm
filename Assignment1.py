from re import I
from scipy.io import loadmat
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pandas
from pandas import Series
import cv2
from sklearn.utils import shuffle

img = cv2.imread('Suez Canal.png')

def distance_transform_helper(img, distance_measure, img_type):
    d_twoships = 0
    d_bigship_leftcanal = 0
    d_bigship_right_canal = 0
    d_smallship_leftcanal = 0
    d_smallship_rightcanal = 0
    shift = 0

    h,w = img.shape
    img = np.array(img, dtype = np.float32)
    for row in range(0,h):
        for col in range(0,w):
            if img[row, col] == 0:
                img[row, col] = np.inf
            elif img[row, col] == 255:
                img[row, col] = 1
 
    # first pass
    for row in range(1,h-1):
        for col in range(1,w-1):   
            kernel=(img[row-1,col-1],img[row-1,col],img[row,col-1],img[row+1,col-1],img[row,col])
            m=min(kernel)
            idx=kernel.index(m)
            if distance_measure == 'cityblock':
                if idx == 0:
                    img[row,col]=img[row-1,col-1] + 2
                elif idx == 1:
                    img[row,col]=img[row-1,col] + 1
                elif idx == 2:
                    img[row,col]=img[row,col-1] + 1
                elif idx == 3:
                    img[row,col]=img[row+1,col-1] + 2

            if distance_measure == 'chessboard':
                if idx == 0:
                    img[row,col]=img[row-1,col-1] + 1
                elif idx == 1:
                    img[row,col]=img[row-1,col] + 1
                elif idx == 2:
                    img[row,col]=img[row,col-1] + 1
                elif idx == 3:
                    img[row,col]=img[row+1,col-1] + 1
            
            if distance_measure == 'euclidean':
                if idx == 0:
                    img[row,col]=img[row-1,col-1] + np.sqrt(np.power((row - (row-1)), 2) + np.power((col - (col-1)), 2))
                elif idx == 1:
                    img[row,col]=img[row-1,col] + np.sqrt(np.power((row - (row-1)), 2) + np.power((col - col), 2))
                elif idx == 2:
                    img[row,col]=img[row,col-1] + np.sqrt(np.power((row - row), 2) + np.power((col - (col-1)), 2))
                elif idx == 3:
                    img[row,col]=img[row+1,col-1] + np.sqrt(np.power((row - (row+1)), 2) + np.power((col - (col-1)), 2))

    first_pass = img.copy()

    #second pass                 
    for row in range(h-2,0,-1):
        for col in range(w-2,0,-1):      
            kernel = (img[row+1,col+1],img[row+1,col],img[row,col+1],img[row-1,col+1],img[row,col])
            m = min(kernel)       
            idx = kernel.index(m)
            if distance_measure == 'cityblock':
                if idx == 0:
                    img[row,col]=img[row+1,col+1] + 2
                elif idx == 1:
                    img[row,col]=img[row+1,col] + 1
                elif idx == 2:
                    img[row,col]=img[row,col+1] + 1
                elif idx == 3:
                    img[row,col]=img[row-1,col+1] + 2

            if distance_measure == 'chessboard':
                if idx == 0:
                    img[row,col]=img[row+1,col+1] + 1
                elif idx == 1:
                    img[row,col]=img[row+1,col] + 1
                elif idx == 2:
                    img[row,col]=img[row,col+1] + 1
                elif idx == 3:
                    img[row,col]=img[row-1,col+1] + 1

            if distance_measure == 'euclidean':
                if idx == 0:
                    img[row,col]=img[row+1,col+1] + np.sqrt(np.power((row - (row+1)), 2) + np.power((col - (col+1)), 2))
                elif idx == 1:
                    img[row,col]=img[row+1,col] + np.sqrt(np.power((row - (row+1)), 2) + np.power((col - col), 2))
                elif idx == 2:
                    img[row,col]=img[row,col+1] + np.sqrt(np.power((row - row), 2) + np.power((col - (col+1)), 2))
                elif idx == 3:
                    img[row,col]=img[row-1,col+1] + np.sqrt(np.power((row - (row-1)), 2) + np.power((col - (col+1)), 2))

    if distance_measure == 'cityblock':
        shift = abs(150-200) + abs(200-200)
    if distance_measure == 'chessboard':
        shift = max(abs(150-200),abs(200-200))
    if distance_measure == 'euclidean':
        shift = np.sqrt(np.power((150-200),2) + np.power((200-200),2))

    if img_type == 'bigship':
        d_bigship_right_canal = img[150,292]
        d_bigship_leftcanal = img[150,151]
        d_twoships = img[310,175] + shift
        return first_pass, img, d_twoships, d_bigship_leftcanal, d_bigship_right_canal

    elif img_type == 'smallship':
        d_smallship_rightcanal = img[310,289]
        d_smallship_leftcanal = img[310,132]
        return first_pass, img, d_smallship_leftcanal, d_smallship_rightcanal

    else:
        return first_pass, img

def salt_pepper(prob):
    # Extract image dimensions
    row, col,_ = img.shape
    # Declare salt & pepper noise ratio
    s_vs_p = 0.5
    output = np.copy(img)
    # Apply salt noise on each pixel individually
    num_salt = np.ceil(prob * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
        for i in img.shape]
    output[coords] = 1
    # Apply pepper noise on each pixel individually
    num_pepper = np.ceil(prob * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in img.shape]
    output[coords] = 0
    return output

img = salt_pepper(0.5)
# cv2.imwrite('NoiseFil.bmp', img)


kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])

def distance_transform_algo(img, distance_measure, Dist):
    # pre-processing steps
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)    
    img = cv2.filter2D(img, -1, kernel_sharpening)
    img = cv2.Canny(img, 50, 250)
    cv2.imwrite('Preprocessing.bmp', img)
    
    h,w = img.shape
    img0 = img.copy()

    first_pass, Final_output = distance_transform_helper(img0, distance_measure,'')

    #get points of interest
    leftcanal_bigship = 0
    rightcanal_bigship = 0
    leftcanal_smallship = 0
    rightcanal_smallship = 0
    leftside_bigship = 0
    rightside_bigship = 0
    leftside_smallship = 0
    rightside_smallship = 0

    big_ship = img.copy()
    small_ship = img.copy()

    for j in range(200,w):
        if j in range(205,215) and big_ship[150,j] == 255:
            rightside_bigship = (150,j)
            # print("Point of the right side of the bigship", rightside_bigship)
        if j in range(220,300) and big_ship[150,j] == 255:
            rightcanal_bigship = (150,j)
            # print("Point of the right side of the canal w.r.t bigship",rightcanal_bigship)

    for j in range(200,0,-1):
        if j in range(160,194) and big_ship[150,j] == 255:
            leftside_bigship = (150,j)
            # print("Point of the left side of the bigship",leftside_bigship)
        if j in range(150,159) and big_ship[150,j] == 255:
            leftcanal_bigship = (150,j)
            # print("Point of the left side of the canal w.r.t bigship",leftcanal_bigship)

    for j in range(175,w):
        if j in range(180,200) and small_ship[310,j] == 255:
            rightside_smallship = (310,j)
            # print("Point of the right side of the smallship",rightside_smallship)
        if j in range(210,290) and small_ship[310,j] == 255:
            rightcanal_smallship = (310,j)
            # print("Point of the right side of the canal w.r.t smallship",rightcanal_smallship)

    for j in range(175,0,-1):
        if j in range(150,175) and small_ship[310,j] == 255:
            leftside_smallship = (310,j)
            # print("Point of the left side of the smallship",leftside_smallship)
        if j in range(130,140) and small_ship[310,j] == 255:
            leftcanal_smallship = (310,j)
            # print("Point of the left side of the canal w.r.t smallship",leftcanal_smallship)

    #isolate the big ship alone
    for i in range(0,h):
        for j in range(0,w):
            if j in range(0,190) or j in range(280,w):
                big_ship[i,j] = 0

    bigship_firstpass,bigship_output,d_twoships,d_bigship_leftcanal,d_bigship_rightcanal = distance_transform_helper(big_ship, distance_measure, 'bigship')
    print("Distance between 2 ships = " + str(d_twoships) + '\n' + 
        "Distcance of Big Ship with Left Canal = " + str(d_bigship_leftcanal) + '\n' + 
        "Distcance of Big Ship with Right Canal = " +str(d_bigship_rightcanal) + '\n')
    # Isolate the small ship alone in the image
    for i in range(0,h):
        for j in range(0,w):
            if j in range(0,172) or j in range(280,w):
                small_ship[i,j] = 0
            if i in range(0,200) and j in range(165,280):
                small_ship[i,j] = 0
    smallship_firstpass,smallship_output,d_smallship_leftcanal,d_smallship_rightcanal = distance_transform_helper(small_ship, distance_measure, 'smallship')
    print("Distance of Small Ship with Left Canal = " + str(d_smallship_leftcanal) + '\n' 
        + "Distance of Small Ship with Right Canal =" + str(d_smallship_rightcanal) + '\n')
    
    Dist.extend([d_twoships,d_bigship_leftcanal,d_bigship_rightcanal,d_smallship_leftcanal,d_smallship_rightcanal])
    
    f = open('Dist.txt','a+')    
    for m in range(len(Dist)):
        f.write(str(Dist[m]) + '\n')
        shuffle(Dist)
    f.close()
    
    return first_pass, Final_output

Dist = ["CityBlock"]
first_pass1, city = distance_transform_algo(img, 'cityblock', Dist)
cv2.imwrite('Suez_1_City.bmp', first_pass1)
cv2.imwrite('Suez_final_City.bmp', city)

Dist = ["ChessBoard"]
first_pass2, chess = distance_transform_algo(img, 'chessboard', Dist)
cv2.imwrite('Suez_1_Chess.bmp', first_pass2)
cv2.imwrite('Suez_final_Chess.bmp', chess)

Dist = ["Euclidean"]
first_pass3, euc = distance_transform_algo(img, 'euclidean', Dist)
cv2.imwrite('Suez_1_Euclidean.bmp', first_pass3)
cv2.imwrite('Suez_final_Euclidean.bmp', euc) 

# //////////////////////////////////////////////////////////////////////// Part 2 ////////////////////////////////////////////////
img2 = cv2.imread("GUC.jpg")

def pre_process(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift

def low_pass_filter(ft_img, filter_type, filter_order, d0):
    M, N = ft_img.shape[0:2]
    h = np.zeros(ft_img.shape)
    if filter_type == "ILPF":
        for u in range(len(ft_img)):
            for v in range(len(ft_img[u])):
                d_uv = np.sqrt(np.power((u - (M/2)), 2) + np.power((v - (N/2)), 2))
                if d_uv < d0:
                    h[u][v] = 1
    if filter_type == "BLPF":
        n = filter_order
        for u in range(len(ft_img)):
            for v in range(len(ft_img[u])):
                d_uv = np.sqrt(np.power((u - (M/2)), 2) + np.power((v - (N/2)), 2))
                den_power = np.power((d_uv/d0), 2 * n)
                den = 1 + den_power
                h[u][v] = 1/den
    if filter_type == "GLPF":
        n = filter_order
        for u in range(len(ft_img)):
            for v in range(len(ft_img[u])):
                d_uv = np.sqrt(np.power((u - (M/2)), 2) + np.power((v - (N/2)), 2))
                h[u][v] = np.exp((-1) * (np.power(d_uv, 2)/(2 * np.power(d0, 2))))
    return h

def filter_img(img):
    ft_img = pre_process(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    h = low_pass_filter(ft_img, "GLPF", 1, 50)
    result = ft_img * h
    inverse_ft_result = np.fft.ifftshift(result)
    inverse_ft_result = np.fft.ifft2(inverse_ft_result)
    result_img = np.zeros(inverse_ft_result.shape)
    for i in range(len(inverse_ft_result)):
        for j in range(len(inverse_ft_result[i])):
            result_img[i][j] = abs(inverse_ft_result[i][j].real)
    print(result_img)
    return result_img

result = filter_img(img2)
cv2.imwrite("GUC_GLPF_50.jpg", result)
print(result.shape)