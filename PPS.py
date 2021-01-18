#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import os
import PIL
from PIL import Image
import numpy as np
import math
import cv2
import os
import itertools
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import extract
from IPython import get_ipython

arr = os.listdir(r"E:\\CAPSTONE PROJECT\\DATA")
ANCHOR_POINT = 6000
MIDZONE_THRESHOLD = 15000
# Features are defined here as global variables
BASELINE_ANGLE = 0.0
TOP_MARGIN = 0.0
LETTER_SIZE = 0.0
LINE_SPACING = 0.0
WORD_SPACING = 0.0
PEN_PRESSURE = 0.0
SLANT_ANGLE = 0.0
X_baseline_angle = []
X_top_margin = []
X_letter_size = []
X_line_spacing = []
X_word_spacing = []
X_pen_pressure = []
X_slant_angle = []
y_t1 = []
y_t2 = []
y_t3 = []
y_t4 = []
y_t5 = []
y_t6 = []
y_t7 = []
y_t8 = []
page_ids = []

def res(im,i): 
    basewidth = 800
    wpercent = (basewidth / float(im.size[0]))
    hsize = int((float(im.size[1]) * float(wpercent)))
    im1 = im.resize((basewidth, hsize), PIL.Image.ANTIALIAS) 
    width, height = im1.size
    left =0
    top = height/6
    right = width
    bottom =height-height/5   
    im2 = im1.crop((left, top,right,bottom))
    im2.save("C:\\Users\\Lenovo\\Desktop\\capstone\\images\\res\\"+arr[i])
    ##print(width, height)
''' function for bilateral filtering '''
def bilateralFilter(image, d):
    image = cv2.bilateralFilter(image,d,50,50)
    return image

''' function for median filtering '''
def medianFilter(image, d):
    image = cv2.medianBlur(image,d)
    return image

''' function for INVERTED binary threshold '''    
def threshold(image, t):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,image = cv2.threshold(image,t,255,cv2.THRESH_BINARY_INV)
    return image

''' function for dilation of objects in the image '''
def dilate(image, kernalSize):
    kernel = np.ones(kernalSize, np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    return image
    
''' function for erosion of objects in the image '''
def erode(image, kernalSize):
    kernel = np.ones(kernalSize, np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return image
    
''' function for finding countours and straightening them horizontally. Straightened lines will give better result with horizontal projections. '''
def straighten(image):

    global BASELINE_ANGLE
    
    angle = 0.0
    angle_sum = 0.0
    countour_count = 0
    
    # these four variables are not being used, please ignore
    positive_angle_sum = 0.0 #downward
    negative_angle_sum = 0.0 #upward
    positive_count = 0
    negative_count = 0
    
    # apply bilateral filter
    filtered = bilateralFilter(image, 3)
    ##cv2.imshow('filtered',filtered)

    # convert to grayscale and binarize the image by INVERTED binary thresholding
    thresh = threshold(filtered, 120)
    ##cv2.imshow('thresh',thresh)
    
    # dilate the handwritten lines in image with a suitable kernel for contour operation
    dilated = dilate(thresh, (5 ,100))
    ##cv2.imshow('dilated',dilated)
    
    ctrs,hier = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, ctr in enumerate(ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        
        # We can be sure the contour is not a line if height > width or height is < 20 pixels. Here 20 is arbitrary.
        if h>w or h<20:
            continue
        
        # We extract the region of interest/contour to be straightened.
        roi = image[y:y+h, x:x+w]
        #rows, cols = ctr.shape[:2]
        
        # If the length of the line is less than half the document width, especially for the last line,
        # ignore because it may yeild inacurate baseline angle which subsequently affects proceeding features.
        if w < image.shape[1]/2 :
            roi = 255
            image[y:y+h, x:x+w] = roi
            continue

        # minAreaRect is necessary for straightening
        rect = cv2.minAreaRect(ctr)
        center = rect[0]
        angle = rect[2]
        ##print "original: "+str(i)+" "+str(angle)
        # I actually gave a thought to this but hard to remember anyway!
        if angle < -45.0:
            angle += 90.0;
        ##print "+90 "+str(i)+" "+str(angle)
            
        rot = cv2.getRotationMatrix2D(((x+w)/2,(y+h)/2), angle, 1)
        #extract = cv2.warpAffine(roi, rot, (w,h), borderMode=cv2.BORDER_TRANSPARENT)
        extract = cv2.warpAffine(roi, rot, (w,h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        #cv2.imshow('warpAffine:'+str(i),extract)

        # image is overwritten with the straightened contour
        image[y:y+h, x:x+w] = extract
        '''
        # Please Ignore. This is to draw visual representation of the contour rotation.
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(display,[box],0,(0,0,255),1)
        cv2.rectangle(display,(x,y),( x + w, y + h ),(0,255,0),1)
        '''
        #print(angle)
        angle_sum += angle
        countour_count += 1
    '''    
        # sum of all the angles of downward baseline
        if(angle>0.0):
            positive_angle_sum += angle
            positive_count += 1
        # sum of all the angles of upward baseline
        else:
            negative_angle_sum += angle
            negative_count += 1
            
    if(positive_count == 0): positive_count = 1
    if(negative_count == 0): negative_count = 1
    average_positive_angle = positive_angle_sum / positive_count
    average_negative_angle = negative_angle_sum / negative_count
    #print "average_positive_angle: "+str(average_positive_angle)
    #print "average_negative_angle: "+str(average_negative_angle)
    
    if(abs(average_positive_angle) > abs(average_negative_angle)):
        average_angle = average_positive_angle
    else:
        average_angle = average_negative_angle
    
    #print "average_angle: "+str(average_angle)
    '''
    #cv2.imshow('countours', display)
    
    # mean angle of the contours (not lines) is found
    mean_angle = angle_sum / countour_count
    BASELINE_ANGLE = mean_angle
    #print ("Average baseline angle: "+str(mean_angle))
    return image

''' function to calculate horizontal projection of the image pixel rows and return it '''
def horizontalProjection(img):
    # Return a list containing the sum of the pixels in each row
    (h, w) = img.shape[:2]
    sumRows = []
    for j in range(h):
        row = img[j:j+1, 0:w] # y1:y2, x1:x2
        sumRows.append(np.sum(row))
    return sumRows
    
''' function to calculate vertical projection of the image pixel columns and return it '''
def verticalProjection(img):
    # Return a list containing the sum of the pixels in each column
    (h, w) = img.shape[:2]
    sumCols = []
    for j in range(w):
        col = img[0:h, j:j+1] # y1:y2, x1:x2
        sumCols.append(np.sum(col))
    return sumCols
    
''' function to extract lines of handwritten text from the image using horizontal projection '''
def extractLines(img):

    global LETTER_SIZE
    global LINE_SPACING
    global TOP_MARGIN
    
    # apply bilateral filter
    filtered = bilateralFilter(img, 5)
    
    # convert to grayscale and binarize the image by INVERTED binary thresholding
    # it's better to clear unwanted dark areas at the document left edge and use a high threshold value to preserve more text pixels
    thresh = threshold(filtered, 160)
    #cv2.imshow('thresh', lthresh)

    # extract a python list containing values of the horizontal projection of the image into 'hp'
    hpList = horizontalProjection(thresh)

    # Extracting 'Top Margin' feature.
    topMarginCount = 0
    for sum in hpList:
        # sum can be strictly 0 as well. Anyway we take 0 and 255.
        if(sum<=255):
            topMarginCount += 1
        else:
            break
            
    ##print "(Top margin row count: "+str(topMarginCount)+")"
            
    # FIRST we extract the straightened contours from the image by looking at occurance of 0's in the horizontal projection.
    lineTop = 0
    lineBottom = 0
    spaceTop = 0
    spaceBottom = 0
    indexCount = 0
    setLineTop = True
    setSpaceTop = True
    includeNextSpace = True
    space_zero = [] # stores the amount of space between lines
    lines = [] # a 2D list storing the vertical start index and end index of each contour
    
    # we are scanning the whole horizontal projection now
    for i, sum in enumerate(hpList):
        # sum being 0 means blank space
        if(sum==0):
            if(setSpaceTop):
                spaceTop = indexCount
                setSpaceTop = False # spaceTop will be set once for each start of a space between lines
            indexCount += 1
            spaceBottom = indexCount
            if(i<len(hpList)-1): # this condition is necessary to avoid array index out of bound error
                if(hpList[i+1]==0): # if the next horizontal projectin is 0, keep on counting, it's still in blank space
                    continue
            # we are using this condition if the previous contour is very thin and possibly not a line
            if(includeNextSpace):
                space_zero.append(spaceBottom-spaceTop)
            else:
                if (len(space_zero)==0):
                    previous = 0
                else:
                    previous = space_zero.pop()
                space_zero.append(previous + spaceBottom-lineTop)
            setSpaceTop = True # next time we encounter 0, it's begining of another space so we set new spaceTop
        
        # sum greater than 0 means contour
        if(sum>0):
            if(setLineTop):
                lineTop = indexCount
                setLineTop = False # lineTop will be set once for each start of a new line/contour
            indexCount += 1
            lineBottom = indexCount
            if(i<len(hpList)-1): # this condition is necessary to avoid array index out of bound error
                if(hpList[i+1]>0): # if the next horizontal projectin is > 0, keep on counting, it's still in contour
                    continue
                    
                # if the line/contour is too thin <10 pixels (arbitrary) in height, we ignore it.
                # Also, we add the space following this and this contour itself to the previous space to form a bigger space: spaceBottom-lineTop.
                if(lineBottom-lineTop<20):
                    includeNextSpace = False
                    setLineTop = True # next time we encounter value > 0, it's begining of another line/contour so we set new lineTop
                    continue
            includeNextSpace = True # the line/contour is accepted, new space following it will be accepted
            
            # append the top and bottom horizontal indices of the line/contour in 'lines'
            lines.append([lineTop, lineBottom])
            setLineTop = True # next time we encounter value > 0, it's begining of another line/contour so we set new lineTop
    
    '''
    # #printing the values we found so far.
    for i, line in enumerate(lines):
        #print
        #print i
        #print line[0]
        #print line[1]
        #print len(hpList[line[0]:line[1]])
        #print hpList[line[0]:line[1]]
    
    for i, line in enumerate(lines):
        cv2.imshow("line "+str(i), img[line[0]:line[1], : ])
    '''
    
    # SECOND we extract the very individual lines from the lines/contours we extracted above.
    fineLines = [] # a 2D list storing the horizontal start index and end index of each individual line
    for i, line in enumerate(lines):
    
        anchor = line[0] # 'anchor' will locate the horizontal indices where horizontal projection is > ANCHOR_POINT for uphill or < ANCHOR_POINT for downhill(ANCHOR_POINT is arbitrary yet suitable!)
        anchorPoints = [] # python list where the indices obtained by 'anchor' will be stored
        upHill = True # it implies that we expect to find the start of an individual line (vertically), climbing up the histogram
        downHill = False # it implies that we expect to find the end of an individual line (vertically), climbing down the histogram
        segment = hpList[line[0]:line[1]] # we put the region of interest of the horizontal projection of each contour here
        
        for j, sum in enumerate(segment):
            if(upHill):
                if(sum<ANCHOR_POINT):
                    anchor += 1
                    continue
                anchorPoints.append(anchor)
                upHill = False
                downHill = True
            if(downHill):
                if(sum>ANCHOR_POINT):
                    anchor += 1
                    continue
                anchorPoints.append(anchor)
                downHill = False
                upHill = True
                
        ##print anchorPoints
        
        # we can ignore the contour here
        if(len(anchorPoints)<2):
            continue
        
        '''
        # the contour turns out to be an individual line
        if(len(anchorPoints)<=3):
            fineLines.append(line)
            continue
        '''
        # len(anchorPoints) > 3 meaning contour composed of multiple lines
        lineTop = line[0]
        for x in range(1, len(anchorPoints)-1, 2):
            # 'lineMid' is the horizontal index where the segmentation will be done
            lineMid = (anchorPoints[x]+anchorPoints[x+1])/2
            lineBottom = lineMid
            # line having height of pixels <20 is considered defects, so we just ignore it
            # this is a weakness of the algorithm to extract lines (anchor value is ANCHOR_POINT, see for different values!)
            if(lineBottom-lineTop < 20):
                continue
            fineLines.append([lineTop, lineBottom])
            lineTop = lineBottom
        if(line[1]-lineTop < 20):
            continue
        fineLines.append([lineTop, line[1]])
        
    # LINE SPACING and LETTER SIZE will be extracted here
    # We will count the total number of pixel rows containing upper and lower zones of the lines and add the space_zero/runs of 0's(excluding first and last of the list ) to it.
    # We will count the total number of pixel rows containing midzones of the lines for letter size.
    # For this, we set an arbitrary (yet suitable!) threshold MIDZONE_THRESHOLD = 15000 in horizontal projection to identify the midzone containing rows.
    # These two total numbers will be divided by number of lines (having at least one row>MIDZONE_THRESHOLD) to find average line spacing and average letter size.
    space_nonzero_row_count = 0
    midzone_row_count = 0
    lines_having_midzone_count = 0
    flag = False
    for i, line in enumerate(fineLines):
        segment =hpList[int(line[0]):int(line[1])]
        for j, sum in enumerate(segment):
            if(sum<MIDZONE_THRESHOLD):
                space_nonzero_row_count += 1
            else:
                midzone_row_count += 1
                flag = True
                
        # This line has contributed at least one count of pixel row of midzone
        if(flag):
            lines_having_midzone_count += 1
            flag = False
    
    # error prevention ^-^
    if(lines_having_midzone_count == 0): lines_having_midzone_count = 1
    
    
    total_space_row_count = space_nonzero_row_count + np.sum(space_zero[1:-1]) #excluding first and last entries: Top and Bottom margins
    # the number of spaces is 1 less than number of lines but total_space_row_count contains the top and bottom spaces of the line
    average_line_spacing = float(total_space_row_count) / lines_having_midzone_count 
    average_letter_size = float(midzone_row_count) / lines_having_midzone_count
    # letter size is actually height of the letter and we are not considering width
    LETTER_SIZE = average_letter_size
    # error prevention ^-^
    if(average_letter_size == 0): average_letter_size = 1
    # We can't just take the average_line_spacing as a feature directly. We must take the average_line_spacing relative to average_letter_size.
    # Let's take the ratio of average_line_spacing to average_letter_size as the LINE SPACING, which is perspective to average_letter_size.
    relative_line_spacing = average_line_spacing / average_letter_size
    LINE_SPACING = relative_line_spacing
    
    # Top marging is also taken relative to average letter size of the handwritting
    relative_top_margin = float(topMarginCount) / average_letter_size
    TOP_MARGIN = relative_top_margin
    
    
    # showing the final extracted lines
    ##for i, line in enumerate(fineLines):
        ##cv2.imshow("line "+str(i), img[line[0]:line[1], : ])
    
    
    ##print space_zero
    ##print lines
    ##print fineLines
    ##print midzone_row_count
    ##print total_space_row_count
    ##print len(hpList)
    ##print average_line_spacing
    ##print lines_having_midzone_count
    ##print i
    #print("Average letter size: "+str(average_letter_size))
    #print("Top margin relative to average letter size: "+str(relative_top_margin))
    #print("Average line spacing relative to average letter size: "+str(relative_line_spacing))

    return fineLines
    
''' function to extract words from the lines using vertical projection '''
def extractWords(image, lines):

    global LETTER_SIZE
    global WORD_SPACING
    
    # apply bilateral filter
    filtered = bilateralFilter(image, 5)
    
    # convert to grayscale and binarize the image by INVERTED binary thresholding
    thresh = threshold(filtered, 180)
    #cv2.imshow('thresh', wthresh)
    
    # Width of the whole document is found once.
    width = thresh.shape[1]
    space_zero = [] # stores the amount of space between words
    words = [] # a 2D list storing the coordinates of each word: y1, y2, x1, x2
    
    # Isolated words or components will be extacted from each line by looking at occurance of 0's in its vertical projection.
    for i, line in enumerate(lines):
        extract = thresh[int(line[0]):int(line[1]), 0:width] # y1:y2, x1:x2
        vp = verticalProjection(extract)
        ##print i
        ##print vp
        
        wordStart = 0
        wordEnd = 0
        spaceStart = 0
        spaceEnd = 0
        indexCount = 0
        setWordStart = True
        setSpaceStart = True
        includeNextSpace = True
        spaces = []
        
        # we are scanning the vertical projection
        for j, sum in enumerate(vp):
            # sum being 0 means blank space
            if(sum==0):
                if(setSpaceStart):
                    spaceStart = indexCount
                    setSpaceStart = False # spaceStart will be set once for each start of a space between lines
                indexCount += 1
                spaceEnd = indexCount
                if(j<len(vp)-1): # this condition is necessary to avoid array index out of bound error
                    if(vp[j+1]==0): # if the next vertical projectin is 0, keep on counting, it's still in blank space
                        continue

                # we ignore spaces which is smaller than half the average letter size
                if((spaceEnd-spaceStart) > int(LETTER_SIZE/2)):
                    spaces.append(spaceEnd-spaceStart)
                    
                setSpaceStart = True # next time we encounter 0, it's begining of another space so we set new spaceStart
            
            # sum greater than 0 means word/component
            if(sum>0):
                if(setWordStart):
                    wordStart = indexCount
                    setWordStart = False # wordStart will be set once for each start of a new word/component
                indexCount += 1
                wordEnd = indexCount
                if(j<len(vp)-1): # this condition is necessary to avoid array index out of bound error
                    if(vp[j+1]>0): # if the next horizontal projectin is > 0, keep on counting, it's still in non-space zone
                        continue
                
                # append the coordinates of each word/component: y1, y2, x1, x2 in 'words'
                # we ignore the ones which has height smaller than half the average letter size
                # this will remove full stops and commas as an individual component
                count = 0
                for k in range(int(line[1])-int(line[0])):
                    row = thresh[int(line[0]+k):int(line[0]+k+1), wordStart:wordEnd] # y1:y2, x1:x2
                    if(np.sum(row)):
                        count += 1
                if(count > int(LETTER_SIZE/2)):
                    words.append([line[0], line[1], wordStart, wordEnd])
                    
                setWordStart = True # next time we encounter value > 0, it's begining of another word/component so we set new wordStart
        
        space_zero.extend(spaces[1:-1])
    
    ##print space_zero
    space_columns = np.sum(space_zero)
    space_count = len(space_zero)
    if(space_count == 0):
        space_count = 1
    average_word_spacing = float(space_columns) / space_count
    relative_word_spacing = average_word_spacing / LETTER_SIZE
    WORD_SPACING = relative_word_spacing
    ##print "Average word spacing: "+str(average_word_spacing)
    #print("Average word spacing relative to average letter size: "+str(relative_word_spacing))
    
    return words
        
''' function to determine the average slant of the handwriting '''
def extractSlant(img, words):
    
    global SLANT_ANGLE
    '''
    0.01 radian = 0.5729578 degree :: I had to put this instead of 0.0 becuase there was a bug yeilding inacurate value which I could not figure out!
    5 degree = 0.0872665 radian :: Hardly noticeable or a very little slant
    15 degree = 0.261799 radian :: Easily noticeable or average slant
    30 degree = 0.523599 radian :: Above average slant
    45 degree = 0.785398 radian :: Extreme slant
    '''
    # We are checking for 9 different values of angle
    theta = [-0.785398, -0.523599, -0.261799, -0.0872665, 0.01, 0.0872665, 0.261799, 0.523599, 0.785398]
    #theta = [-0.785398, -0.523599, -0.436332, -0.349066, -0.261799, -0.174533, -0.0872665, 0, 0.0872665, 0.174533, 0.261799, 0.349066, 0.436332, 0.523599, 0.785398]

    # Corresponding index of the biggest value will be the index of the most likely angle in 'theta'
    s_function = [0.0] * 9
    count_ = [0]*9
    
    # apply bilateral filter
    filtered = bilateralFilter(img, 5)
    
    # convert to grayscale and binarize the image by INVERTED binary thresholding
    # it's better to clear unwanted dark areas at the document left edge and use a high threshold value to preserve more text pixels
    thresh = threshold(filtered, 180)
    #cv2.imshow('thresh', lthresh)
    
    # loop for each value of angle in theta
    for i, angle in enumerate(theta):
        s_temp = 0.0 # overall sum of the functions of all the columns of all the words!
        count = 0 # just counting the number of columns considered to contain a vertical stroke and thus contributing to s_temp
        
        #loop for each word
        for j, word in enumerate(words):
            original = thresh[int(word[0]):int(word[1]), int(word[2]):int(word[3])] # y1:y2, x1:x2

            height = int(word[1])-int(word[0])
            width = int(word[3])-int(word[2])
            
            # the distance in pixel we will shift for affine transformation
            # it's divided by 2 because the uppermost point and the lowermost points are being equally shifted in opposite directions
            shift = (math.tan(angle) * height) / 2
            
            # the amount of extra space we need to add to the original image to preserve information
            # yes, this is adding more number of columns but the effect of this will be negligible
            pad_length = abs(int(shift))
            
            # create a new image that can perfectly hold the transformed and thus widened image
            blank_image = np.zeros((height,width+pad_length*2,3), np.uint8)
            new_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
            new_image[:, pad_length:width+pad_length] = original            
            # points to consider for affine transformation
            (height, width) = new_image.shape[:2]
            x1 = width/2
            y1 = 0
            x2 = width/4
            y2 = height
            x3 = 3*width/4
            y3 = height
    
            pts1 = np.float32([[x1,y1],[x2,y2],[x3,y3]])
            pts2 = np.float32([[x1+shift,y1],[x2-shift,y2],[x3-shift,y3]])
            M = cv2.getAffineTransform(pts1,pts2)
            deslanted = cv2.warpAffine(new_image,M,(width,height))
            
            # find the vertical projection on the transformed image
            vp = verticalProjection(deslanted)
            
            # loop for each value of vertical projection, which is for each column in the word image
            for k, sum in enumerate(vp):
                # the columns is empty
                if(sum == 0):
                    continue
                
                # this is the number of foreground pixels in the column being considered
                num_fgpixel = sum / 255

                # if number of foreground pixels is less than onethird of total pixels, it is not a vertical stroke so we can ignore
                if(num_fgpixel < int(height/3)):
                    continue
                
                # the column itself is extracted, and flattened for easy operation
                column = deslanted[0:height, k:k+1]
                column = column.flatten()
                
                # now we are going to find the distance between topmost pixel and bottom-most pixel
                # l counts the number of empty pixels from top until and upto a foreground pixel is discovered
                for l, pixel in enumerate(column):
                    if(pixel==0):
                        continue
                    break
                # m counts the number of empty pixels from bottom until and upto a foreground pixel is discovered
                for m, pixel in enumerate(column[::-1]):
                    if(pixel==0):
                        continue
                    break
                
                # the distance is found as delta_y, I just followed the naming convention in the research paper I followed
                delta_y = height - (l+m)
            
                # please refer the research paper for more details of this function, anyway it's nothing tricky
                h_sq = (float(num_fgpixel)/delta_y)**2
                
                # I am multiplying by a factor of num_fgpixel/height to the above function to yeild better result
                # this will also somewhat negate the effect of adding more columns and different column counts in the transformed image of the same word
                h_wted = (h_sq * num_fgpixel) / height

                '''
                # just #printing
                if(j==0):
                    #print column
                    #print str(i)+' h_sq='+str(h_sq)+' h_wted='+str(h_wted)+' num_fgpixel='+str(num_fgpixel)+' delta_y='+str(delta_y)
                '''
                
                # add up the values from all the loops of ALL the columns of ALL the words in the image
                s_temp += h_wted
                
                count += 1
            
            
            if(j==0):
                #plt.subplot(),plt.imshow(deslanted),plt.title('Output '+str(i))
                #plt.show()
                cv2.imshow('Output '+str(i)+str(j), deslanted)
                ##print vp
                ##print 'line '+str(i)+' '+str(s_temp)
                ##print
            
                
        s_function[i] = s_temp
        count_[i] = count
    
    # finding the largest value and corresponding index
    max_value = 0.0
    max_index = 4
    for index, value in enumerate(s_function):
        #print(str(index)+" "+str(value)+" "+str(count_[index]))
        if(value > max_value):
            max_value = value
            max_index = index
            
    # We will add another value 9 manually to indicate irregular slant behaviour.
    # This will be seen as value 4 (no slant) but 2 corresponding angles of opposite sign will have very close values.
    if(max_index == 0):
        angle = 45
        result =  " : Extremely right slanted"
    elif(max_index == 1):
        angle = 30
        result = " : Above average right slanted"
    elif(max_index == 2):
        angle = 15
        result = " : Average right slanted"
    elif(max_index == 3):
        angle = 5
        result = " : A little right slanted"
    elif(max_index == 5):
        angle = -5
        result = " : A little left slanted"
    elif(max_index == 6):
        angle = -15
        result = " : Average left slanted"
    elif(max_index == 7):
        angle = -30
        result = " : Above average left slanted"
    elif(max_index == 8):
        angle = -45
        result = " : Extremely left slanted"
    elif(max_index == 4):
        p = s_function[4] / s_function[3]
        q = s_function[4] / s_function[5]
        #print('p='+str(p)+' q='+str(q))
        # the constants here are abritrary but I think suits the best
        if((p <= 1.2 and q <= 1.2) or (p > 1.4 and q > 1.4)):
            angle = 0
            result = " : No slant"
        elif((p <= 1.2 and q-p > 0.4) or (q <= 1.2 and p-q > 0.4)):
            angle = 0
            result = " : No slant"
        else:
            max_index = 9
            angle = 180
            result =  " : Irregular slant behaviour"
        
        if angle == 0:
            print("Slant determined to be straight.")
        else:
            print("Slant determined to be erratic.")
        '''
        type = raw_input("Enter if okay, else enter 'c' to change: ")
        if type=='c':
            if angle == 0:
                angle = 180
                result =  " : Irregular slant behaviour"
            else:
                angle = 0
                result = " : No slant"
        '''
        
    SLANT_ANGLE = angle
    #print("Slant angle(degree): "+str(SLANT_ANGLE)+result)
    return

''' function to extract average pen pressure of the handwriting '''
def barometer(image):

    global PEN_PRESSURE
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:]
    inverted = image
    for x in range(h):
        for y in range(w):
            inverted[x][y] = 255 - image[x][y]
    filtered = bilateralFilter(inverted, 3)
    ret, thresh = cv2.threshold(filtered, 100, 255, cv2.THRESH_TOZERO)
    cv2.imshow('thresh', thresh)
    total_intensity = 0
    pixel_count = 0
    for x in range(h):
        for y in range(w):
            if(thresh[x][y] > 0):
                total_intensity += thresh[x][y]
                pixel_count += 1
    average_intensity = float(total_intensity) / pixel_count
    PEN_PRESSURE = average_intensity
    #print("Average pen pressure: "+str(average_intensity))
    return
    
''' main '''
def main(i):
    image = cv2.imread("C:\\Users\\Lenovo\\Desktop\\capstone\\images\\res\\"+arr[i])
    ##cv2.imshow('image',image)
    
    # Extract pen pressure. It's such a cool function name!
    #barometer(image)

    # apply contour operation to straighten the contours which may be a single line or composed of multiple lines
    # the returned image is straightened version of the original image without filtration and binarization
    straightened = straighten(image)
    ##cv2.imshow('straightened',straightened)# read image from disk


    # apply contour operation to straighten the contours which may be a single line or composed of multiple lines
    # the returned image is straightened version of the original image without filtration and binarization
   
    
    # extract lines of handwritten text from the image using the horizontal projection
    # it returns a 2D list of the vertical starting and ending index/pixel row location of each line in the handwriting
    #lineIndices = extractLines(straightened)
    ##print lineIndices
    ##print
    
    # extract words from each line using vertical projection
    # it returns a 4D list of the vertical starting and ending indices and horizontal starting and ending indices (in that order) of each word in the handwriting
    #wordCoordinates = extractWords(straightened, lineIndices)
    
    ##print wordCoordinates
    ##print len(wordCoordinates)
    #for i, item in enumerate(wordCoordinates):
    #    cv2.imshow('item '+str(i), straightened[item[0]:item[1], item[2]:item[3]])
    
    # extract average slant angle of all the words containing a long vertical stroke
    #extractSlant(straightened, wordCoordinates)
    
    cv2.waitKey(0)
    return
    
#data_feature = pd.DataFrame(columns = ['BASELINE_ANGLE', 'TOP_MARGIN', 'LETTER_SIZE', 'LINE_SPACING', 'WORD_SPACING', 'PEN_PRESSURE', 'SLANT_ANGLE',"FILE_NAME"],ignore_index = True)
def start(i):

    global BASELINE_ANGLE
    global TOP_MARGIN
    global LETTER_SIZE
    global LINE_SPACING
    global WORD_SPACING
    global PEN_PRESSURE
    global SLANT_ANGLE
    #global data_feature
    # read image from disk
    image = cv2.imread("C:\\Users\\Lenovo\\Desktop\\capstone\\images\\res\\"+arr[i])
    barometer(image)
    straightened = straighten(image)
    lineIndices = extractLines(straightened)
    wordCoordinates = extractWords(straightened, lineIndices)
    extractSlant(straightened, wordCoordinates)
    BASELINE_ANGLE = round(BASELINE_ANGLE, 2)
    TOP_MARGIN = round(TOP_MARGIN, 2)
    LETTER_SIZE = round(LETTER_SIZE, 2)
    LINE_SPACING = round(LINE_SPACING, 2)
    WORD_SPACING = round(WORD_SPACING, 2)
    PEN_PRESSURE = round(PEN_PRESSURE, 2)
    SLANT_ANGLE = round(SLANT_ANGLE, 2)

    return [BASELINE_ANGLE, TOP_MARGIN, LETTER_SIZE, LINE_SPACING, WORD_SPACING, PEN_PRESSURE, SLANT_ANGLE,arr[i]]
#data_feature.append({'BASELINE_ANGLE':BASELINE_ANGLE, 'TOP_MARGIN':TOP_MARGIN, 'LETTER_SIZE':LETTER_SIZE, 'LINE_SPACING':LINE_SPACING, 'WORD_SPACING':WORD_SPACING, 'PEN_PRESSURE':PEN_PRESSURE, 'SLANT_ANGLE':SLANT_ANGLE,"FILE_NAME":arr[i]},ignore_index = True)
    #data_feature.append({'BASELINE_ANGLE':BASELINE_ANGLE, 'TOP_MARGIN':TOP_MARGIN, 'LETTER_SIZE':LETTER_SIZE, 'LINE_SPACING':LINE_SPACING, 'WORD_SPACING':WORD_SPACING, 'PEN_PRESSURE':PEN_PRESSURE, 'SLANT_ANGLE':SLANT_ANGLE,"FILE_NAME":arr[i]},ignore_index = True)
    #return data_feature


#categorize py
def determine_baseline_angle(raw_baseline_angle):
    comment = ""
    # falling
    if(raw_baseline_angle >= 0.2):
        baseline_angle = 0
        comment = "DESCENDING"
    # rising
    elif(raw_baseline_angle <= -0.3):
        baseline_angle = 1
        comment = "ASCENDING"
    # straight
    else:
        baseline_angle = 2
        comment = "STRAIGHT"
        
    return baseline_angle, comment

def determine_top_margin(raw_top_margin):
    comment = ""
    # medium and bigger
    if(raw_top_margin >= 1.7):
        top_margin = 0
        comment = "MEDIUM OR BIGGER"
    # narrow
    else:
        top_margin = 1
        comment = "NARROW"
        
    return top_margin, comment

def determine_letter_size(raw_letter_size):
    comment = ""
    # big
    if(raw_letter_size >= 18.0):
        letter_size = 0
        comment = "BIG"
    # small
    elif(raw_letter_size < 13.0):
        letter_size = 1
        comment = "SMALL"
    # medium
    else:
        letter_size = 2
        comment = "MEDIUM"
        
    return letter_size, comment

def determine_line_spacing(raw_line_spacing):
    comment = ""
    # big
    if(raw_line_spacing >= 3.5):
        line_spacing = 0
        comment = "BIG"
    # small
    elif(raw_line_spacing < 2.0):
        line_spacing = 1
        comment = "SMALL"
    # medium
    else:
        line_spacing = 2
        comment = "MEDIUM"
        
    return line_spacing, comment

def determine_word_spacing(raw_word_spacing):
    comment = ""
    # big
    if(raw_word_spacing > 2.0):
        word_spacing = 0
        comment = "BIG"
    # small
    elif(raw_word_spacing < 1.2):
        word_spacing = 1
        comment = "SMALL"
    # medium
    else:
        word_spacing = 2
        comment = "MEDIUM"
        
    return word_spacing, comment

def determine_pen_pressure(raw_pen_pressure):
    comment = ""
    # heavy
    if(raw_pen_pressure > 180.0):
        pen_pressure = 0
        comment = "HEAVY"
    # light
    elif(raw_pen_pressure < 151.0):
        pen_pressure = 1
        comment = "LIGHT"
    # medium
    else:
        pen_pressure = 2
        comment = "MEDIUM"
        
    return pen_pressure, comment

def determine_slant_angle(raw_slant_angle):
    comment = ""
    # extremely reclined
    if(raw_slant_angle == -45.0 or raw_slant_angle == -30.0):
        slant_angle = 0
        comment = "EXTREMELY RECLINED"
    # a little reclined or moderately reclined
    elif(raw_slant_angle == -15.0 or raw_slant_angle == -5.0 ):
        slant_angle = 1
        comment = "A LITTLE OR MODERATELY RECLINED"
    # a little inclined
    elif(raw_slant_angle == 5.0 or raw_slant_angle == 15.0 ):
        slant_angle = 2
        comment = "A LITTLE INCLINED"
    # moderately inclined
    elif(raw_slant_angle == 30.0 ):
        slant_angle = 3
        comment = "MODERATELY INCLINED"
    # extremely inclined
    elif(raw_slant_angle == 45.0 ):
        slant_angle = 4
        comment = "EXTREMELY INCLINED"
    # straight
    elif(raw_slant_angle == 0.0 ):
        slant_angle = 5
        comment = "STRAIGHT"
    # irregular
    #elif(raw_slant_angle == 180 ):
    else:
        slant_angle = 6
        comment = "IRREGULAR"

    return slant_angle, comment

# feature_routine.py
def determine_trait_1(baseline_angle, slant_angle):
    # trait_1 = emotional stability | 1 = stable, 0 = not stable
    if (slant_angle == 0 or slant_angle == 4 or slant_angle == 6 or baseline_angle == 0):
        return 0
    else:
        return 1
    
def determine_trait_2(letter_size, pen_pressure):
    # trait_2 = mental energy or will power | 1 = high or average, 0 = low
    if ((pen_pressure == 0 or pen_pressure == 2) or (letter_size == 1 or letter_size == 2)):
        return 1
    else:
        return 0
    
def determine_trait_3(top_margin, letter_size):
    # trait_3 = modesty | 1 = observed, 0 = not observed (not necessarily the opposite)
    if (top_margin == 0 or  letter_size == 1):
        return 1
    else:
        return 0
    
def determine_trait_4(line_spacing, word_spacing):
    # trait_4 = personal harmony and flexibility | 1 = harmonious, 0 = non harmonious
    if (line_spacing == 2 and word_spacing == 2):
        return 1
    else:
        return 0
    
def determine_trait_5(top_margin, slant_angle):
    # trait_5 = lack of discipline | 1 = observed, 0 = not observed (not necessarily the opposite)
    if (top_margin == 1 and slant_angle == 6):
        return 1
    else:
        return 0
    
def determine_trait_6(letter_size, line_spacing):
    # trait_6 = poor concentration power | 1 = observed, 0 = not observed (not necessarily the opposite)
    if (letter_size == 0 and line_spacing == 1):
        return 1
    else:
        return 0
    
def determine_trait_7(letter_size, word_spacing):
    # trait_7 = non communicativeness | 1 = observed, 0 = not observed (not necessarily the opposite)
    if (letter_size == 1 and word_spacing == 0):
        return 1
    else:
        return 0
    
def determine_trait_8(line_spacing, word_spacing):
    # trait_8 = social isolation | 1 = observed, 0 = not observed (not necessarily the opposite)
    if (word_spacing == 0 or line_spacing == 0):
        return 1
    else:
        return 0
    
    
if os.path.isfile("label_list.txt"):
    #print("Info: label_list found.")
    # =================================================================
    with open("label_list.txt", "r") as labels:
        for line in labels:
            content = line.split()

            baseline_angle = float(content[0])
            X_baseline_angle.append(baseline_angle)

            top_margin = float(content[1])
            X_top_margin.append(top_margin)

            letter_size = float(content[2])
            X_letter_size.append(letter_size)

            line_spacing = float(content[3])
            X_line_spacing.append(line_spacing)

            word_spacing = float(content[4])
            X_word_spacing.append(word_spacing)

            pen_pressure = float(content[5])
            X_pen_pressure.append(pen_pressure)

            slant_angle = float(content[6])
            X_slant_angle.append(slant_angle)

            trait_1 = float(content[7])
            y_t1.append(trait_1)

            trait_2 = float(content[8])
            y_t2.append(trait_2)

            trait_3 = float(content[9])
            y_t3.append(trait_3)

            trait_4 = float(content[10])
            y_t4.append(trait_4)

            trait_5 = float(content[11])
            y_t5.append(trait_5)

            trait_6 = float(content[12])
            y_t6.append(trait_6)

            trait_7 = float(content[13])
            y_t7.append(trait_7)

            trait_8 = float(content[14])
            y_t8.append(trait_8)

            page_id = content[15]
            page_ids.append(page_id)
    # ===============================================================

    # emotional stability
    X_t1 = []
    for a, b in zip(X_baseline_angle, X_slant_angle):
        X_t1.append([a, b])

    # mental energy or will power
    X_t2 = []
    for a, b in zip(X_letter_size, X_pen_pressure):
        X_t2.append([a, b])

    # modesty
    X_t3 = []
    for a, b in zip(X_letter_size, X_top_margin):
        X_t3.append([a, b])

    # personal harmony and flexibility
    X_t4 = []
    for a, b in zip(X_line_spacing, X_word_spacing):
        X_t4.append([a, b])

    # lack of discipline
    X_t5 = []
    for a, b in zip(X_slant_angle, X_top_margin):
        X_t5.append([a, b])

    # poor concentration
    X_t6 = []
    for a, b in zip(X_letter_size, X_line_spacing):
        X_t6.append([a, b])

    # non communicativeness
    X_t7 = []
    for a, b in zip(X_letter_size, X_word_spacing):
        X_t7.append([a, b])

    # social isolation
    X_t8 = []
    for a, b in zip(X_line_spacing, X_word_spacing):
        X_t8.append([a, b])

    # #print X_t1
    # #print type(X_t1)
    # #print len(X_t1)

    X_train, X_test, y_train, y_test = train_test_split(X_t1, y_t1, test_size=.30, random_state=8)
    clf1 = SVC(kernel='rbf')
    clf1.fit(X_train, y_train)
    ##print("Classifier 1 accuracy: ", accuracy_score(clf1.predict(X_test), y_test))

    X_train, X_test, y_train, y_test = train_test_split(X_t2, y_t2, test_size=.30, random_state=16)
    clf2 = SVC(kernel='rbf')
    clf2.fit(X_train, y_train)
    ##print("Classifier 2 accuracy: ", accuracy_score(clf2.predict(X_test), y_test))

    X_train, X_test, y_train, y_test = train_test_split(X_t3, y_t3, test_size=.30, random_state=32)
    clf3 = SVC(kernel='rbf')
    clf3.fit(X_train, y_train)
    ##print("Classifier 3 accuracy: ", accuracy_score(clf3.predict(X_test), y_test))

    X_train, X_test, y_train, y_test = train_test_split(X_t4, y_t4, test_size=.30, random_state=64)
    clf4 = SVC(kernel='rbf')
    clf4.fit(X_train, y_train)
    ##print("Classifier 4 accuracy: ", accuracy_score(clf4.predict(X_test), y_test))

    X_train, X_test, y_train, y_test = train_test_split(X_t5, y_t5, test_size=.30, random_state=42)
    clf5 = SVC(kernel='rbf')
    clf5.fit(X_train, y_train)
    ##print("Classifier 5 accuracy: ", accuracy_score(clf5.predict(X_test), y_test))

    X_train, X_test, y_train, y_test = train_test_split(X_t6, y_t6, test_size=.30, random_state=52)
    clf6 = SVC(kernel='rbf')
    clf6.fit(X_train, y_train)
    ##print("Classifier 6 accuracy: ", accuracy_score(clf6.predict(X_test), y_test))

    X_train, X_test, y_train, y_test = train_test_split(X_t7, y_t7, test_size=.30, random_state=21)
    clf7 = SVC(kernel='rbf')
    clf7.fit(X_train, y_train)
    ##print("Classifier 7 accuracy: ", accuracy_score(clf7.predict(X_test), y_test))

    X_train, X_test, y_train, y_test = train_test_split(X_t8, y_t8, test_size=.30, random_state=73)
    clf8 = SVC(kernel='rbf')
    clf8.fit(X_train, y_train)
    ##print("Classifier 8 accuracy: ", accuracy_score(clf8.predict(X_test), y_test))

    # ================================================================================================


def report():
    raw_features = start(0)
    raw_baseline_angle = raw_features[0]
    baseline_angle, comment = determine_baseline_angle(raw_baseline_angle)
    return ("Baseline Angle: " + comment)
    
    raw_top_margin = raw_features[1]
    top_margin, comment = determine_top_margin(raw_top_margin)
    return ("Top Margin: " + comment)
    
    raw_letter_size = raw_features[2]
    letter_size, comment = determine_letter_size(raw_letter_size)
    return("Letter Size: " + comment)
    
    raw_line_spacing = raw_features[3]
    line_spacing, comment = determine_line_spacing(raw_line_spacing)
    return("Line Spacing: " + comment)
    
    raw_word_spacing = raw_features[4]
    word_spacing, comment = determine_word_spacing(raw_word_spacing)
    return("Word Spacing: " + comment)
    
    raw_pen_pressure = raw_features[5]
    pen_pressure, comment = determine_pen_pressure(raw_pen_pressure)
    return("Pen Pressure: " + comment)
    
    raw_slant_angle = raw_features[6]
    slant_angle, comment = determine_slant_angle(raw_slant_angle)
    return("Slant: " + comment)
    
    
    return("-------------------------------------------------------------------------------------------------")
    return("----------------------------TEAM MARS PERSONALITY PREDICTION SYSTEN------------------------------")
    return("\n")
    if int(clf1.predict([[baseline_angle, slant_angle]]))==1:
        return("Emotional Stability: You are a stable person")
    else:
        return("Emotional Stability: You are not a stable person")
    
    if int(clf2.predict([[letter_size, pen_pressure]]))==1:
        return("Mental Energy or Will Power:: You are mentally a strong person and have a great will power")
    else:
        return("Mental Energy or Will Power:: Your mental energy is low")
    
    if int(clf3.predict([[letter_size, top_margin]]))==1:
        return("Modesty:  You are a modest person")
    else:
        return("Modesty: You are not a modest person")
        
    if int(clf4.predict([[line_spacing, word_spacing]]))==1:
        return("Personal Harmony and Flexibility: You love personal harmony also you are a flexible person")
    else:
        return("Personal Harmony and Flexibility: You are not at personal harmony. You have a Stuborn personality")
    
    if int(clf5.predict([[slant_angle, top_margin]]))==1:
        return("Lack of Discipline:   You are having lack of discipline")
    else:
        return("Lack of Discipline:   You are a discipline person")
        
    if int(clf6.predict([[letter_size, line_spacing]]))==1:
        return("Poor Concentration:  You are having Poor Concentration")
    else:
        return("Poor Concentration: You have Good Concentration ")
    
    if int(clf7.predict([[letter_size, word_spacing]]))==1:
        return("Non Communicativeness: Need To improve Your communication")
    else:
        return("Non Communicativeness: You have Good communication ")
        
    if int(clf8.predict([[line_spacing, word_spacing]]))==1:
        return("Social Isolation:  You like Social Isolation")
    else:
        return("Social Isolation: You dont like Social Isolation ")
    return("\n")
    return("-------------------------------------------------------------------------------------------------")


# In[3]:


#get_ipython().system('jupyter nbconvert   --to script  PPS.ipynb')


# In[ ]:




