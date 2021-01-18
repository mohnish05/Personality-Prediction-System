#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from PPS import barometer, straighten, extractLines, extractWords, extractSlant
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
    barometer_m=barometer(image)
    straightened = straighten(image)
    lineIndices = extractLines(straightened)
    wordCoordinates = extractWords(straightened, lineIndices)
    extractSlant_m=extractSlant(straightened, wordCoordinates)
    BASELINE_ANGLE = round(BASELINE_ANGLE, 2)
    TOP_MARGIN = round(TOP_MARGIN, 2)
    LETTER_SIZE = round(LETTER_SIZE, 2)
    LINE_SPACING = round(LINE_SPACING, 2)
    WORD_SPACING = round(WORD_SPACING, 2)
    PEN_PRESSURE = round(PEN_PRESSURE, 2)
    SLANT_ANGLE = round(SLANT_ANGLE, 2)
i=0
def report():
    i=i+1
    raw_features = start(i)
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
    return("----------------------------RAHUL SINGH PERSONALITY PREDICTION SYSTEN------------------------------")
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

def main():
    
   
    """ PPS """
    

    # Title
    st.title("Personality Prediction System By RAHUL SINGH")
    st.subheader("Unfold Your Personality with our model")
    st.markdown("""
        #### Description
        + Click Your Hand written Page and Submit us(Write on Blank Paper)
        """)
    
    uploaded_file = st.file_uploader("Choose an image...")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Analysing your personality.......")
        
    st.sidebar.subheader("About App")
    st.sidebar.text("Personality Prediction System")
    st.sidebar.info("Your Handwritting can tell Your Personality")

    st.sidebar.subheader("By")
    st.sidebar.text("RAHUL SINGH")
    st.sidebar.text("DATA SCIENTIST TRAINEE")
    #summary_result = report()
    if st.button("Predict Personality"):
        
        if int(clf1.predict([[baseline_angle,slant_angle]]))==1:
            st.success("Emotional Stability: You are a stable person")
        else:
            st.success("Emotional Stability: You are not a stable person")
        
        if int(clf2.predict([[letter_size, pen_pressure]]))==1:
            st.success("Mental Energy or Will Power:: You are mentally a strong person and have a great will power")
        else:
            st.success("Mental Energy or Will Power:: Your mental energy is low")
        
        if int(clf3.predict([[letter_size, top_margin]]))==1:
            st.success("Modesty:  You are a modest person")
        else:
            st.success("Modesty: You are not a modest person")
            
        if int(clf4.predict([[line_spacing, word_spacing]]))==1:
            st.success("Personal Harmony and Flexibility: You love personal harmony also you are a flexible person")
        else:
            st.success("Personal Harmony and Flexibility: You are not at personal harmony. You have a Stuborn personality")
        
        if int(clf5.predict([[slant_angle, top_margin]]))==1:
            st.success("Lack of Discipline:   You are having lack of discipline")
        else:
            st.success("Lack of Discipline:   You are a discipline person")
            
        if int(clf6.predict([[letter_size, line_spacing]]))==1:
            st.success("Poor Concentration:  You are having Poor Concentration")
        else:
            st.success("Poor Concentration: You have Good Concentration ")
        
        if int(clf7.predict([[letter_size, word_spacing]]))==1:
            st.success("Non Communicativeness: Need To improve Your communication")
        else:
            st.success("Non Communicativeness: You have Good communication ")
            
        if int(clf8.predict([[line_spacing, word_spacing]]))==1:
            st.success("Social Isolation:  You like Social Isolation")
        else:
            st.success("Social Isolation: You dont like Social Isolation ")
    
    
        
if __name__ == '__main__':
    main()


# In[2]:


get_ipython().system('jupyter nbconvert   --to script  Final_Rahul.ipynb')


# In[ ]:


get_ipython().system('streamlit run Final_Rahul.py')


# In[ ]:





# In[ ]:




