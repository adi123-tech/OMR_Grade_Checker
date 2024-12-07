import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import random

# Function to display images in Streamlit
def show_images(images, titles):
    for index, image in enumerate(images):
        st.image(image, caption=titles[index], use_column_width=True)

# ANSWER_KEY mapping, similar to what you've already defined
ANSWER_KEY = {
    0: 1,
    1: 4,
    2: 0,
    3: 3,
    4: 1
}

# Streamlit UI components
st.title("OMR Scanner and Test Grader")
st.write("Upload the image of the OMR sheet to get the grade.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded image to OpenCV format
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    st.image(edged, caption="Edge Detected Image", use_column_width=True)

    # Find contours
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    # Drawing contours for visualization
    allContourImage = image.copy()
    cv2.drawContours(allContourImage, cnts, -1, (0, 0, 255), 3)
    st.image(allContourImage, caption="Contours Detected", use_column_width=True)

    # Find the document contour
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, closed=True)
            approx = cv2.approxPolyDP(c, epsilon=peri * 0.02, closed=True)
            if len(approx) == 4:
                docCnt = approx
                break

    # Getting the bird's eye view (top view)
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    st.image(paper, caption="Warped Paper", use_column_width=True)
    st.image(warped, caption="Warped Gray Image", use_column_width=True)

    # Threshold the document
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    st.image(thresh, caption="Threshold Image", use_column_width=True)

    # Finding contours in the threshold image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    # Filtering contours for questions
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            questionCnts.append(c)

    # Sorting question contours and checking answers
    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    correct = 0
    questionsContourImage = paper.copy()

    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        cnts = contours.sort_contours(questionCnts[i: i + 5])[0]
        cv2.drawContours(questionsContourImage, cnts, -1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
        bubbled = None

        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        color = (0, 0, 255)
        k = ANSWER_KEY[q]

        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1

        cv2.drawContours(paper, [cnts[k]], -1, color, 3)

    st.image(questionsContourImage, caption="Contours with Colored Answers", use_column_width=True)

    # Final score calculation
    score = (correct / 5.0) * 100
    st.write(f"Score: {score:.2f}%")

    # Display final image with score
    cv2.putText(paper, f"Score: {score:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    st.image(paper, caption="Final Result with Score", use_column_width=True)
