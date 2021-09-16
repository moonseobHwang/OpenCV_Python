
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os 
##################################################################################################################
""" For uploading YouTube videos url """
import pafy
# pip install pafy
# pip install youtube dl

url = 'https://youtu.be/-Tm4H4CrKT0?t=58'
video = pafy.new(url)
best = video.getbest(preftype='mp4')     # 'webm','3gp'
##################################################################################################################
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[1]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    # print(channel_count)
    return masked_image
"""
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
"""
##################################################################################################################
# detection line drawing function.
def drow_the_lines(img, lines):
    img_p = np.copy(img)   
    blank_image = np.zeros((img_p.shape[0], img_p.shape[1], 3), dtype=np.uint8)
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)

        img = cv2.addWeighted(img_p, 0.8, blank_image, 1, 0.0)
    except:
        return img    
    return img
##################################################################################################################
# image1 = cv2.VideoCapture('./videos/roadway_01.mp4')
image1 = cv2.VideoCapture(best.url)
image1.set(cv2.CAP_PROP_FRAME_WIDTH , 1280)
image1.set(cv2.CAP_PROP_FRAME_HEIGHT , 720)


while True:
    _, image_old = image1.read()
    try:
        image = cv2.cvtColor(image_old, cv2.COLOR_BGR2RGB)
    except:
        break
    scaling = 1
    height = image.shape[0]
    width = image.shape[1]
##################################################################################################################
    #the structure of cropping the reference point to Mask
    region_of_interest_vertices = [
        ((width/2)-100, (height*65)/100),
        (150, height),
        (250, height),
        ((width/2), (height*65)/100),
        (width-300,height),
        (width, height),
        ((width/2)+100, (height*65)/100),
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    """Applies the Grayscale transform"""

    canny_image = cv2.Canny(gray_image, 50, 200)
    """Applies the Canny transform"""

    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32),)

    kernel = np.ones((11, 11), np.uint8) 
    cropped_image = cv2.morphologyEx(cropped_image, cv2.MORPH_CLOSE, kernel)
    """ Applies the opening operation """
##################################################################################################################
    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(cropped_image,
                            rho=1,
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=100,
                            maxLineGap=10)
##################################################################################################################
    image_with_lines = drow_the_lines(image, lines)

    cv2.imshow("cropped_image",cropped_image)
    cv2.imshow("DrawLineonMyWay",image_with_lines)
    cv2.imshow("original_image",image_old)
    if cv2.waitKey(1) == ord("q"):
        break
    if _ == False:
        break
    
cv2.destroyAllWindows()
