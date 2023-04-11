import numpy as np
import cv2

COLOR_NAMES = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "red2"]

COLOR_RANGES_HSV = {
    "red": [(0, 50, 10), (10, 255, 255)],
    "orange": [(10, 50, 10), (25, 255, 255)],
    "yellow": [(25, 50, 10), (35, 255, 255)],
    "green": [(35, 50, 10), (80, 255, 255)],
    "cyan": [(80, 50, 10), (100, 255, 255)],
    "blue": [(100, 50, 10), (130, 255, 255)],
    "purple": [(130, 50, 10), (170, 255, 255)],
    "red2": [(170, 50, 10), (180, 255, 255)]
}

def getMask(frame, color):
    """
    Get a color mask for the input image frame.
    
    Args:
        frame (np.ndarray): Image frame.
        color (str): Color name from COLOR_NAMES list.
    
    Returns:
        np.ndarray: Color mask of the input image frame.
    """
    blurredFrame = cv2.GaussianBlur(frame, (3, 3), 0)
    hsvFrame = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)

    colorRange = COLOR_RANGES_HSV[color]
    lower = np.array(colorRange[0])
    upper = np.array(colorRange[1])

    colorMask = cv2.inRange(hsvFrame, lower, upper)
    colorMask = cv2.bitwise_and(blurredFrame, blurredFrame, mask=colorMask)

    return colorMask


def getDominantColor(roi):
    """
    Get the dominant color within the region of interest (ROI).
    
    Args:
        roi (np.ndarray): Region of interest (ROI) in the image.
    
    Returns:
        str: Dominant color in the ROI.
    """
    roi = np.float32(roi)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4

    # if the circle isn't fully in the frame don't worry about it
    try:
        ret, label, center = cv2.kmeans(roi, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    except:
        return None

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(roi.shape)

    pixelsPerColor = []
    for color in COLOR_NAMES:
        mask = getMask(res2, color)
        greyMask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        count = cv2.countNonZero(greyMask)
        pixelsPerColor.append(count)

    return COLOR_NAMES[pixelsPerColor.index(max(pixelsPerColor))]