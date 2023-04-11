import numpy as np
import sys
import cv2
from pdb import set_trace
from time import sleep, time
from colors import getDominantColor

METHOD = 'FILE'  # CAMERA or FILE
TEST = True      # If False: overwrite captured images.  If True: save all images for analysis
                 # Note: TEST does nothing if METHOD != CAMERA

if(METHOD == 'CAMERA'):
    from picamera2 import Picamera2, Preview


def setup_camera():
    """
    Set up and configure the camera for use.
    
    Returns:
        Picamera2: Configured camera object.
    """

    camera = Picamera2()
    config = camera.create_preview_configuration()
    config['main']['size'] = (1920, 1440)
    camera.configure(config)
    camera.start()
    camera.set_controls({"AfMode": 0, "LensPosition": 0})
    sleep(4)
    return camera


def capture_load_image(camera, filename):
    """
    Capture an image using the camera or load it from a file.
    
    Args:
        camera (Picamera2): Camera object.
        filename (str): Filename to save or load the image.
    
    Returns:
        tuple: A tuple containing the loaded image, grayscale version, and binary version.
    """

    # take a save a pic if that's the method
    if(METHOD == 'CAMERA'):
        camera.capture_file(filename)
    
    img = cv2.imread(filename,cv2.IMREAD_ANYCOLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, baw = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    return (img, gray, baw)


def draw_circles(circles, outputImg):
    """
    Draw circles on the output image based on the input circles.
    
    Args:
        circles (np.ndarray): Array containing circle information (x, y, radius).
        output_img (np.ndarray): Image on which to draw the circles.
    """

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        #cv2.circle(outputImg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(outputImg,(i[0],i[1]),2,(0,0,255),3)

    outputScaled = cv2.resize(outputImg, (640,480))
    cv2.imshow('detected circles',outputScaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_charlie(charlie, outputImg):
    """
    Draw a circle around Charlie, the big red bottle cap.
    
    Args:
        charlie (np.ndarray): Array containing Charlie's circle information (x, y, radius).
        output_img (np.ndarray): Image on which to draw Charlie.
    """
    circles = charlie.reshape(1,1,-1)
    draw_circles(circles, outputImg)


def find_biccies(baw):
    """
    Find the biscuit locations in the binary image.
    
    Args:
        baw (np.ndarray): Binary image.
    
    Returns:
        np.ndarray: Array containing biscuit circle information (x, y, radius), or None if no biscuits found.
    """
    biccies = cv2.HoughCircles(baw,cv2.HOUGH_GRADIENT,1,8,
                               param1=1,param2=6,minRadius=3,maxRadius=7)
    return biccies


def find_charlie(grayImg, img):
    """
    Find Charlie, the big red bottle cap, in the grayscale image.
    
    Args:
        gray_img (np.ndarray): Grayscale image.
        img (np.ndarray): Original image.
    
    Returns:
        np.ndarray: Array containing Charlie's circle information (x, y, radius), or None if Charlie not found.
    """

    circles = cv2.HoughCircles(grayImg,cv2.HOUGH_GRADIENT,1,40,
                               param1=40,param2=28,minRadius=18,maxRadius=32)
    if circles is not None:
        for circle in circles[0, :]:
            x, y, r = circle

            # test the region of interest (roi) to see the dominate color, if red it's charlie
            roi = img[int(y-r/2):int(y+r/2), int(x-r/2):int(x+r/2)]
            color = getDominantColor(roi)
            if color in {'red', 'red2'}:
                    return circle
    return None


def main():
    """
    Main function to execute the program.
    
    Captures and processes images to find biscuits and Charlie.
    Saves the results to a secret file.
    """

    secret = ""

    camera = None
    if(METHOD == 'CAMERA'):
        camera = setup_camera()

    # capture or load in the biscuit image
    if(METHOD == 'CAMERA' and TEST == False):
        filename = "test.jpg"
    else:
        filename = "testbiccies.jpg"
    (img, gray, baw) = capture_load_image(camera,filename)

    # find the biscuit locations
    biccies = find_biccies(baw)
    # add biccies to secret str
    for biscuit in biccies[0]:
        secret += f"{biscuit[0]}{biscuit[1]}"   # x coordinate, y coordinate
    
    input(f"{biccies.shape[1]} buccies found.  Press any key to start when charlie is ready")
    
    startTime = time()
    i = 0
    while True:
        
        # wait 1 sec for next pic
        while time() - (startTime + i) < 1:
            sleep(0.01)
        print(f"i: {i} round time: {time()-startTime}")

        # take next picture
        if(METHOD == 'CAMERA' and TEST == False):
            filename = "test.jpg"
        else:
            filename = f"test{i}.jpg"
        img, gray, baw = capture_load_image(camera, filename)

        # see if we can find charlie, if so add his coords to secret
        charlie = find_charlie(gray, img)
        if charlie is not None:
            secret += f"{charlie[0]}{charlie[1]}"
            print(f"Charlie found at x: {charlie[0]}, y:{charlie[1]}")
        
        # if we can't find charlie see if there's any biccies left
        else:
            biccies = find_biccies(baw)
            if biccies is not None:
                print(f"{biccies.shape[1]} biccies found, charlie out of frame, continuing")
                secret += '00'
        
            # if no biccies left and charlie not found then finish
            else:
                print("No biccies found, charlie not found, ending")
                break
        i += 1

    # add the final time to the secret
    secret += str(time()-startTime)
    with open("secret.txt", 'w') as f:
        f.write(secret)

if __name__ == "__main__":
    main()

