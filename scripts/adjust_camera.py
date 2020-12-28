# %%
import cv2
from PIL import Image
import numpy as np
import cv2
import numpy as np
import RPi.GPIO as GPIO
import matplotlib.pyplot as plt

def preprocess(image):
    image = cv2.transpose(image)
    image = cv2.flip(image, flipCode=0)
    image = image[800:1200, 250:900]
    norm_img = np.zeros((image.shape[0], image.shape[1]))
    image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
    image = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY)[1]
    image = cv2.medianBlur(image, ksize=7)

    pil_image = Image.fromarray(image)
    return pil_image

camera = cv2.VideoCapture(0)
GPIO.setmode(GPIO.BOARD)
inputPin = 15
outputPin = 23
GPIO.setup(inputPin, GPIO.IN)
GPIO.setup(outputPin, GPIO.OUT)

while True:
    x = GPIO.input(inputPin)
    ret, frame = camera.read()  # stream from camera

    if x == 1:  # Default value on input pin 15
        GPIO.output(outputPin, 0)  # Led signal off

    if ret and x == 0:  # If button pressed
        GPIO.output(outputPin, 1)  # Led signal on
        image = preprocess(frame)
        plt.imshow(np.array(image))
        plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
