# %%
import pytesseract
import time
import cv2
import matplotlib.pyplot as plt
import re
from datetime import datetime
import numpy as np
import RPi.GPIO as GPIO



def preprocess(image):
    image = image[240:650, 400:1150]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    norm_img = np.zeros((image.shape[0], image.shape[1]))
    image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
    image = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY)[1]
    image = cv2.medianBlur(image, ksize=7)
    return image


def extract_text(image):
    result = dict()
    start = time.time()
    results = pytesseract.image_to_string(image).split('\n')
    for text in results:
        if len(text) == 14:
            result['GTIN'] = text
        elif len(text) == 13:
            result['SN'] = text
        elif '/' in text:
            match = re.search(r'\d{2}/\d{4}', text)
            date = datetime.strptime(match.group(), '%m/%Y').date()
            result['EXP'] = date.strftime("%Y-%m-%d")
        else:
            if text.isdigit():
                result['BATCH'] = text
    result['PROCESSING TIME'] = round(time.time() - start, 3) * 1000
    return result


camera = cv2.VideoCapture(0)

GPIO.setmode(GPIO.BOARD)
inputPin = 15
outputPin = 23
GPIO.setup(inputPin, GPIO.IN)
GPIO.setup(outputPin, GPIO.OUT)

while True:
    x = GPIO.input(inputPin)

    ret, frame = camera.read()

    if x == 1:
        GPIO.output(outputPin, 0)

    if ret and x == 0:
        GPIO.output(outputPin, 1)
        image = preprocess(frame)
        result = extract_text(image)
        plt.imshow(image)
        try:
            plt.title(f'GTIN: {result["GTIN"]}\nSN: {result["SN"]}\nBATCH: {result["BATCH"]}\nEXP: {result["EXP"]}'
                      f'\nPROCESSING TIME: {result["PROCESSING TIME"]}', fontsize=12)
        except KeyError:
            plt.title('Error')
        plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
# %%
plt.imshow(preprocess(frame), cmap='Greys_r')
plt.show()
# %%
extract_text(image)
# %%
cv2.imwrite('17.png', frame)