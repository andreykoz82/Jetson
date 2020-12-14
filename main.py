import os
import cffi
import time
import re
from datetime import datetime
import cv2
import numpy as np
import RPi.GPIO as GPIO
import matplotlib.pyplot as plt
from scripts.helper_functions import get_abs_path_of_library
from scripts.helper_functions import c_wrapper
from scripts.preprocess import pil_to_pix32
from scripts.preprocess import preprocess
import pandas as pd

ffi = cffi.FFI()
ffi.cdef(c_wrapper())

tess_libname = get_abs_path_of_library('tesseract')
lept_libname = get_abs_path_of_library('lept')

tessdata = '/usr/local/share/tessdata/'
os.environ['TESSDATA_PREFIX'] = tessdata

# Load libraries in ABI mode
if os.path.exists(tess_libname):
    tesseract = ffi.dlopen(tess_libname)
else:
    print(f"'{tess_libname}' does not exists!")
tesseract_version = ffi.string(tesseract.TessVersion())
print('Tesseract-ocr version', tesseract_version.decode('utf-8'))

if os.path.exists(lept_libname):
    leptonica = ffi.dlopen(lept_libname)
else:
    print(f"'{lept_libname}' does not exists!")
leptonica_version = ffi.string(leptonica.getLeptonicaVersion())
print(leptonica_version.decode('utf-8'))
api = None

if api:
    tesseract.TessBaseAPIEnd(api)
    tesseract.TessBaseAPIDelete(api)
api = tesseract.TessBaseAPICreate()

lang = "eng"
oem = tesseract.OEM_DEFAULT
tesseract.TessBaseAPIInit2(api, tessdata.encode(), lang.encode(), oem)
tesseract.TessBaseAPISetPageSegMode(api, tesseract.PSM_AUTO)


def extract_text(image):
    result = dict()
    start = time.time()

    tesseract.TessBaseAPISetImage2(api, image)
    tesseract.TessBaseAPIRecognize(api, ffi.NULL)

    results = ffi.string(tesseract.TessBaseAPIGetUTF8Text(api)).decode('utf-8').split()

    for text in results:
        if len(text) == 14:
            result['GTIN'] = text
        elif len(text) == 13:
            result['SN'] = text.upper()
        elif '/' in text:
            try:
                match = re.search(r'\d{2}/\d{4}', text)
                date = datetime.strptime(match.group(), '%m/%Y').date()
                result['EXP'] = date.strftime("%Y-%m-%d")
            except AttributeError:
                result['EXP'] = "Error"
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

if __name__ == '__main__':
    statistics = pd.DataFrame()
    while True:
        x = GPIO.input(inputPin)
        ret, frame = camera.read()  # stream from camera

        if x == 1:  # Default value on input pin 15
            GPIO.output(outputPin, 0)  # Led signal off

        if ret and x == 0:  # If button pressed
            GPIO.output(outputPin, 1)  # Led signal on
            image = preprocess(frame)
            pix = pil_to_pix32(image, leptonica, ffi)
            result = extract_text(pix)
            statistics = pd.concat([statistics, pd.DataFrame([result])])

            plt.imshow(np.array(image))
            try:
                plt.title(f'GTIN: {result["GTIN"]}\nSN: {result["SN"]}\nBATCH: {result["BATCH"]}\nEXP: {result["EXP"]}'
                          f'\nPROCESSING TIME: {result["PROCESSING TIME"]}', fontsize=12)
            except KeyError:
                plt.title('Error')
            plt.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()