import streamlit as st
import cv2
import binarize as binarize
import numpy as np
from PIL import Image
from letters import Letter
import joblib


def add_pad(img, pad = 1, color = (0, 0, 0)):
    width, height = img.size
    new_width = width + pad + pad
    new_height = height + pad + pad
    result = Image.new(img.mode, (new_width, new_height), color)
    result.paste(img, (pad, pad))
    return np.asarray(result)

def get_blobs(orig):
    im = orig.copy()
    if(len(orig.shape) == 3):
        im = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
    im = binarize.binarize(im)
    letters = all_letters(im)
    return letters
def all_letters(im):
    max_label, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(im ^ 255, connectivity=4)
    return [Letter(label, labels, stats[label], centroids[label]) \
            for label in range(1, max_label)]

model = joblib.load("minst_model.pkl")
sc = model.scaler

def extract_number_plate(image):
#     display(image)
    each_characters = get_blobs(image)
    vehicle_number = ""
    for idx,each_letter in enumerate(each_characters):
        x1,y1,x2,y2 = each_letter.get_coord()
        height = y2 - y1
        width = x2 - x1
        if height/width > 0.5:
        #     image = cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
            crop = image[y1:y2, x1:x2]
            crop = add_pad(Image.fromarray(crop), pad = 5, color = (255, 255, 255))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        #     ret, crop = cv2.threshold(crop, 128, 255, cv2.THRESH_BINARY)
            crop = cv2.resize(crop,(28,28))
    #         display(Image.fromarray(crop))
            #cv2.imwrite(f"crop_{idx}.png", crop)
            crop = crop.flatten()
            crop = np.array([crop])

            crop = sc.transform(crop)

            each_char = model.predict(crop)[0]
            vehicle_number += each_char
    return vehicle_number


st.title('Vehicle Number Plate Extraction')

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

    text_e = extract_number_plate(image)

    # Placeholder for displaying the extracted number plate
    st.write('Extracted Number Plate: ',text_e)
