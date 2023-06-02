from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow import keras
from waitress import serve

app = Flask(__name__)
model = keras.models.load_model('Model/plants.h5')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['post'])
def predict():
    image = request.files['upload']
    # print(image)
    image.save(os.path.join("static", image.filename))

    user_img = cv2.imread('static/' + image.filename)
    # print(user_img.shape)
    user_img = cv2.resize(user_img, (256, 256))
    user_img = user_img / 255.0
    user_img = user_img.reshape(1, 256, 256, 3)
    print(user_img.shape)

    x = model.predict(user_img)
    y = np.argmax(x, axis=1)
    print(y)
    if y == 0:
        y = 'Apple___Apple_scab plant'
    elif y == 1:
        y = 'Apple___Black_rot plant'
    elif y == 2:
        y = "Apple___Cedar_apple_rust plant"
    elif y == 3:
        y = 'Apple___healthy plant'
    elif y == 4:
        y = 'Blueberry___healthy'
    elif y == 5:
        y = 'Cherry_(including_sour)___Powdery_mildew'
    elif y == 6:
        y = 'Cherry_(including_sour)___healthy'
    elif y == 7:
        y = 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot'
    elif y == 8:
        y = 'Corn_(maize)___Common_rust_'
    elif y == 9:
        y = 'Corn_(maize)___Northern_Leaf_Blight'
    elif y == 10:
        y = 'Corn_(maize)___healthy'
    elif y == 11:
        y = 'Grape___Black_rot'
    elif y == 12:
        y = 'Grape___Esca_(Black_Measles)'
    elif y == 13:
        y = 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'
    elif y == 14:
        y = 'Grape___healthy'
    elif y == 15:
        y = 'Orange___Haunglongbing_(Citrus_greening)'
    elif y == 16:
        y = 'Peach___Bacterial_spot'
    elif y == 17:
        y = 'Peach___healthy'
    elif y == 18:
        y = 'Pepper,_bell___Bacterial_spot'
    elif y == 19:
        y = 'Pepper,_bell___healthy'
    elif y == 20:
        y = 'Potato___Early_blight'
    elif y == 21:
        y = 'Potato___Late_blight'
    elif y == 22:
        y = 'Potato___healthy'
    elif y == 23:
        y = 'Raspberry___healthy'
    elif y == 24:
        y = 'Soybean___healthy'
    elif y == 25:
        y = 'Squash___Powdery_mildew'
    elif y == 26:
        y = 'Strawberry___Leaf_scorch'
    elif y == 27:
        y = 'Strawberry___healthy'
    elif y == 28:
        y = 'Tomato___Bacterial_spot'
    elif y == 29:
        y = 'Tomato___Early_blight'
    elif y == 30:
        y = 'Tomato___Late_blight'
    elif y == 31:
        y = 'Tomato___Leaf_Mold'
    elif y == 32:
        y = 'Tomato___Septoria_leaf_spot'
    elif y == 33:
        y = 'Tomato___Spider_mites Two-spotted_spider_mite'
    elif y == 34:
        y = 'Tomato___Target_Spot'
    elif y == 35:
        y = 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
    elif y == 36:
        y = 'Tomato___Tomato_mosaic_virus'
    elif y == 37:
        y = 'Tomato___healthy'
    else:
        y = "Image not Classifier"

    print(y[1])
    return render_template("index.html", y=y, img_path='static/' + image.filename)



mode = "prod"
if __name__ == "__main__":
    if mode == 'prod':
        app.run(host='0.0.0.0', debug=True, port=5001)
    else:
        serve(app, host='0.0.0.0', port=5001, threads=2)
