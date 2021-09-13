import os

from flask import Flask, request, json, url_for
from tensorflow import keras
from keras.preprocessing import image
import imageio
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.debug = True
    app.run()

@app.route('/model')
def model():
    if request.method == "GET":
        md = keras.models.load_model("./model_top.h5")
        filename = request.args['filename']
        img = image.load_img('C:/Users/asd36/PycharmProjects/fashion_forecast/server/public/uploads/{}'.format(filename), target_size=(400, 400, 3))
        img = image.img_to_array(img)
        img = img / 255
        result = md.predict(img.reshape(1, 400, 400, 3))
        print(result.tolist())
        return str(result.tolist()[0][0])
    return None;