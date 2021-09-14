import os

from flask import Flask, request, json, url_for
from tensorflow import keras
from keras.preprocessing import image
from flask import jsonify
import imageio
import numpy as np
import pandas as pd

app = Flask(__name__)
if __name__ == '__main__':
    app.debug = True
    app.run()

def get_result(filename, kind):
    # 모델에 받은 이미지 넣고 돌림
    md = keras.models.load_model("./deep_learning/model/model_{}.h5".format(kind))
    img = image.load_img('C:/Users/asd36/PycharmProjects/fashion_forecast/server/public/uploads/{}'.format(filename), target_size=(400, 400, 3))
    img = image.img_to_array(img)
    img = img / 255
    result = md.predict(img.reshape(1, 400, 400, 3)).tolist()[0]

    # 결과값 출력
    cols = pd.read_csv("./deep_learning/data/dummy/{}.csv".format(kind)).columns.tolist()
    kor_cate = []
    num_cate = []
    kor_len = []
    num_len = []
    kor_top_len = []
    num_top_len = []
    kor_bottom_len = []
    num_bottom_len = []
    answer = []
    for i, val in enumerate(cols):
        if val.split("_")[0] == "cate":
            kor_cate.append(val.split("_")[1])
            num_cate.append(result[i])
            continue
        if val.split("_")[0] == "len":
            kor_len.append(val.split("_")[1])
            num_len.append(result[i])
            continue
        if val.split("_")[0] == "top_len":
            kor_top_len.append(val.split("_")[1])
            num_top_len.append(result[i])
            continue
        if val.split("_")[0] == "bottom_len":
            kor_bottom_len.append(val.split("_")[1])
            num_bottom_len.append(result[i])
            continue
    # 원피스일 때
    if kind == "op":
        cate = kor_cate[num_cate.index(max(num_cate))]
        top_len = kor_top_len[num_top_len.index(max(num_top_len))]
        bottom_len = kor_bottom_len[num_bottom_len.index(max(num_bottom_len))]
        answer.append(cate)
        answer.append(top_len)
        answer.append(bottom_len)

    # 원피스가 아닐 때
    else:
        cate = kor_cate[num_cate.index(max(num_cate))]
        all_len = kor_len[num_len.index(max(num_len))]
        answer.append(cate)
        answer.append(all_len)
    return answer

@app.route('/model')
def model():
    if request.method == "GET":
        filename = request.args['filename']
        kind = request.args['kind']
        clothes_result = get_result(filename, kind)
        return jsonify(clothes_result)

