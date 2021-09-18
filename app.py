import os
from flask import Flask, request, json, url_for
from tensorflow import keras
from keras.preprocessing import image
from flask import jsonify
import imageio
import numpy as np
import pandas as pd
import torch
import numpy as np
from torchvision import models as models
import torch.nn as nn
import cv2
import torchvision.transforms as transforms

app = Flask(__name__)
if __name__ == '__main__':
    app.debug = True
    app.run()

def get_result_top(filename):
    targets = ["니트웨어", "브라탑", "블라우스", "셔츠", "탑", "티셔츠", "후드티", "7부소매", "긴팔", "민소매", "반팔", "없음", "캡"]
    # 테스팅할 때 '카테고리'와 '소매기장'에 각각 소속하는 클래스를 구분해내기 위한 커스텀 함수
    def find_best_indices_for_each_big_class(sorted_indices):
        sorted_indices_arr = sorted_indices.tolist()
        highest_index = sorted_indices_arr[-1]
        leftovers = sorted_indices_arr[:-1]

        best = [highest_index]
        category_class = range(0, 7)
        sleeve_class = range(7, 13)
        for num in reversed(leftovers):
            if highest_index in category_class:  # 가장 높아서 먼저 뽑힌 애가 '카테고리' 관련 클래스였다면
                if num in sleeve_class:
                    best.append(num)
                    break
            elif highest_index in sleeve_class:  # '소매기장' 관련 클래스였다면
                if num in category_class:
                    best.append(num)
                    break
        return sorted(best)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # intialize the model
    model = models.resnet50(progress=True, pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(2048, 13)  # resnet50
    model.to(device)

    # load the model checkpoint
    # gpu에서 학습시킨 모델을 cpu환경에서 로딩하려고 할때: https://github.com/ultralytics/yolov5/issues/1976
    checkpoint = torch.load(
        './deep_learning/model/model_15th.pth',
        map_location=torch.device('cpu'))

    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 단일 이미지를 모델에 넣어보는 작업
    image_array = np.fromfile(
        'C:/Users/asd36/PycharmProjects/fashion_forecast_server/public/uploads/{}'.format(filename), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # convert the image from BGR to RGB color format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # apply image transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    image = transform(image).to(device)

    outputs = model(image[None, :])
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    sorted_indices = np.argsort(outputs[0])
    # best = sorted_indices[-3:]
    best = find_best_indices_for_each_big_class(sorted_indices)  # 최상위 '카테고리'와 '소매기장' 1개씩 list에 담아 리턴
    answer = []
    for i in range(len(best)):
        label_str = targets[best[i]].split(".")[-1]
        answer.append(label_str)
    return answer

def get_result_other(filename, kind):
    # 모델에 받은 이미지 넣고 돌림
    md = keras.models.load_model("./deep_learning/model/model_{}.h5".format(kind))
    img = image.load_img('C:/Users/asd36/PycharmProjects/fashion_forecast_server/public/uploads/{}'.format(filename), target_size=(400, 400, 3))
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
        if val.split("_")[0] == "top":
            kor_top_len.append(val.split("_")[2])
            num_top_len.append(result[i])
            continue
        if val.split("_")[0] == "bottom":
            kor_bottom_len.append(val.split("_")[2])
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
        if kind == "top":
            # 언니 코드로
            clothes_result = get_result_top(filename)
        else:
            # 내 코드로
            clothes_result = get_result_other(filename, kind)
        return jsonify(clothes_result)

