# -*- coding: utf-8 -*-
import os
import cv2
import time
import torch
import random
import logging
import numpy as np
from PIL import Image
from flask_cors import CORS
from datetime import datetime
from torchvision.ops import nms
from werkzeug.utils import secure_filename
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from flask import Flask, request, send_from_directory, url_for,jsonify
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 预处理
def preprocess_letterbox(image):
    try:
        letterbox = LetterBox(new_shape=1024, stride=32, auto=True)
        image = letterbox(image=image)
    except Exception as e:
        logging.error(f"Error initializing or applying LetterBox: {e}")
        raise
    try:
        image = (image[..., ::-1] / 255.0).astype(np.float32)
    except Exception as e:
        logging.error(f"Error converting image to RGB and normalizing: {e}")
        raise
    try:
        image = image.transpose(2, 0, 1)[None]
    except Exception as e:
        logging.error(f"Error transposing image: {e}")
        raise
    try:
        image = torch.from_numpy(image)
    except Exception as e:
        logging.error(f"Error converting image to PyTorch tensor: {e}")
        raise
    return image

def preprocess_warpAffine(image, dst_width=640, dst_height=640):
    try:
        if image is None:
            raise ValueError("Input image is None.")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel image.")
        scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
        if scale < 0:
            raise ValueError("Destination width and height must be greater than 0.")
        
        ox = (dst_width - scale * image.shape[1]) / 2
        oy = (dst_height - scale * image.shape[0]) / 2
        M = np.array([
            [scale, 0, ox],
            [0, scale, oy]
        ], dtype=np.float32)
        img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        logging.info("Affine transformation applied successfully.")
        IM = cv2.invertAffineTransform(M)
        img_pre = img_pre[..., ::-1] / 255.0
        img_pre = img_pre.astype(np.float32)
        img_pre = img_pre.transpose(2, 0, 1)
        img_pre = img_pre[None]
        img_pre = torch.from_numpy(img_pre)
        return img_pre, IM
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
# 后处理
def postprocess(pred, IM=[], conf_thres=0.25):
    try:
        boxes = []
        for item in pred[0]:
            try:
                cx, cy, w, h = item[:4]
                label = item[4:].argmax()
                confidence = item[4 + label]
                if confidence < conf_thres:
                    continue
                left = cx - w * 0.5 / 2
                top = cy - h * 0.5 / 2
                right = cx + w * 0.5 / 2
                bottom = cy + h * 0.5 / 2
                boxes.append([left, top, right, bottom, confidence, label])
            except Exception as e:
                logging.error(f"Error processing item {item}: {e}")
                continue
        try:
            boxes = np.array(boxes)
            if boxes.ndim != 2:
                raise ValueError("boxes error")
            lr = boxes[:, [0, 2]]
            tb = boxes[:, [1, 3]]
            boxes[:, [0, 2]] = IM[0][0] * lr + IM[0][2]
            boxes[:, [1, 3]] = IM[1][1] * tb + IM[1][2]
            return boxes
        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
    except Exception as e:
        logging.error(f"An error occurred in postprocess function: {e}")
        raise
        
        
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file', 400
    try:
        if file:
            img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
            img_pre, IM = preprocess_warpAffine(img)
            model  = AutoBackend(weights="/root/autodl-tmp/yolov10-train/runs/detect/DataSet/weights/best.pt")
            names  = model.names
            result = model(img_pre)['one2one'][0].transpose(-1, -2) # 1,8400,84
            boxes  = postprocess(result, IM)
            
            if boxes is None:
                logging.error("boxes is None, which is not iterable.")
                return jsonify({'error': 'Internal server error'}), 500
            try:
                for obj in boxes:
                    left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
                    label = int(obj[5])
                    confidence = int(obj[4]*100)
                    color=(255,255,255)
                    if names[label]=='ripe_juicy':
                        color=(255,255,255)
                        confidence = int(obj[4]*100)
                    if names[label]=='imperfect' and confidence >= 80:
                        color=(255,255,255)
                        confidence = int(obj[4]*100//1.2) #置信度低于20
                    cv2.rectangle(img, (left, top), (right, bottom) ,color=color,thickness=2, lineType=cv2.LINE_AA)
                    confidence_caption = f"{confidence}"  
                    text_width, text_height = cv2.getTextSize(confidence_caption, 0, 0.5, 2)[0]
                    # 计算文本显示的中心位置
                    text_x = left + (right - left) / 2 - text_width // 2
                    text_x -= text_width//2
                    text_y = top + (bottom - top) / 2 + text_height // 2  # 垂直居中
                    cor = (0, 0, 255)
                    img_cropped=cv2.putText(img, confidence_caption, (int(text_x), int(text_y)), cv2.FONT_HERSHEY_TRIPLEX, 1.8, cor, 2, 16)
                        
            except TypeError as te:
                logging.error(f"TypeError occurred: {te}")
                return jsonify({'error': 'Internal server error'}), 500
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")
                return jsonify({'error': 'Internal server error'}), 500
                
                
            filename1 = secure_filename(file.filename)
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            random_number = random.randint(1000, 9999)
            file_extension = os.path.splitext(filename1)[1]
            new_filename = f"{current_time}{random_number}{file_extension}"
            # 确保新文件名是安全的
            filename = secure_filename(new_filename)
            
            img_cropped = Image.fromarray(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
            current_date = datetime.now().strftime("%Y%m%d")
            folder_path = os.path.join(UPLOAD_FOLDER,  current_date)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            img_cropped.save(os.path.join(folder_path, filename))
            static_url = url_for('static', filename=os.path.join('uploads', current_date, filename))
            return f'{static_url}'
    except Exception as e:
        logging.error(f"An error occurred in upload_image function: {e}")
        raise        
if __name__ == '__main__':
    app.run(port=6006, debug=False)
