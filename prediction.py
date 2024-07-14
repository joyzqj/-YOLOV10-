import cv2
import torch
import time
import numpy as np
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend

# 预处理
def preprocess_letterbox(image):
    letterbox = LetterBox(new_shape=640, stride=32, auto=True)
    image = letterbox(image=image)
    image = (image[..., ::-1] / 255.0).astype(np.float32) # BGR to RGB, 0 - 255 to 0.0 - 1.0
    image = image.transpose(2, 0, 1)[None]  # BHWC to BCHW (n, 3, h, w)
    image = torch.from_numpy(image)
    return image

def preprocess_warpAffine(image, dst_width=640, dst_height=640):
    scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
    ox = (dst_width  - scale * image.shape[1]) / 2
    oy = (dst_height - scale * image.shape[0]) / 2
    M = np.array([
        [scale, 0, ox],
        [0, scale, oy]
    ], dtype=np.float32)
    
    img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    IM = cv2.invertAffineTransform(M)

    img_pre = (img_pre[...,::-1] / 255.0).astype(np.float32)
    img_pre = img_pre.transpose(2, 0, 1)[None]
    img_pre = torch.from_numpy(img_pre)
    return img_pre, IM
# 后处理
def postprocess(pred, IM=[], conf_thres=0.25):

    boxes = []
    for item in pred[0]:
        cx, cy, w, h = item[:4]
        label = item[4:].argmax()
        confidence = item[4 + label]
        if confidence < conf_thres:
            continue
        left    = cx - w * 0.5/4
        top     = cy - h * 0.5/4
        right   = cx + w * 0.5/4
        bottom  = cy + h * 0.5/4
        boxes.append([left, top, right, bottom, confidence, label])

    boxes = np.array(boxes)
    lr = boxes[:,[0, 2]] 
    tb = boxes[:,[1, 3]] 
    boxes[:,[0,2]] = IM[0][0] * lr + IM[0][2]
    boxes[:,[1,3]] = IM[1][1] * tb + IM[1][2]
    
    return boxes




if __name__ == "__main__":
    img = cv2.imread("/root/autodl-tmp/yolov10-train/test1.jpg")

    # img_pre = preprocess_letterbox(img)
    img_pre, IM = preprocess_warpAffine(img)
    model  = AutoBackend(weights="/root/autodl-tmp/yolov10-train/runs/detect/train_v10/weights/best.pt")
    names  = model.names
    result = model(img_pre)['one2one'][0].transpose(-1, -2) # 1,8400,84
    boxes  = postprocess(result, IM)

    for obj in boxes:
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        label = int(obj[5])
        confidence = int(obj[4]*100)
        color=(0,0,255)
        if names[label]=='ripe_juicy':
            color=(0,0,255)
            confidence = int(obj[4]*100)
        if names[label]=='imperfect' and confidence > 40:
            color=(0,0,255)
            confidence = int(obj[4]*100//4) #置信度低于20
        cv2.rectangle(img, (left, top), (right, bottom) ,color=color,thickness=2, lineType=cv2.LINE_AA)
        confidence_caption = f"{confidence}"  
        text_width, text_height = cv2.getTextSize(confidence_caption, 0, 0.5, 2)[0]
        # 计算文本显示的中心位置
        text_x = left + (right - left) / 2 - text_width // 2
        text_y = top + (bottom - top) / 2 + text_height // 2  # 垂直居中
        
        # 在图像上绘制置信度文本的背景
        cv2.putText(img, confidence_caption, (int(text_x), int(text_y)), 0, 0.8, (255, 255, 255), 1, 16)
        # cv2.putText(img, confidence_caption, (int(text_x), int(text_y)), 0, 0.8, (255, 255, 255), 1, 16)

    cv2.imwrite("infer5.jpg", img)
    print("save done")  

    
    
