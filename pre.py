import cv2
import torch
import numpy as np
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend

# 定义一个函数，用于对图像进行预处理，使其符合letterbox要求
def preprocess_letterbox(image):
    # 创建一个LetterBox对象，指定新形状为640，步长为32，自动调整大小
    letterbox = LetterBox(new_shape=640, stride=32, auto=True)
    # 使用LetterBox对象对图像进行调整
    image = letterbox(image=image)
    # 将图像的BGR颜色空间转换为RGB，并将像素值从0-255标准化到0.0-1.0
    image = (image[..., ::-1] / 255.0).astype(np.float32)
    # 将图像的维度从BHWC (批次大小, 高度, 宽度, 通道数) 转换为BCHW
    # 这里通过transpose函数将维度重新排列，然后使用None增加一个批次维度
    image = image.transpose(2, 0, 1)[None]
    # 将numpy数组转换为torch张量，以便于后续使用PyTorch框架进行处理
    image = torch.from_numpy(image)
    # 返回预处理后的图像
    return image

# 定义一个函数，用于对图像进行仿射变换预处理，使其尺寸与目标尺寸一致
def preprocess_warpAffine(image, dst_width=640, dst_height=640):
    # 计算缩放比例，取宽度和高度缩放比例中的较小值
    scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
    # 计算仿射变换中平移参数ox和oy，使得图像居中
    ox = (dst_width  - scale * image.shape[1]) / 2
    oy = (dst_height - scale * image.shape[0]) / 2
    # 创建仿射变换矩阵M
    M = np.array([
        [scale, 0, ox],  # 缩放和平移参数
        [0, scale, oy]
    ], dtype=np.float32)
    
    # 使用cv2.warpAffine函数对图像进行仿射变换
    # 参数包括：原图像，仿射变换矩阵，目标尺寸，插值方式，边界模式和边界值
    img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    # 使用cv2.invertAffineTransform函数获取逆仿射变换矩阵
    IM = cv2.invertAffineTransform(M)

    # 将图像从BGR颜色空间转换为RGB，并进行归一化处理
    img_pre = (img_pre[...,::-1] / 255.0).astype(np.float32)
    # 将图像的维度从BHWC (批次大小, 高度, 宽度, 通道数) 转换为BCHW
    img_pre = img_pre.transpose(2, 0, 1)[None]
    # 将numpy数组转换为torch张量，以便于后续使用PyTorch框架进行处理
    img_pre = torch.from_numpy(img_pre)
    # 返回预处理后的图像和逆仿射变换矩阵
    return img_pre, IM

# 定义一个函数，用于对模型推理结果进行后处理
def postprocess(pred, IM=[], conf_thres=0.25):
    # 输入pred是模型推理的结果，包含8400个预测框
    # 预测框的格式为[1,8400,84]，其中84维包含了中心点坐标(cx, cy)、宽高(w, h)以及80个类别的概率
    boxes = []
    for item in pred[0]:  # 遍历每一个预测框
        cx, cy, w, h = item[:4]  # 提取中心点坐标和宽高
        label = item[4:].argmax()  # 找到概率最高的类别索引
        confidence = item[4 + label]  # 获取该类别的置信度
        if confidence < conf_thres:  # 如果置信度低于设定的阈值，则跳过该预测框
            continue
        # 计算预测框的左上角和右下角坐标
        left    = cx - w * 0.5 / 4
        top     = cy - h * 0.5 / 4
        right   = cx + w * 0.5 / 4
        bottom  = cy + h * 0.5 / 4
        # 将计算得到的坐标和置信度、类别索引添加到boxes列表中
        boxes.append([left, top, right, bottom, confidence, label])

    # 将boxes列表转换为numpy数组
    boxes = np.array(boxes)
    # 提取boxes中的左右(left, right)和上下(top, bottom)坐标
    lr = boxes[:, [0, 2]]  # 左右坐标
    tb = boxes[:, [1, 3]]  # 上下坐标
    # 如果提供了逆仿射变换矩阵IM，则使用它来调整boxes中的坐标
    if IM:
        boxes[:, [0, 2]] = IM[0][0] * lr + IM[0][2]  # 调整左右坐标
        boxes[:, [1, 3]] = IM[1][1] * tb + IM[1][2]  # 调整上下坐标

    # 返回调整后的预测框数组
    return boxes

if __name__ == "__main__":
    
    # 使用cv2.imread函数读取图像
    img = cv2.imread("/root/autodl-tmp/yolov10-train/test8.jpg")

    # # 如果需要使用letterbox预处理，取消下面一行的注释，并注释掉preprocess_warpAffine函数的调用
    # img_pre = preprocess_letterbox(img)
    # 使用warpAffine预处理函数对图像进行仿射变换预处理，并获取逆仿射变换矩阵
    img_pre, IM = preprocess_warpAffine(img)

    # 初始化模型，加载权重文件
    model = AutoBackend(weights="/root/autodl-tmp/yolov10-train/runs/detect/train_v10/weights/best.pt")
    # 获取模型能识别的类别名称列表
    names = model.names
    # 使用模型对预处理后的图像进行推理，并获取结果
    result = model(img_pre)['one2one'][0].transpose(-1, -2)  # 结果格式为1,8400,84
    # 对推理结果进行后处理，获取预测框
    boxes = postprocess(result, IM)

    # 遍历所有预测框
    for obj in boxes:
        # 将预测框的坐标转换为整数
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]*100  # 置信度
        label = int(obj[5])  # 类别索引
        color = (0, 0, 255)  # 绘制矩形的颜色为红色
        # 在原图上绘制预测框
        cv2.rectangle(img, (left, top), (right, bottom), color=color, thickness=2, lineType=cv2.LINE_AA)
        # 格式化置信度数值为两位小数
        confidence_caption = f"{confidence:.2f}"
        # 计算置信度文本的宽度和高度，用于绘制背景
        w, h = cv2.getTextSize(confidence_caption, 0, 1, 2)[0]
        # 在图像上绘制置信度文本的背景
        cv2.rectangle(img, (left - 1, top - 13), (left + w - 10, top), color, -1)
        # 在图像上绘制置信度文本
        cv2.putText(img, confidence_caption, (left, top - 1), 0, 0.5, (255, 255, 255), 1, 16)

    # 将绘制了预测框和置信度的图像保存到文件
    cv2.imwrite("infer1.jpg", img)
    # 打印保存完成的消息
    print("save done")

    
    
