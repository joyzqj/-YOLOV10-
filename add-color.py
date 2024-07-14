import cv2
import numpy as np

# 读取图像
image = cv2.imread('/root/autodl-tmp/yolov10-train/test6.jpg')

# 转换到HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 增加暖色光效果
# 增加黄色的饱和度
mask = cv2.inRange(hsv_image, np.array([20, 100, 100]), np.array([30, 255, 255]))
hsv_image[..., 1] = cv2.add(hsv_image[..., 1], 25)  # 增加饱和度

# 减少蓝色的饱和度
mask_blue = cv2.inRange(hsv_image, np.array([100, 100, 100]), np.array([140, 255, 255]))
hsv_image[mask_blue] = (hsv_image[mask_blue] - [0, 15, 0]).clip(0, 255)  # 减少饱和度

# 转换回BGR颜色空间
image_warm = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# 显示或保存图像
cv2.imwrite('infer.jpg', image_warm)