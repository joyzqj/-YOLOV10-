# from werkzeug.utils import secure_filename
# from flask import Flask, request, send_from_directory, url_for
# from PIL import Image
# from flask_cors import CORS
# import os

# app = Flask(__name__)
# CORS(app)

# # 定义保存图片的目录，这里使用静态文件夹
# UPLOAD_FOLDER = 'static/uploads/'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         return 'No file part', 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return 'No selected file', 400
    
#     if file:
#         img = Image.open(file)
        
#         width, height = img.size
#         left = top = 0
#         right = width // 2
#         bottom = height // 2
#         img_cropped = img.crop((left, top, right, bottom))
        
#         if not os.path.exists(app.config['UPLOAD_FOLDER']):
#             os.makedirs(app.config['UPLOAD_FOLDER'])
#         filename = secure_filename(file.filename)
#         img_cropped.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
#         # 确保返回的URL是HTTP格式的，可以通过拼接方式
#         return f'https://u386745-9777-1c5ee906.cqa1.seetacloud.com:8443/' + url_for('static', filename=os.path.join('uploads', filename))

# if __name__ == '__main__':
#     app.run(port=6006, debug=True)


