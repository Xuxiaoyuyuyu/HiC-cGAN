import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import time
import Generate

#设置允许的文件格式

total_time= 0
sample_cnt = 0


ALLOWED_EXTENSIONS = set(['hkl'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def upload_file(file):
    if not (file and allowed_file(file.filename)):
        return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于.hkl文件"})

    basepath = os.path.dirname(__file__)
    upload_path = os.path.join(basepath, 'static/data', secure_filename('/low.hkl'))
    file.save(upload_path)
    size = os.path.getsize(upload_path)
    if size >= 1024*1024*1024:
        file_size = str(round(size/(1024),2)) + "GB"
    elif size >= 1024*1024:
        file_size = str(round(size/(1024*1024),2)) + "MB"
    elif size >= 1024:
        file_size = str(round(size/(1024*1024*1024),2)) + "KB"
    else:
        file_size = str(size) + "B"
    return file.filename,file_size

app = Flask(__name__)

# 首页函数
@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        name,size = upload_file(request.files['file'])
        return render_template('upload.html', val1=time.time(), file_name=name, file_size=size)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html', val1=time.time())
    else:
        name,size = upload_file(request.files['file'])
        return render_template('upload.html', val1=time.time(), file_name=name, file_size=size)

@app.route('/result',methods=['GET', 'POST'])
def result():
    global total_time ,sample_cnt
    if request.method == 'POST':
        id_toshow = request.form.get('id')
        return render_template('result.html', val1=time.time(),id_toshow=id_toshow, sample_cnt=sample_cnt, total_time=round(float(total_time), 3))
    else:
        id_toshow = request.form.get('id')
        if id_toshow is None:
            id_toshow = 1
        total_time,sample_cnt = Generate.predict('./static/data/low.hkl')
        return render_template('result.html', val1=time.time(), id_toshow=id_toshow, sample_cnt=sample_cnt, total_time=round(float(total_time),3))

@app.route("/download/<string:filename>", methods=['GET'])
def download(filename):
    if request.method == "GET":
        filepath = filename + '.hkl'
        if os.path.isfile(os.path.join('static/data', filepath)):
            return send_from_directory('static/data', filename=filepath, as_attachment=True)
        pass

if __name__ == '__main__':
    app.run(host='10.249.150.6', port=9281, debug=True)
