from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ 部署成功！数据中心能源调度系统运行中！"

@app.route('/api')
def test():
    return "✅ API 正常运行"