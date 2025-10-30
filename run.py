# run.py
from app import app
from database.models import db

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # 自动创建 users, trades, forecasts 表
    print("数据库表已创建！")
    print("启动服务器：http://127.0.0.1:5000")
    app.run(debug=True)