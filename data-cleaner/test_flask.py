from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:ljn050825@localhost:3306/data_analysis'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 使用 with app.app_context() 替代 before_first_request
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run()
