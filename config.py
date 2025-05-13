import os

class Config:
    # 安全设置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    
    # 数据库配置
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'mysql+pymysql://root:123456@localhost:3306/pythonsy'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # 上传文件配置
    MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64MB
    UPLOAD_FOLDER = 'uploads'
    
    # 会话配置
    SESSION_TIMEOUT = 7200  # 2小时
    
    # 跨域配置
    CORS_ORIGINS = ['*']
    CORS_METHODS = ['GET', 'POST', 'OPTIONS']
    CORS_ALLOW_HEADERS = ['Content-Type', 'Authorization'] 