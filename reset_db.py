from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash
import datetime

# Recreate minimal app configuration to reset the database
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:ljn050825@localhost:3306/data_analysis'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define minimal User model with increased password length
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)  # Increased length
    user_type = db.Column(db.String(20), default='regular')
    email = db.Column(db.String(120))
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    last_login = db.Column(db.DateTime)

# Define minimal History model to avoid foreign key issues
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id')) 
    filename = db.Column(db.String(120))
    action = db.Column(db.String(120))
    action_details = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(255))
    status = db.Column(db.String(20), default='success')
    
    user = db.relationship('User', backref=db.backref('history', lazy=True))

if __name__ == '__main__':
    with app.app_context():
        # Drop all tables
        db.drop_all()
        print("All tables dropped successfully.")
        
        # Recreate all tables
        db.create_all()
        print("All tables recreated successfully.")
        
        # Create admin user with specified credentials
        admin_username = '872056224'
        admin_password = '910256999'
        
        admin_user = User(
            username=admin_username,
            password=generate_password_hash(admin_password),
            email='admin@example.com',
            user_type='super_admin',
            created_at=datetime.datetime.now()
        )
        db.session.add(admin_user)
        
        # Create test admin account
        test_admin = User(
            username='testadmin',
            password=generate_password_hash('testadmin123'),
            email='testadmin@example.com',
            user_type='admin',
            created_at=datetime.datetime.now()
        )
        db.session.add(test_admin)
        
        # Create test regular user account
        test_user = User(
            username='testuser',
            password=generate_password_hash('testuser123'),
            email='testuser@example.com',
            user_type='regular',
            created_at=datetime.datetime.now()
        )
        db.session.add(test_user)
        
        # Commit changes
        db.session.commit()
        
        print(f"\n\n====================超级管理员账户已创建====================")
        print(f"用户名: {admin_username}")
        print(f"密码: {admin_password}")
        print(f"==========================================================\n\n")
        
        print("Database reset completed. Admin account created with specified credentials.") 