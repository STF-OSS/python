from flask import Flask, request, jsonify, send_file, session, redirect
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import chardet
import matplotlib
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
import uuid
import functools
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import json
import getpass

# 自定义JSON编码器，处理NumPy类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return float('inf') if obj > 0 else float('-inf')
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# 处理可能包含Infinity或NaN的值，确保JSON可序列化
def safe_json_value(val):
    if isinstance(val, (np.integer, np.floating)):
        if np.isnan(val):
            return None
        elif np.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        return float(val)
    elif isinstance(val, np.ndarray):
        return [safe_json_value(x) for x in val]
    return val

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 增加到64MB
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost:3306/pythonsy'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key_here'  # 用于会话加密
app.json_encoder = NumpyEncoder  # 使用自定义JSON编码器
db = SQLAlchemy(app)

# 添加模板目录配置
app.template_folder = 'templates'

# 创建uploads目录，用于存储上传的文件
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
@login_required
def upload_file(user_id, username, user_type):
    try:
        if 'file' not in request.files:
            print("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        filename = file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving file to: {save_path}")
        
        # 确保上传目录存在
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # 增加错误处理
        try:
            file.save(save_path)
            print("File saved successfully")
        except Exception as e:
            print(f"Error saving file: {e}")
            return jsonify({'error': f'文件保存失败: {str(e)}'}), 500
        
        # 检查文件大小
        file_size = os.path.getsize(save_path)
        if file_size > 50 * 1024 * 1024:  # 50MB
            print(f"Large file detected: {file_size / (1024 * 1024):.2f} MB")
        
        # 根据文件扩展名选择正确的读取方法
        try:
            if filename.lower().endswith('.csv'):
                # 对于CSV文件，尝试检测编码
                with open(save_path, 'rb') as f:
                    raw = f.read(10000)  # 只读取前10KB用于检测编码
                    result = chardet.detect(raw)
                encoding = result['encoding'] if result['encoding'] else 'utf-8'
                print(f"Detected encoding: {encoding}")
                try:
                    # 使用chunksize读取大文件
                    df_chunks = pd.read_csv(save_path, encoding=encoding, on_bad_lines='skip', chunksize=1000)
                    df = next(df_chunks)  # 只获取第一个chunk用于预览
                    total_rows = sum(1 for _ in pd.read_csv(save_path, encoding=encoding, on_bad_lines='skip', iterator=True, chunksize=1000))
                except Exception as e:
                    print(f"Error reading CSV: {e}")
                    return jsonify({'error': f'CSV文件读取失败：{e}，请用Excel另存为UTF-8编码后重试。'}), 400
            elif filename.lower().endswith(('.xls', '.xlsx')):
                # 对于Excel文件，使用openpyxl引擎
                try:
                    # 对于大文件，可能需要设置IO相关参数
                    df = pd.read_excel(save_path, engine='openpyxl', nrows=1000)  # 仅读取前1000行用于预览
                    total_rows = pd.read_excel(save_path, engine='openpyxl', usecols=[0], header=None).shape[0]  # 获取总行数
                    print("Successfully read Excel file")
                except Exception as e:
                    print(f"Error reading Excel: {e}")
                    return jsonify({'error': f'Excel文件读取失败：{e}'}), 400
            else:
                return jsonify({'error': '不支持的文件格式，请上传CSV或Excel文件'}), 400
            
            print(f"Data shape: {df.shape}")
            print(f"Total rows in file: {total_rows}")
            
            # 记录文件上传操作
            log_user_action(username, user_id, "上传文件", 
                           filename=filename, 
                           details=f"文件大小: {os.path.getsize(save_path)} 字节, 数据行数: {total_rows}, 列数: {df.shape[1]}")
            
            # 准备数据以安全序列化为JSON
            # 替换NaN为None (在JSON中会转为null)
            df = df.astype(object).where(pd.notnull(df), None)
            
            # 获取列名
            columns = df.columns.tolist()
            
            # 最多返回1000行预览，确保UI能够处理
            max_preview_rows = min(1000, len(df))
            
            # 使用.to_dict('records')获取行字典列表，更好地处理复杂类型
            rows_data = df.head(max_preview_rows).to_dict('records')
            
            # 将行字典列表转换为行值列表，确保所有值可序列化
            rows = []
            for row_dict in rows_data:
                row = []
                for col in columns:
                    val = row_dict.get(col)
                    # 处理非原始类型
                    if not (val is None or isinstance(val, (str, int, float, bool))):
                        val = str(val)
                    row.append(val)
                rows.append(row)
            
            # 返回预览数据，使用jsonify自动处理序列化
            preview = {
                'columns': columns,
                'rows': rows,
                'total_rows': total_rows,
                'total_columns': len(columns),
                'filename': filename,  # 添加文件名字段到响应中
                'preview_rows': max_preview_rows  # 添加预览的行数信息
            }
            
            # 确保正确设置content-type和字符编码
            response = jsonify(preview)
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
            return response
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"File processing error: {e}")
            return jsonify({'error': f'文件处理出错: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': f'上传文件出错: {str(e)}'}), 500
@app.route('/export', methods=['POST'])
@login_required
def export_data(user_id, username, user_type):
    data = request.json
    filename = data.get('filename')
    format_type = data.get('format', 'csv')
    
    global UPLOAD_FOLDER
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    if filename.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    
    # 记录导出操作
    log_user_action(username, user_id, "导出数据", 
                   filename=filename, 
                   details=f"导出格式: {format_type}, 行数: {len(df)}, 列数: {df.shape[1]}")
    
    if format_type == 'csv':
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='exported_data.csv'
        )
    elif format_type == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='exported_data.xlsx'
        )
    
    return jsonify({'error': 'Invalid export format'}), 400




if __name__ == '__main__':
    try:
        print("正在启动服务器...")
        print("请访问: http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    except Exception as e:
        print(f"服务器启动失败: {str(e)}")
        import traceback
        traceback.print_exc()
