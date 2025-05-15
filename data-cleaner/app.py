from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import chardet
import io

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB

# 创建uploads目录
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        filename = file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # 保存文件
        file.save(save_path)
        
        # 读取文件
        try:
            if filename.lower().endswith('.csv'):
                with open(save_path, 'rb') as f:
                    raw = f.read(10000)
                    result = chardet.detect(raw)
                encoding = result['encoding'] if result['encoding'] else 'utf-8'
                df = pd.read_csv(save_path, encoding=encoding)
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(save_path)
            else:
                return jsonify({'error': '不支持的文件格式，请上传CSV或Excel文件'}), 400
        except Exception as e:
            return jsonify({'error': f'文件读取失败：{str(e)}'}), 400
        
        # 准备预览数据
        preview = {
            'columns': df.columns.tolist(),
            'rows': df.head(1000).values.tolist(),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'filename': filename
        }
        
        return jsonify(preview)
            
    except Exception as e:
        return jsonify({'error': f'上传文件出错: {str(e)}'}), 500

@app.route('/clean', methods=['POST'])
def clean_data():
    try:
        data = request.json
        filename = data.get('filename')
        method = data.get('method', 'auto')  # 'drop', 'fill', 'auto'
        fill_value = data.get('fill_value', 0)
        detect_outliers = data.get('detect_outliers', True)
        outlier_method = data.get('outlier_method', 'iqr')  # 'iqr', 'zscore'
        columns_to_clean = data.get('columns', [])  # 要处理的列
        
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # 读取文件
        if filename.endswith('.csv'):
            with open(filepath, 'rb') as f:
                result = chardet.detect(f.read())
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
            df = pd.read_csv(filepath, encoding=encoding)
        else:
            df = pd.read_excel(filepath)
        
        # 保存原始数据副本
        original_df = df.copy()
        
        # 确定要处理的列
        columns_to_process = columns_to_clean if columns_to_clean else df.columns.tolist()
        
        # 1. 缺失值处理
        missing_stats_before = df[columns_to_process].isnull().sum().to_dict()
        
        if method == 'drop':
            df = df.dropna(subset=columns_to_process)
        elif method == 'fill':
            for col in columns_to_process:
                df[col] = df[col].fillna(fill_value)
        elif method == 'auto':
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in columns_to_process:
                if col in numeric_cols:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    most_frequent = df[col].mode()[0] if not df[col].mode().empty else ""
                    df[col] = df[col].fillna(most_frequent)
        
        missing_stats_after = df[columns_to_process].isnull().sum().to_dict()
        
        # 2. 异常值处理
        outlier_stats = {}
        if detect_outliers:
            numeric_cols = df.select_dtypes(include=['number']).columns
            cols_to_check = [col for col in numeric_cols if col in columns_to_process]
            
            for col in cols_to_check:
                if len(df[col].dropna()) > 10:
                    if outlier_method == 'iqr':
                        # IQR方法
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                        
                        if not outliers.empty:
                            outlier_stats[col] = {
                                'count': len(outliers),
                                'percentage': len(outliers) / len(df) * 100,
                                'lower_bound': float(lower_bound),
                                'upper_bound': float(upper_bound)
                            }
                            
                            # 处理异常值
                            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    
                    elif outlier_method == 'zscore':
                        # Z-score方法
                        zscore_threshold = 3
                        zscores = abs((df[col] - df[col].mean()) / df[col].std())
                        outliers = df[col][zscores > zscore_threshold]
                        
                        if not outliers.empty:
                            outlier_stats[col] = {
                                'count': len(outliers),
                                'percentage': len(outliers) / len(df) * 100,
                                'threshold': zscore_threshold
                            }
                            
                            # 将异常值替换为均值
                            df.loc[zscores > zscore_threshold, col] = df[col].mean()
        
        # 保存清洗后的数据
        cleaned_filename = f'cleaned_{filename}'
        cleaned_path = os.path.join(UPLOAD_FOLDER, cleaned_filename)
        
        if filename.endswith('.csv'):
            df.to_csv(cleaned_path, index=False)
        else:
            df.to_excel(cleaned_path, index=False)
        
        # 生成报告
        report = {
            'missing_values': {
                'before': missing_stats_before,
                'after': missing_stats_after,
                'removed_rows': len(original_df) - len(df)
            },
            'outliers': outlier_stats,
            'data_shape': {
                'original_rows': len(original_df),
                'cleaned_rows': len(df),
                'columns': len(df.columns)
            }
        }
        
        return jsonify({
            'message': '数据清洗完成',
            'cleaned_filename': cleaned_filename,
            'original_rows': len(original_df),
            'cleaned_rows': len(df),
            'cleaning_report': report
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export', methods=['POST'])
def export_data():
    try:
        data = request.json
        filename = data.get('filename')
        format_type = data.get('format', 'csv')
        
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
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
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        print("正在启动服务器...")
        print("请访问: http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"服务器启动失败: {str(e)}")