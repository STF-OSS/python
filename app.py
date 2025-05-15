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

def plot_cluster_analysis(X_scaled, clusters, kmeans, wcss, silhouette_scores, n_clusters,
                          numeric_cols, df, filename, feature_importance,
                          optimal_clusters_elbow, optimal_clusters_silhouette):
    """将聚类分析图表生成逻辑提取到单独的函数"""
    try:
        # 确保使用全局变量
        global UPLOAD_FOLDER

        # 创建子图布局，为结果创建更多视图
        fig, axes = plt.subplots(3, 2, figsize=(18, 18))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        # 1. 聚类散点图
        ax1 = axes[0, 0]
        if X_scaled.shape[1] >= 2:
            # 获取特征重要性最高的两个特征索引
            top_features = list(feature_importance.keys())[:2] if len(feature_importance) >= 2 else numeric_cols[:2]
            top_feature_indices = [numeric_cols.index(f) for f in top_features]

            # 绘制散点图
            scatter = ax1.scatter(X_scaled[:, top_feature_indices[0]],
                                  X_scaled[:, top_feature_indices[1]],
                                  c=clusters, cmap='viridis', alpha=0.7)

            # 绘制聚类中心
            centers = kmeans.cluster_centers_
            ax1.scatter(centers[:, top_feature_indices[0]],
                        centers[:, top_feature_indices[1]],
                        s=200, c='red', marker='X', label='聚类中心')

            ax1.set_title('主要特征二维聚类图')
            ax1.set_xlabel(top_features[0])
            ax1.set_ylabel(top_features[1])
            ax1.legend()

            # 添加图例
            legend1 = ax1.legend(*scatter.legend_elements(), title="聚类组")
            ax1.add_artist(legend1)
        else:
            ax1.text(0.5, 0.5, '特征数量不足，无法生成二维散点图', ha='center', va='center')

        # 2. 肘部法则图，帮助确定最佳聚类数
        ax2 = axes[0, 1]
        x_range = range(2, 2 + len(wcss))
        ax2.plot(x_range, wcss, marker='o', linestyle='-', color='blue', label='WCSS')

        # 标记最优聚类数（肘部法则）
        if optimal_clusters_elbow:
            ax2.axvline(x=optimal_clusters_elbow, color='r', linestyle='--',
                        label=f'最优聚类数(肘部法则): {optimal_clusters_elbow}')

        # 标记当前选择的聚类数
        ax2.axvline(x=n_clusters, color='g', linestyle=':',
                    label=f'当前选择: {n_clusters}')

        ax2.set_title('肘部法则图 - 确定最佳聚类数')
        ax2.set_xlabel('聚类数量')
        ax2.set_ylabel('WCSS (组内平方和)')
        ax2.legend()

        # 3. 轮廓系数图，另一种评估最佳聚类数的方法
        ax3 = axes[1, 0]
        x_range = range(2, 2 + len(silhouette_scores))
        ax3.plot(x_range, silhouette_scores, marker='o', linestyle='-', color='orange')

        # 标记最优聚类数（轮廓系数）
        if optimal_clusters_silhouette:
            ax3.axvline(x=optimal_clusters_silhouette, color='r', linestyle='--',
                        label=f'最优聚类数(轮廓系数): {optimal_clusters_silhouette}')

        # 标记当前选择的聚类数
        ax3.axvline(x=n_clusters, color='g', linestyle=':',
                    label=f'当前选择: {n_clusters}')

        ax3.set_title('轮廓系数图 - 另一种评估最佳聚类数的方法')
        ax3.set_xlabel('聚类数量')
        ax3.set_ylabel('轮廓系数 (越高越好)')
        ax3.legend()

        # 4. 聚类分布图
        ax4 = axes[1, 1]
        cluster_counts = np.bincount(clusters)
        bars = ax4.bar(range(len(cluster_counts)), cluster_counts, color='skyblue')
        ax4.set_title('各聚类数据量分布')
        ax4.set_xlabel('聚类组')
        ax4.set_ylabel('数据点数量')
        ax4.set_xticks(range(len(cluster_counts)))

        # 在柱子上显示具体数值
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height}',
                     ha='center', va='bottom')

        # 5. 特征重要性图
        ax5 = axes[2, 0]

        # 提取特征重要性得分并处理None值
        features = list(feature_importance.keys())
        importance_scores = []
        for f in features:
            score = feature_importance[f]['importance']
            if score is None or isinstance(score, str):
                importance_scores.append(0.0)  # 将None或字符串值替换为0
            else:
                importance_scores.append(float(score))

        # 按重要性排序，并显示前10个特征
        sorted_indices = np.argsort(importance_scores)[::-1]
        top_n = min(10, len(features))

        top_features = [features[i] for i in sorted_indices[:top_n]]
        top_scores = [importance_scores[i] for i in sorted_indices[:top_n]]

        # 创建水平条形图
        bars = ax5.barh(range(len(top_features)), top_scores, color='lightgreen')
        ax5.set_yticks(range(len(top_features)))
        ax5.set_yticklabels(top_features)
        ax5.set_title('特征重要性')
        ax5.set_xlabel('重要性分数 (%)')

        # 在条形上显示数值
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax5.text(width + 1, i, f'{width:.1f}%', va='center')

        # 6. 各聚类特征均值热力图
        ax6 = axes[2, 1]
        if len(numeric_cols) > 1:
            # 计算每个聚类的特征均值
            cluster_means = df.groupby('Cluster')[numeric_cols].mean()

            # 创建热力图
            sns.heatmap(cluster_means, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax6)
            ax6.set_title('聚类特征均值热力图')
            ax6.set_ylabel('聚类组')
        else:
            ax6.text(0.5, 0.5, '特征数量不足，无法生成热力图', ha='center', va='center')

        # 调整布局
        plt.tight_layout()

        # 保存图像到文件
        img_path = os.path.join(UPLOAD_FOLDER, f'cluster_{filename}.png')
        plt.savefig(img_path, format='png', dpi=100)
        plt.close()

        return True
    except Exception as e:
        print(f"Error generating cluster plots: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/visualize', methods=['POST'])
@login_required
def visualize_data(user_id, username, user_type):
    """生成数据可视化图表"""
    try:
        data = request.json
        filename = data.get('filename')
        chart_type = data.get('chart_type', 'bar')  # 默认为柱状图
        column = data.get('column')  # 要可视化的列

        # 调试信息
        print(f"Debug - Visualize - Filename: {filename}")
        print(f"Debug - Visualize - Chart type: {chart_type}")
        print(f"Debug - Visualize - Column: {column}")

        # 验证输入
        if not filename or not column:
            return jsonify({'error': '缺少必要的参数：文件名或列名'}), 400

        # 构建文件路径
        global UPLOAD_FOLDER
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        print(f"Debug - Visualize - Path: {filepath}")

        if not os.path.exists(filepath):
            return jsonify({'error': f'找不到文件: {filename}'}), 404

        # 读取文件
        try:
            if filename.endswith('.csv'):
                with open(filepath, 'rb') as f:
                    result = chardet.detect(f.read())
                encoding = result['encoding'] if result['encoding'] else 'utf-8'
                df = pd.read_csv(filepath, encoding=encoding)
            else:
                df = pd.read_excel(filepath, engine='openpyxl')
        except Exception as e:
            print(f"文件读取错误: {str(e)}")
            return jsonify({'error': f'文件读取错误: {str(e)}'}), 500

        # 检查列是否存在
        if column not in df.columns:
            return jsonify({'error': f'列 {column} 不存在于数据中'}), 400

        # 创建图像
        plt.figure(figsize=(10, 6))

        # 根据图表类型生成不同的可视化
        if chart_type == 'bar':
            # 柱状图
            try:
                value_counts = df[column].value_counts().sort_index()
                plt.bar(value_counts.index.astype(str), value_counts.values)
                plt.title(f'{column} 分布')
                plt.xlabel(column)
                plt.ylabel('频数')
                plt.xticks(rotation=45)
            except Exception as e:
                print(f"Bar chart error: {str(e)}")
                return jsonify({'error': f'生成柱状图出错: {str(e)}'}), 500

        elif chart_type == 'pie':
            # 饼图
            try:
                value_counts = df[column].value_counts()
                plt.pie(value_counts.values, labels=value_counts.index.astype(str), autopct='%1.1f%%')
                plt.title(f'{column} 分布')
            except Exception as e:
                print(f"Pie chart error: {str(e)}")
                return jsonify({'error': f'生成饼图出错: {str(e)}'}), 500

        elif chart_type == 'line':
            # 折线图 - 检查数据是否为数值型
            try:
                # 如果不是数值型，尝试转换
                if not pd.api.types.is_numeric_dtype(df[column]):
                    try:
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    except:
                        return jsonify({'error': f'列 {column} 不是数值型数据，无法创建折线图'}), 400

                # 绘制折线图
                plt.plot(range(len(df)), df[column].fillna(0))
                plt.title(f'{column} 趋势')
                plt.xlabel('索引')
                plt.ylabel(column)
                # 如果数据点过多，只显示部分刻度
                if len(df) > 20:
                    plt.xticks(range(0, len(df), len(df) // 10))
            except Exception as e:
                print(f"Line chart error: {str(e)}")
                return jsonify({'error': f'生成折线图出错: {str(e)}'}), 500

        elif chart_type == 'histogram':
            # 直方图 - 检查数据是否为数值型
            try:
                # 如果不是数值型，尝试转换
                if not pd.api.types.is_numeric_dtype(df[column]):
                    try:
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    except:
                        return jsonify({'error': f'列 {column} 不是数值型数据，无法创建直方图'}), 400

                plt.hist(df[column].dropna(), bins=20, alpha=0.7, color='skyblue')
                plt.title(f'{column} 分布')
                plt.xlabel(column)
                plt.ylabel('频数')
            except Exception as e:
                print(f"Histogram error: {str(e)}")
                return jsonify({'error': f'生成直方图出错: {str(e)}'}), 500

        elif chart_type == 'box':
            # 箱线图 - 检查数据是否为数值型
            try:
                # 如果不是数值型，尝试转换
                if not pd.api.types.is_numeric_dtype(df[column]):
                    try:
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    except:
                        return jsonify({'error': f'列 {column} 不是数值型数据，无法创建箱线图'}), 400

                plt.boxplot(df[column].dropna())
                plt.title(f'{column} 箱线图')
                plt.ylabel(column)
            except Exception as e:
                print(f"Box plot error: {str(e)}")
                return jsonify({'error': f'生成箱线图出错: {str(e)}'}), 500

        elif chart_type == 'scatter' and len(data.get('columns', [])) > 1:
            # 散点图 (需要两个列)
            try:
                columns = data.get('columns', [])
                if len(columns) >= 2:
                    x_col, y_col = columns[0], columns[1]
                    if x_col in df.columns and y_col in df.columns:
                        # 如果不是数值型，尝试转换
                        if not pd.api.types.is_numeric_dtype(df[x_col]):
                            try:
                                df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
                            except:
                                return jsonify({'error': f'列 {x_col} 不是数值型数据，无法创建散点图'}), 400

                        if not pd.api.types.is_numeric_dtype(df[y_col]):
                            try:
                                df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
                            except:
                                return jsonify({'error': f'列 {y_col} 不是数值型数据，无法创建散点图'}), 400

                        plt.scatter(df[x_col], df[y_col], alpha=0.5)
                        plt.title(f'{x_col} vs {y_col}')
                        plt.xlabel(x_col)
                        plt.ylabel(y_col)
                    else:
                        return jsonify({'error': '指定的列不存在'}), 400
                else:
                    return jsonify({'error': '散点图需要指定两个列'}), 400
            except Exception as e:
                print(f"Scatter plot error: {str(e)}")
                return jsonify({'error': f'生成散点图出错: {str(e)}'}), 500

        else:
            return jsonify({'error': f'不支持的图表类型: {chart_type}'}), 400

        # 调整布局
        try:
            plt.tight_layout()
        except Exception as e:
            print(f"Layout adjustment error: {str(e)}")
            # 布局调整失败不应该中断整个流程

        # 保存图像到内存缓冲区
        try:
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100)
            img_buffer.seek(0)
            plt.close()
        except Exception as e:
            print(f"Image saving error: {str(e)}")
            return jsonify({'error': f'图像保存出错: {str(e)}'}), 500

        # 记录可视化操作
        log_user_action(username, user_id, "数据可视化",
                        filename=filename,
                        details=f"图表类型: {chart_type}, 列: {column}")

        # 返回图像
        return send_file(img_buffer, mimetype='image/png')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

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

if __name__ == '__main__':
    try:
        print("正在启动服务器...")
        print("请访问: http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    except Exception as e:
        print(f"服务器启动失败: {str(e)}")
        import traceback
        traceback.print_exc()
