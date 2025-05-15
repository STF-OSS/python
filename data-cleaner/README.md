# 交互式数据分析系统

这是一个基于 Flask 和 Vue.js 的交互式数据分析系统，支持数据上传、清洗、分析和可视化功能。

## 功能特点

- 支持 CSV 和 Excel 文件上传
- 数据预览和基本统计信息展示
- K-Means 聚类分析
- 数据可视化（散点图）
- 支持 CSV 和 Excel 格式导出

## 系统要求

- Python 3.7+
- Node.js (可选，用于开发)
- 现代浏览器（Chrome、Firefox、Edge 等）

## 安装步骤

1. 克隆项目到本地：
```bash
git clone [项目地址]
cd data-analysis-system
```

2. 安装 Python 依赖：
```bash
pip install -r requirements.txt
```

## 运行系统

1. 启动后端服务：
```bash
python app.py
```

2. 在浏览器中打开 `index.html` 文件

## 使用说明

1. 数据上传
   - 点击"数据上传"标签
   - 选择 CSV 或 Excel 文件
   - 系统会自动显示数据预览

2. 数据分析
   - 点击"数据分析"标签
   - 选择分析方法（目前支持 K-Means 聚类）
   - 选择要分析的列
   - 点击"开始分析"按钮

3. 数据导出
   - 点击"数据导出"标签
   - 选择导出格式（CSV 或 Excel）
   - 点击"导出数据"按钮

## 项目结构

```
data-analysis-system/
├── app.py              # Flask 后端应用
├── index.html          # 前端界面
├── requirements.txt    # Python 依赖
└── uploads/            # 上传文件存储目录
```

## 开发团队

- [团队成员1] - 后端开发
- [团队成员2] - 前端开发
- [团队成员3] - 数据分析
- [团队成员4] - 测试
- [团队成员5] - 文档

## 许可证

MIT License 