@app.route('/analyze', methods=['POST'])
def analyze_data():
    # ... 省略其它代码 ...
    if analysis_type == 'kmeans':
        # 只允许数值型列参与聚类
        numeric_cols = selected_data.select_dtypes(include=[np.number]).columns.tolist()
        selected_data = selected_data[numeric_cols]
        n_clusters = data.get('n_clusters', 3)
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(selected_data)
        # KMeans聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        # 生成聚类可视化
        image_path = plot_cluster_analysis(
            X_scaled, clusters, kmeans, wcss, silhouette_scores, 
            n_clusters, numeric_cols, df_for_plot, filename,
            feature_importance, optimal_clusters_elbow, 
            optimal_clusters_silhouette
        )
        # 返回结果
        return jsonify({
            'success': True,
            'message': '聚类分析完成',
            'image_path': f'/cluster-image/{filename}',
            'report': report,
            # ... 其它字段 ...
        }) 