import multiprocessing

# 工作进程数
workers = multiprocessing.cpu_count() * 2 + 1

# 工作模式
worker_class = 'sync'

# 绑定地址
bind = '0.0.0.0:8000'

# 超时设置
timeout = 120

# 访问日志和错误日志
accesslog = 'logs/access.log'
errorlog = 'logs/error.log'
loglevel = 'info'

# 进程名称
proc_name = 'data-analysis-system'

# 守护进程模式
daemon = False

# 最大请求数
max_requests = 1000
max_requests_jitter = 50 