# Project Structure
satellite_congestion_control/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── grid_router.py          # 网格路由实现
│   ├── memory_cell.py         # 记忆细胞管理
│   ├── antibody.py            # 抗体生成与优化
│   ├── congestion_detector.py # 拥塞检测
│   └── packet.py              # 数据包定义
├── models/
│   ├── __init__.py
│   ├── satellite.py           # 卫星节点模型
│   └── link.py               # 链路模型
├── utils/
│   ├── __init__.py
│   ├── metrics.py            # 性能指标计算
│   └── config.py            # 配置参数
└── main.py                   # 主程序入口