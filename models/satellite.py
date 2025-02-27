from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class Satellite:
    """卫星节点类"""
    grid_pos: Tuple[int, int]  # 网格坐标
    is_monitored: bool = False  # 是否需要监控
    links: Dict[str, 'Link'] = field(default_factory=dict)  # 四个方向的链路

    def __post_init__(self):
        self.links = {}

    def add_link(self, direction: str, target_sat, capacity: float = 25.0):
        """添加链路"""
        from models.link import Link  # 避免循环导入
        self.links[direction] = Link(self.grid_pos, target_sat.grid_pos, capacity)