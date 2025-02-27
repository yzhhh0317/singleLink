from typing import Tuple, List, Dict
import numpy as np
from models.satellite import Satellite


class GridRouter:
    """网格路由实现"""

    def __init__(self, num_planes: int = 6, sats_per_plane: int = 11):
        self.num_planes = num_planes
        self.sats_per_plane = sats_per_plane

    def calculate_next_hop(self, current_pos: Tuple[int, int],
                           target_pos: Tuple[int, int],
                           sat: Satellite) -> str:
        """计算下一跳方向"""
        curr_i, curr_j = current_pos
        target_i, target_j = target_pos

        # 计算相对距离
        delta_i = target_i - curr_i  # 轨道面差
        delta_j = target_j - curr_j  # 同一轨道内卫星编号差
        r_dist = abs(delta_j) / (abs(delta_i) if abs(delta_i) > 0 else 1)

        # 根据相对位置选择方向，移除高纬度判断
        if r_dist < 1:  # 垂直距离更大
            return 'north' if delta_i < 0 else 'south'
        else:  # 水平距离更大
            return 'east' if delta_j > 0 else 'west'