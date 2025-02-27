from typing import List, Set, Tuple
import numpy as np
from models.satellite import Satellite
import logging

logger = logging.getLogger(__name__)

class AntibodyGenerator:
    """抗体生成器"""
    def __init__(self, initial_split_ratio: float = 0.4,  # 增加初始分流比例
                 split_step: float = 0.1,  # 增加调整步长
                 min_split: float = 0.3,  # 提高最小分流比例
                 max_split: float = 0.8):
        self.initial_split_ratio = initial_split_ratio
        self.split_step = split_step
        self.min_split = min_split
        self.max_split = max_split

    def generate_alternative_paths(self, sat: Satellite,
                                   congested_direction: str,
                                   target_pos: Tuple[int, int]) -> List[str]:
        """生成备选路径"""
        available_directions = set(['north', 'south', 'east', 'west'])
        available_directions.discard(congested_direction)

        # 检查实际可用的链路
        valid_directions = []
        for direction in available_directions:
            if direction in sat.links:
                link = sat.links[direction]
                if link.queue_occupancy < 0.6:  # 只选择负载较低的链路
                    valid_directions.append(direction)

        return valid_directions[:2]  # 最多返回两个备选方向

    def calculate_affinity(self, initial_occupancy: float,
                         final_occupancy: float,
                         max_occupancy: float,
                         w1: float = 0.6,
                         w2: float = 0.4) -> float:
        """计算亲和度"""
        relative_reduction = (initial_occupancy - final_occupancy) / initial_occupancy
        return w1 * relative_reduction + w2 * (1 - max_occupancy)

    def mutate_split_ratio(self, current_ratio: float,
                         occupancy_trend: List[float]) -> float:
        """变异操作 - 调整分流比例"""
        # 分析最近3个周期的队列占用率趋势
        if len(occupancy_trend) >= 3:
            recent_trend = all(occupancy_trend[i] > occupancy_trend[i - 1]
                             for i in range(1, len(occupancy_trend)))

            if recent_trend:  # 占用率持续上升
                new_ratio = current_ratio + self.split_step
            else:  # 占用率稳定或下降
                new_ratio = current_ratio

            # 确保在合理范围内
            return np.clip(new_ratio, self.min_split, self.max_split)

        return current_ratio

    def should_apply_antibody(self, affinity: float,
                            threshold: float = 0.75) -> bool:
        """判断是否应用抗体"""
        return affinity >= threshold