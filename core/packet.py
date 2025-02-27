# satellite_congestion_control/core/packet.py
from dataclasses import dataclass
from typing import Tuple
import time
import numpy as np


@dataclass
class DataPacket:
    """数据包类"""
    id: int  # 唯一标识符
    source: Tuple[int, int]  # 源节点网格坐标
    destination: Tuple[int, int]  # 目标节点网格坐标
    size: int = 1024 * 8  # 数据包大小(bits)
    creation_time: float = None  # 创建时间

    def __post_init__(self):
        if self.creation_time is None:
            self.creation_time = time.time()


class TrafficGenerator:
    """流量生成器"""

    def __init__(self, link_capacity: float = 25.0):
        """
        Args:
            link_capacity: 链路容量(Mbps)
        """
        self.link_capacity = link_capacity * 1024 * 1024  # 转换为bps
        self.packet_size = 1024 * 8  # bits
        self.time_step = 0.01  # 时间步长(秒)

        # 状态对应的流量比例
        self.state_ratios = {
            'normal': 0.3,  # 正常状态：45%容量
            'warning': 0.5,  # 预警状态：65%容量
            'congestion': 0.7,  # 拥塞状态：85%容量
        }

    def calculate_packets_per_step(self, state: str) -> int:
        """计算每个时间步应生成的数据包数量"""
        state = state if state in self.state_ratios else 'normal'
        target_rate = self.link_capacity * self.state_ratios[state]

        # 计算理论包数
        packets_per_step = (target_rate * self.time_step) / self.packet_size

        # 添加随机扰动
        actual_packets = int(packets_per_step * (1 + np.random.uniform(-0.1, 0.1)))

        # 保证至少生成一些包，但不要太多
        return max(5, min(actual_packets, 100))

    def generate_packets(self, source: Tuple[int, int], state: str,
                         num_satellites: int) -> list:
        """生成一组数据包

        Args:
            source: 源节点坐标
            state: 当前状态
            num_satellites: 总卫星数量

        Returns:
            list: 生成的数据包列表
        """
        packets = []
        num_packets = self.calculate_packets_per_step(state)

        for _ in range(num_packets):
            # 随机选择目标卫星
            dest_i = np.random.randint(0, num_satellites // 11)  # 轨道面
            dest_j = np.random.randint(0, 11)  # 轨道内编号
            destination = (dest_i, dest_j)

            # 确保目标不是源节点
            if destination == source:
                continue

            packet = DataPacket(
                id=int(time.time() * 1000),  # 毫秒级时间戳作为ID
                source=source,
                destination=destination
            )
            packets.append(packet)

        return packets

@dataclass
class CongestionWarningPacket:
    """拥塞预警包"""
    link_id: str  # 拥塞链路标识
    queue_occupancy: float  # 队列占用率
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class CongestionReleasePacket:
    """拥塞解除包"""
    link_id: str  # 拥塞链路标识
    state: str = "RELEASED"  # 链路状态
    timestamp: float = None
    release_duration: int = 3  # 连续多少个周期低于拥塞阈值才确认解除

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class QueueStateUpdatePacket:
    """队列状态更新包"""
    link_id: str  # 链路标识
    queue_occupancy: float  # 当前队列占用率
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()