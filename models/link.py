from dataclasses import dataclass
from typing import Optional, Tuple, List
import time
import numpy as np


@dataclass
class Link:
    def __init__(self, source_id: Tuple[int, int], target_id: Tuple[int, int],
                 capacity: float = 25.0, queue_size: int = 100):
        self.source_id = source_id
        self.target_id = target_id
        self.capacity = capacity * 1024 * 1024  # Mbps转换为bps
        self.queue_size = queue_size
        self.queue = []
        self.last_update_time = time.time()
        self.processed_bytes = 0
        self.start_time = time.time()

    @property
    def queue_occupancy(self) -> float:
        """计算队列占用率，考虑时间因素"""
        current_time = time.time()
        cycle_time = (current_time - self.start_time) % 60  # 60s一个周期

        # 基础占用率
        base_occupancy = len(self.queue) / self.queue_size

        # 在拥塞高峰期增加占用率
        if 29.98 <= cycle_time <= 35.65:
            peak_factor = np.exp(-(cycle_time - 32.5) ** 2 / 2)  # 高斯分布模拟峰值
            return min(1.0, base_occupancy + 0.3 * peak_factor)

        return base_occupancy

    def enqueue(self, packet: 'DataPacket') -> bool:
        """入队，考虑周期性拥塞模式"""
        current_time = time.time()
        cycle_time = (current_time - self.start_time) % 60

        # 计算当前周期
        cycle = int((current_time - self.start_time) / 60)

        # 根据周期调整队列容量
        effective_queue_size = self.queue_size * (1.0 - cycle * 0.1)  # 每周期减少10%容量

        if len(self.queue) >= effective_queue_size:
            # 在拥塞高峰期提高丢包概率
            if 29.98 <= cycle_time <= 35.65:
                drop_prob = 0.8  # 80%的丢包概率
            else:
                drop_prob = 0.2  # 20%的基础丢包概率

            if np.random.random() < drop_prob:
                return False

        self.queue.append(packet)
        return True

    def dequeue(self) -> Optional['DataPacket']:
        """出队，考虑链路容量和处理时间"""
        current_time = time.time()
        elapsed_time = current_time - self.last_update_time

        # 计算这段时间内能处理的数据量
        processable_bytes = int(self.capacity * elapsed_time / 8)  # 转换为字节

        if self.queue and self.processed_bytes <= processable_bytes:
            packet = self.queue.pop(0)
            self.processed_bytes += packet.size

            # 如果已处理数据量超过阈值，更新时间和计数
            if self.processed_bytes >= processable_bytes:
                self.last_update_time = current_time
                self.processed_bytes = 0

            return packet

        return None

    def update_queue_history(self):
        """更新队列历史记录"""
        current_time = time.time()
        self.queue_history.append((current_time - self.last_update_time, len(self.queue)))
        self.last_update_time = current_time

    def get_packet_loss_rate(self) -> float:
        """计算丢包率"""
        if self.total_packets == 0:
            return 0.0
        return self.dropped_packets / self.total_packets

    def get_average_queue_length(self) -> float:
        """计算平均队列长度"""
        if not self.queue_history:
            return 0.0
        return sum(self.queue_history) / len(self.queue_history)

    def update_metrics(self, metrics):
        """更新性能指标"""
        from core.congestion_detector import QueueStateUpdatePacket
        # 修改链路ID的格式为统一格式
        link_id = f"S{self.source_id[0]}-{self.source_id[1]}-{self.target_id[0]}-{self.target_id[1]}"
        qsup = QueueStateUpdatePacket(
            link_id=link_id,
            queue_occupancy=self.queue_occupancy
        )
        metrics.process_qsup(qsup)