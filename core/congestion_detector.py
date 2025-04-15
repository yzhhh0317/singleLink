from typing import Dict, List
from models.satellite import Satellite
from models.link import Link
from core.packet import DataPacket
import time


class CongestionDetector:
    """拥塞检测器"""

    def __init__(self, warning_threshold: float = 0.5,
                 congestion_threshold: float = 0.75,
                 release_duration: int = 3):
        self.warning_threshold = warning_threshold
        self.congestion_threshold = congestion_threshold
        self.release_duration = release_duration
        self.state_history = {}  # 记录每个链路的状态历史
        self.detection_start_time = time.time()  # 记录开始时间
        self.cycle_stats = {}  # 每个周期的检测统计 {cycle: {link_id: stats}}

    def get_current_cycle(self) -> int:
        """获取当前周期"""
        current_time = time.time() - self.detection_start_time
        return min(3, int(current_time / 60))  # 限制最大周期为3

    def initialize_cycle_stats(self, link_id: str):
        """初始化周期统计数据"""
        cycle = self.get_current_cycle()
        if cycle not in self.cycle_stats:
            self.cycle_stats[cycle] = {}
        if link_id not in self.cycle_stats[cycle]:
            self.cycle_stats[cycle][link_id] = {
                'detections': 0,  # 检测次数
                'warnings': 0,  # 预警次数
                'congestions': 0,  # 拥塞次数
                'threshold': self.congestion_threshold - cycle * 0.05  # 随周期降低阈值
            }

    def check_link_state(self, link: Link) -> str:
        """检查链路状态，降低检测频率"""
        link_id = f"S{link.source_id}-S{link.target_id}"
        occupancy = link.queue_occupancy
        current_state = 'normal'  # 初始化默认状态
        current_time = time.time()
        cycle = self.get_current_cycle()

        # 检测到拥塞状态变化时记录时间点
        if occupancy >= self.congestion_threshold:
            if link_id in self.state_history and self.state_history[link_id][-1] != 'congestion':
                # 获取周期内相对时间
                relative_time = (current_time - self.detection_start_time) % 60

                # 将此时间点传递给metrics
                if hasattr(self, 'metrics'):
                    self.metrics.record_congestion_detection(cycle, link_id, relative_time)

        # 初始化状态历史和统计
        if link_id not in self.state_history:
            self.state_history[link_id] = []
        self.initialize_cycle_stats(link_id)

        cycle = self.get_current_cycle()
        # 降低检测阈值的幅度
        cycle_threshold = self.congestion_threshold - cycle * 0.03  # 从0.05改为0.03

        # 降低统计频率，每10次检测才记录一次
        if self.cycle_stats[cycle][link_id]['detections'] % 10 == 0:
            if occupancy >= cycle_threshold:
                current_state = 'congestion'
                self.cycle_stats[cycle][link_id]['congestions'] += 1
            elif occupancy >= self.warning_threshold:
                current_state = 'warning'
                self.cycle_stats[cycle][link_id]['warnings'] += 1

            # 更新状态历史
            self.state_history[link_id].append(current_state)
            if len(self.state_history[link_id]) > self.release_duration:
                self.state_history[link_id].pop(0)

        self.cycle_stats[cycle][link_id]['detections'] += 1
        return current_state

    def generate_warning_packet(self, link: Link, metrics) -> 'CongestionWarningPacket':
        """生成拥塞预警包，考虑学习效果"""
        link_id = f"S{link.source_id[0]}-{link.source_id[1]}-{link.target_id[0]}-{link.target_id[1]}"
        cycle = self.get_current_cycle()

        # 获取当前周期的统计数据
        if cycle in self.cycle_stats and link_id in self.cycle_stats[cycle]:
            stats = self.cycle_stats[cycle][link_id]
            # 根据检测经验调整预警敏感度
            adjusted_occupancy = link.queue_occupancy * (1 - cycle * 0.05)
        else:
            adjusted_occupancy = link.queue_occupancy

        cwp = CongestionWarningPacket(link_id, adjusted_occupancy)
        metrics.process_cwp(cwp)
        return cwp

    def should_release_congestion(self, link: Link) -> bool:
        """判断是否应该解除拥塞状态，考虑学习效果"""
        link_id = f"S{link.source_id}-S{link.target_id}"
        cycle = self.get_current_cycle()

        if link_id not in self.state_history:
            return False

        # 检查状态历史
        history = self.state_history[link_id]
        if len(history) < self.release_duration:
            return False

        # 根据周期调整判断标准
        required_normal_states = max(2, self.release_duration - cycle)  # 随周期降低要求
        normal_states = sum(1 for state in history[-required_normal_states:]
                            if state == 'normal')

        return normal_states >= required_normal_states

    def get_detection_stats(self, link_id: str, cycle: int = None) -> dict:
        """获取特定周期的检测统计"""
        if cycle is None:
            cycle = self.get_current_cycle()

        if cycle in self.cycle_stats and link_id in self.cycle_stats[cycle]:
            stats = self.cycle_stats[cycle][link_id]
            return {
                'detections': stats['detections'],
                'warnings': stats['warnings'],
                'congestions': stats['congestions'],
                'threshold': stats['threshold']
            }
        return {
            'detections': 0,
            'warnings': 0,
            'congestions': 0,
            'threshold': self.congestion_threshold
        }


class CongestionWarningPacket:
    """拥塞预警包"""

    def __init__(self, link_id: str, queue_occupancy: float):
        self.link_id = link_id
        self.queue_occupancy = queue_occupancy
        self.timestamp = time.time()


class CongestionReleasePacket:
    """拥塞解除包"""

    def __init__(self, link_id: str, release_duration: int = 3):
        self.link_id = link_id
        self.state = "RELEASED"
        self.timestamp = time.time()
        self.release_duration = release_duration


class QueueStateUpdatePacket:
    """队列状态更新包"""

    def __init__(self, link_id: str, queue_occupancy: float):
        self.link_id = link_id
        self.queue_occupancy = queue_occupancy
        self.timestamp = time.time()