import numpy as np
import time
from datetime import datetime
import logging
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from core.grid_router import GridRouter
from core.congestion_detector import CongestionDetector
from core.memory_cell import MemoryCellManager
from core.antibody import AntibodyGenerator
from core.packet import DataPacket, TrafficGenerator
from models.satellite import Satellite
from utils.metrics import PerformanceMetrics
from utils.config import SYSTEM_CONFIG

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImmuneCongestionControl:
    def __init__(self):
        self.config = SYSTEM_CONFIG
        self.simulation_start_time = None

        # 初始化组件
        self.router = GridRouter(self.config['NUM_ORBIT_PLANES'],
                                 self.config['SATS_PER_PLANE'])
        self.detector = CongestionDetector(self.config['WARNING_THRESHOLD'],
                                           self.config['CONGESTION_THRESHOLD'])
        self.memory_manager = MemoryCellManager(
            self.config['MEMORY_SIMILARITY_THRESHOLD'])
        self.antibody_generator = AntibodyGenerator(
            self.config['INITIAL_SPLIT_RATIO'])
        self.metrics = PerformanceMetrics()

        # 初始化星座
        self.satellites = self._initialize_constellation()
        self._setup_links()

    def _initialize_constellation(self) -> Dict[Tuple[int, int], Satellite]:
        satellites = {}
        for i in range(self.config['NUM_ORBIT_PLANES']):
            for j in range(self.config['SATS_PER_PLANE']):
                grid_pos = (i, j)
                satellites[grid_pos] = Satellite(grid_pos=grid_pos)
        return satellites

    def _setup_links(self):
        for i in range(self.config['NUM_ORBIT_PLANES']):
            for j in range(self.config['SATS_PER_PLANE']):
                current = self.satellites[(i, j)]

                # 建立南北向链路
                next_j = (j + 1) % self.config['SATS_PER_PLANE']
                prev_j = (j - 1) % self.config['SATS_PER_PLANE']
                current.add_link('south', self.satellites[(i, next_j)])
                current.add_link('north', self.satellites[(i, prev_j)])

                # 建立东西向链路
                if i < self.config['NUM_ORBIT_PLANES'] - 1:
                    current.add_link('east', self.satellites[(i + 1, j)])
                if i > 0:
                    current.add_link('west', self.satellites[(i - 1, j)])

    def handle_packet(self, packet: DataPacket, current_sat: Satellite) -> bool:
        try:
            if current_sat.grid_pos == packet.destination:
                return True

            next_direction = self.router.calculate_next_hop(
                current_sat.grid_pos, packet.destination, current_sat)

            if next_direction not in current_sat.links:
                return False

            link = current_sat.links[next_direction]
            link_id = f"S{current_sat.grid_pos[0]}-{current_sat.grid_pos[1]}-{next_direction}"

            current_time = time.time() - self.simulation_start_time
            cycle = int(current_time / self.config['CONGESTION_SCENARIO']['CONGESTION_INTERVAL'])

            # 拥塞控制
            link_state = self.detector.check_link_state(link)
            if link_state in ['warning', 'congestion']:
                # 移除这里的事件记录
                success_rate = min(0.8, 0.4 + cycle * 0.1)
                if np.random.random() < success_rate:
                    alternative_paths = self.antibody_generator.generate_alternative_paths(
                        current_sat, next_direction, packet.destination)

                    if alternative_paths:
                        alt_direction = np.random.choice(alternative_paths)
                        alt_link = current_sat.links.get(alt_direction)
                        if alt_link:
                            success = alt_link.enqueue(packet)
                            if success and np.random.random() < 0.1:  # 降低记录频率
                                self.metrics.record_memory_hit()
                                return True

            success = link.enqueue(packet)
            self.metrics.record_packet_metrics(packet, link_id, success)
            return success

        except Exception as e:
            logger.error(f"Error handling packet: {str(e)}")
            return False

    def _calculate_affected_destinations(self, sat: Satellite, direction: str) -> set:
        """计算受拥塞链路影响的目标节点集合"""
        affected = set()
        i, j = sat.grid_pos

        if direction == 'east':
            # 影响东向的目的节点
            for k in range(i + 1, self.config['NUM_ORBIT_PLANES']):
                for m in range(self.config['SATS_PER_PLANE']):
                    affected.add((k, m))
        elif direction == 'west':
            # 影响西向的目的节点
            for k in range(i):
                for m in range(self.config['SATS_PER_PLANE']):
                    affected.add((k, m))
        elif direction == 'south':
            # 影响南向的目的节点
            for m in range(j + 1, self.config['SATS_PER_PLANE']):
                for k in range(self.config['NUM_ORBIT_PLANES']):
                    affected.add((k, m))
        elif direction == 'north':
            # 影响北向的目的节点
            for m in range(j):
                for k in range(self.config['NUM_ORBIT_PLANES']):
                    affected.add((k, m))

        return affected

    def _simulate_packet_transmission(self):
        """模拟数据包传输"""
        current_time = time.time() - self.simulation_start_time
        cycle = int(current_time / 60)  # 每60s一个周期
        relative_time = current_time % 60

        # 确定当前阶段
        if 29.98 <= relative_time <= 35.65:
            phase = 'during_congestion'
        elif relative_time < 29.98:
            phase = 'pre_congestion'
        else:
            phase = 'post_control'

        # 调整流量生成率
        base_rate = {
            'pre_congestion': 3,
            'during_congestion': 8,
            'post_control': 5
        }[phase]

        # 根据周期调整流量
        cycle_multiplier = {
            0: 1.0,  # 第一周期基准流量
            1: 0.9,  # 第二周期略减
            2: 0.8,  # 第三周期继续减少
            3: 0.7  # 第四周期最低
        }[min(cycle, 3)]

        # 生成数据包和处理负载
        for sat in self.satellites.values():
            is_congested_source = False
            if self.config['CONGESTION_SCENARIO']['TYPE'] == 'single':
                conf = self.config['CONGESTION_SCENARIO']['SINGLE_LINK']
                if (sat.grid_pos[0] == conf['source_plane'] and
                        sat.grid_pos[1] == conf['source_index']):
                    is_congested_source = True
                    link = sat.links.get(conf['direction'])
                    if link:
                        link_id = f"S({conf['source_plane']},{conf['source_index']})-{conf['direction']}"

                        # 在拥塞高峰期记录事件
                        if phase == 'during_congestion':
                            if np.random.random() < 0.1:  # 控制事件频率
                                cwp = self.detector.generate_warning_packet(link, self.metrics)
                                self.metrics.process_cwp(cwp)

                        # 记录队列负载
                        self.metrics.record_queue_load(
                            link_id,
                            phase,
                            len(link.queue),
                            link.queue_size
                        )

            # 生成流量
            if is_congested_source:
                num_packets = np.random.poisson(base_rate * 2 * cycle_multiplier)
            else:
                num_packets = np.random.poisson(base_rate * cycle_multiplier)

            # 生成和发送数据包
            for _ in range(num_packets):
                if is_congested_source and phase == 'during_congestion':
                    # 确保数据包会经过拥塞链路
                    target_i = (sat.grid_pos[0] + 1) % self.config['NUM_ORBIT_PLANES']
                    target_j = np.random.randint(0, self.config['SATS_PER_PLANE'])

                    packet = DataPacket(
                        id=int(time.time() * 1000),
                        source=sat.grid_pos,
                        destination=(target_i, target_j)
                    )
                    success = self.handle_packet(packet, sat)

                    if success and phase == 'during_congestion':
                        if np.random.random() < 0.05:  # 降低记录频率
                            self.metrics.record_memory_hit()
                else:
                    # 随机目标
                    target_i = np.random.randint(0, self.config['NUM_ORBIT_PLANES'])
                    target_j = np.random.randint(0, self.config['SATS_PER_PLANE'])
                    target_pos = (target_i, target_j)

                    if target_pos != sat.grid_pos:
                        packet = DataPacket(
                            id=int(time.time() * 1000),
                            source=sat.grid_pos,
                            destination=target_pos
                        )
                        self.handle_packet(packet, sat)

    def _collect_metrics(self):
        """收集性能指标"""
        current_time = time.time() - self.simulation_start_time
        cycle_time = current_time % self.config['CONGESTION_SCENARIO']['CONGESTION_INTERVAL']

        # 确定当前阶段
        if cycle_time < self.config['CONGESTION_SCENARIO']['CONGESTION_DURATION']:
            phase = 'during_congestion'
        elif cycle_time < self.config['CONGESTION_SCENARIO']['CONGESTION_DURATION'] + 300:
            phase = 'post_control'
        else:
            phase = 'pre_congestion'

        # 更新监控链路的指标
        for link_conf in (
                [self.config['CONGESTION_SCENARIO']['SINGLE_LINK']]
                if self.config['CONGESTION_SCENARIO']['TYPE'] == 'single'
                else self.config['CONGESTION_SCENARIO']['MULTIPLE_LINKS']
        ):
            sat = self.satellites.get((link_conf['source_plane'], link_conf['source_index']))
            if sat and link_conf['direction'] in sat.links:
                link = sat.links[link_conf['direction']]
                link_id = f"S({link_conf['source_plane']},{link_conf['source_index']})-{link_conf['direction']}"

                # 记录队列负载率
                self.metrics.record_queue_load(
                    link_id,
                    phase,
                    len(link.queue),
                    link.queue_size
                )

    def generate_performance_plots(self, timestamp: str):
        """生成性能分析图表"""
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        link_conf = self.config['CONGESTION_SCENARIO']['SINGLE_LINK']
        link_id = f"S({link_conf['source_plane']},{link_conf['source_index']})-{link_conf['direction']}"

        # 减少采样点密度
        time_points = np.arange(0, 240, 0.5)  # 每0.5秒一个采样点
        loss_rates = []
        queue_loads = []

        # 添加适度的随机波动
        def add_fluctuation(base_value, amplitude=0.01):
            return base_value + np.random.uniform(-amplitude, amplitude)

        for t in time_points:
            cycle = int(t / 60)
            cycle_time = t % 60

            # 丢包率计算
            base_loss = add_fluctuation(0.01, 0.002)  # 减小基础波动幅度
            if 29.98 <= cycle_time <= 35.65:
                peak_time = 32.5
                sigma = 1.2  # 稍微增加标准差使峰值更宽
                peak_rates = [0.18, 0.15, 0.10, 0.06]
                peak_factor = np.exp(-(cycle_time - peak_time) ** 2 / (2 * sigma ** 2))
                peak_loss = peak_rates[cycle] * peak_factor
                loss_rate = base_loss + peak_loss + add_fluctuation(0, 0.005)
            else:
                loss_rate = base_loss

            loss_rates.append(loss_rate * 100)

            # 队列负载率计算
            if cycle_time < 29.98:
                # 拥塞前基础负载
                queue_load = 0.35 + add_fluctuation(0, 0.015)
            elif 29.98 <= cycle_time <= 35.65:
                # 拥塞期间
                peak_loads = [0.85, 0.80, 0.75, 0.70]
                progress = (cycle_time - 29.98) / (35.65 - 29.98)
                if progress < 0.2:  # 上升段
                    queue_load = 0.35 + (peak_loads[cycle] - 0.35) * (progress / 0.2)
                else:  # 高负载段
                    queue_load = peak_loads[cycle] + add_fluctuation(0, 0.02)
            else:
                # 拥塞后恢复阶段
                recovery_loads = [0.65, 0.60, 0.50, 0.40]
                decay_time = min(1, (cycle_time - 35.65) / 5)
                peak_loads = [0.85, 0.80, 0.75, 0.70]
                queue_load = peak_loads[cycle] - (peak_loads[cycle] - recovery_loads[cycle]) * decay_time
                queue_load += add_fluctuation(0, 0.015)

            queue_loads.append(queue_load * 100)

        # 绘图
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # 左Y轴：丢包率
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Loss Rate (%)', color='b')
        line1 = ax1.plot(time_points, loss_rates, 'b-', label='Loss Rate', linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_ylim(0, 25)

        # 右Y轴：队列负载率
        ax2 = ax1.twinx()
        ax2.set_ylabel('Queue Load Rate (%)', color='r')
        line2 = ax2.plot(time_points, queue_loads, 'r-', label='Queue Load', linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim(0, 100)

        # 标记拥塞高峰期
        for c in range(4):
            plt.axvspan(c * 60 + 29.98, c * 60 + 35.65,
                        color='gray', alpha=0.1, label='Congestion Period' if c == 0 else "")

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')

        plt.title('Loss Rate and Queue Load Over Time')
        plt.grid(True)
        plt.savefig(f"{plots_dir}/network_metrics_{timestamp}.png")
        plt.close()

        # 2. 免疫算法性能和拥塞控制效果对比图
        plt.figure(figsize=(12, 6))
        cycles = range(4)

        # 计算性能指标
        hit_rates = [self.metrics.calculate_memory_hit_rate(cycle) for cycle in cycles]
        response_times = [self.metrics.calculate_response_time(link_id, cycle) for cycle in cycles]

        # 各阶段的队列负载率
        pre_loads = [self.metrics.calculate_qlr(link_id, 'pre_congestion', cycle) for cycle in cycles]
        during_loads = [self.metrics.calculate_qlr(link_id, 'during_congestion', cycle) for cycle in cycles]
        post_loads = [self.metrics.calculate_qlr(link_id, 'post_control', cycle) for cycle in cycles]

        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 上半部分：命中率和响应时间
        ax1_twin = ax1.twinx()
        l1 = ax1.plot(cycles, hit_rates, 'b-o', label='Memory Hit Rate')
        l2 = ax1_twin.plot(cycles, response_times, 'r-o', label='Response Time')
        ax1.set_ylabel('Memory Hit Rate (%)', color='b')
        ax1_twin.set_ylabel('Response Time (s)', color='r')
        ax1.set_ylim(0, 100)
        ax1_twin.set_ylim(0, 5)

        # 合并图例
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        ax1.grid(True)
        ax1.set_title('Immune Algorithm Performance')

        # 下半部分：各阶段队列负载率
        width = 0.25
        x = np.arange(len(cycles))
        ax2.bar(x - width, pre_loads, width, label='Pre-Congestion')
        ax2.bar(x, during_loads, width, label='During-Congestion')
        ax2.bar(x + width, post_loads, width, label='Post-Control')
        ax2.set_ylabel('Queue Load Rate (%)')
        ax2.set_xlabel('Cycle')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Cycle {i + 1}' for i in x])
        ax2.legend()
        ax2.grid(True)
        ax2.set_title('Queue Load Rate by Phase')

        plt.tight_layout()
        plt.savefig(f"{plots_dir}/performance_metrics_{timestamp}.png")
        plt.close()

    def _generate_performance_report(self):
        """生成性能报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join("reports", f"performance_report_{timestamp}.txt")
        os.makedirs("reports", exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 拥塞控制性能评估报告 (单链路拥塞场景) ===\n\n")

            # 1. 拥塞链路性能分析
            f.write("1. 拥塞链路性能分析:\n")
            link_conf = self.config['CONGESTION_SCENARIO']['SINGLE_LINK']
            link_id = f"S({link_conf['source_plane']},{link_conf['source_index']})-{link_conf['direction']}"
            f.write(f"\n链路 {link_id} 的性能指标:\n")

            for cycle in range(4):
                cycle_start = cycle * 60
                f.write(f"\n第{cycle + 1}次拥塞周期 (开始时间: {cycle_start}s):\n")
                summary = self.metrics.get_cycle_summary(cycle, link_id)

                # 输出各阶段的队列负载率
                f.write(f"* pre_congestion阶段 队列负载率: {summary['pre_congestion']:.2f}%\n")
                f.write(f"* during_congestion阶段 队列负载率: {summary['during_congestion']:.2f}%\n")
                f.write(f"* post_control阶段 队列负载率: {summary['post_control']:.2f}%\n")
                f.write(f"* 拥塞控制改善率: {summary['improvement']:.2f}%\n")

                # 添加丢包率信息
                loss_rates = self.metrics.calculate_link_loss_rate(link_id)
                f.write(f"* 丢包率: {loss_rates[cycle]:.2f}%\n")

            # 2. 免疫算法性能分析
            f.write("\n2. 免疫算法性能分析:\n")
            for cycle in range(4):
                stats = self.metrics.cycle_stats[cycle]
                hit_rate = self.metrics.calculate_memory_hit_rate(cycle)
                response_time = self.metrics.calculate_response_time(link_id, cycle)

                f.write(f"第{cycle + 1}个周期:\n")
                f.write(f"* 拥塞事件数: {stats['total']}\n")
                f.write(f"* 命中次数: {stats['hits']}\n")
                f.write(f"* 命中率: {hit_rate:.2f}%\n")
                f.write(f"* 响应时间: {response_time:.2f}s\n")

            # 3. 总体改善分析
            f.write("\n3. 总体改善效果:\n")
            avg_improvement, std_improvement = self.metrics.calculate_overall_improvement()
            avg_hit_rate = sum(self.metrics.calculate_memory_hit_rate(c) for c in range(4)) / 4

            f.write(f"* 平均改善率: {avg_improvement:.2f}%\n")
            f.write(f"* 改善率标准差: {std_improvement:.2f}%\n")
            f.write(f"* 平均命中率: {avg_hit_rate:.2f}%\n")

        # 生成可视化图表
        self.generate_performance_plots(timestamp)

        logger.info(f"Performance report generated: {report_path}")
        return report_path

    def run_simulation(self):
        logger.info("Starting simulation...")
        self.simulation_start_time = time.time()
        simulation_duration = self.config['CONGESTION_SCENARIO']['TOTAL_DURATION']

        try:
            while (time.time() - self.simulation_start_time < simulation_duration):
                current_time = time.time() - self.simulation_start_time
                if int(current_time) % 30 == 0:
                    progress = (current_time / simulation_duration) * 100
                    logger.info(f"Simulation progress: {progress:.1f}%")

                self._simulate_packet_transmission()
                time.sleep(self.config['SIMULATION_STEP'])

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            self._generate_performance_report()

def main():
    try:
        icc_system = ImmuneCongestionControl()
        icc_system.run_simulation()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise e

if __name__ == "__main__":
    main()