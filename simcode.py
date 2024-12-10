import numpy as np
import matplotlib.pyplot as plt

# Base class for processors
class Processor:
    def __init__(self, cores, clock_speed):
        self.cores = cores
        self.clock_speed = clock_speed
        self.utilization = 0  # Percentage of utilization

    def execute_instruction(self, workload):
        execution_time = workload / (self.cores * self.clock_speed)
        # Simplified utilization calculation
        self.utilization = min(100, (workload / (self.cores * 100)) * 100)
        return execution_time


# CPU class inheriting from Processor
class CPU(Processor):
    pass

# GPU class inheriting from Processor
class GPU(Processor):
    pass

# Memory unit base class
class MemoryUnit:
    def __init__(self, size, access_time, bandwidth):
        self.size = size  # In MB
        self.access_time = access_time  # In ns
        self.bandwidth = bandwidth  # In MB/s

    def read_data(self, size):
        if size > self.size:
            raise ValueError("Data size exceeds memory capacity.")
        return size / self.bandwidth

    def write_data(self, size):
        if size > self.size:
            raise ValueError("Data size exceeds memory capacity.")
        return size / self.bandwidth

# Cache class inheriting from MemoryUnit
class Cache(MemoryUnit):
    def __init__(self, level, size, access_time, bandwidth):
        super().__init__(size, access_time, bandwidth)
        self.level = level

    def check_cache_hit(self, data_address):
        # Simplified cache hit simulation
        return np.random.choice([True, False], p=[0.8, 0.2])  # 80% hit rate

# Main memory class inheriting from MemoryUnit
class MainMemory(MemoryUnit):
    pass


# Storage class
class Storage:
    def __init__(self, storage_type, capacity, read_speed, write_speed):
        self.storage_type = storage_type  # SSD or HDD
        self.capacity = capacity  # In GB
        self.read_speed = read_speed  # In MB/s
        self.write_speed = write_speed  # In MB/s

    def read_data(self, size):
        return size / self.read_speed

    def write_data(self, size):
        return size / self.write_speed


# NIC class for network interface cards
class NIC:
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth  # In Mbps

    def send_data(self, size):
        return (size * 8) / (self.bandwidth * 1e6)  # Convert MB to bits and divide by Mbps

    def receive_data(self, size):
        return (size * 8) / (self.bandwidth * 1e6)


# Simulation engine class
class SimulationEngine:
    def __init__(self):
        self.current_time = 0

    def run_simulation(self, cpu_workload, memory_workload):
        cpu_time = cpu.execute_instruction(cpu_workload)
        memory_time = memory.read_data(memory_workload)
        total_time = max(cpu_time, memory_time)
        return total_time


# Performance metrics calculator
class MetricsCalculator:
    @staticmethod
    def calculate_cpu_utilization(cpu):
        return cpu.utilization

    @staticmethod
    def calculate_memory_usage(memory_used, memory_total):
        return (memory_used / memory_total) * 100


# Visualization function using Matplotlib
def visualize_performance(cpu_utilization, memory_usage):
    labels = ['CPU Utilization', 'Memory Usage']
    values = [cpu_utilization, memory_usage]
    colors = ['blue', 'green']

    plt.bar(labels, values, color=colors)
    plt.title('System Performance Metrics')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)
    plt.show()


# Example usage of the hardware simulator model

# Create hardware components
cpu = CPU(cores=4, clock_speed=3.5)  # GHz
memory = MainMemory(size=16000, access_time=50, bandwidth=25)  # MB and ns/MB/s

# Simulate workloads and calculate performance metrics
simulation_engine = SimulationEngine()
simulation_engine.run_simulation(cpu_workload=5000, memory_workload=8000)

cpu_utilization = MetricsCalculator.calculate_cpu_utilization(cpu)
memory_usage = MetricsCalculator.calculate_memory_usage(memory_used=8000, memory_total=16000)

# Visualize performance metrics
visualize_performance(cpu_utilization=cpu_utilization, memory_usage=memory_usage)
