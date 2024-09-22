from __future__ import annotations

import logging

import cpuinfo
import GPUtil
from prettytable import PrettyTable


class SystemInfo:
    @staticmethod
    def get_cpu_info() -> str:
        logging.info("Retrieving CPU information...")
        cpu_info = cpuinfo.get_cpu_info()["brand_raw"]
        logging.info("CPU information retrieved successfully.")
        return cpu_info

    @staticmethod
    def get_gpu_info() -> list[dict[str, str]]:
        logging.info("Retrieving GPU information...")
        GPUs = GPUtil.getGPUs()
        gpu_info = [
            {
                "GPU": f"GPU-{i}",
                "Name": gpu.name,
                "Memory Available (MB)": f"{gpu.memoryFree}/{gpu.memoryTotal}",
            }
            for i, gpu in enumerate(GPUs)
        ]
        logging.info("GPU information retrieved successfully.")
        return gpu_info

    @classmethod
    def print_system_info(cls):
        table = PrettyTable()
        table.field_names = ["Component", "Info"]
        table.add_row(["CPU", cls.get_cpu_info()])
        for info in cls.get_gpu_info():
            table.add_row(
                [
                    info["GPU"],
                    f"{info['Name']} - memory available: {info['Memory Available (MB)']}",
                ],
            )
        print(table)
