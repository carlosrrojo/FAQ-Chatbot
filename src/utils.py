import os

BENCHMARK_DIR = "benchmarks"

def load_benchmark(benchmark_name: str) -> list[str]:
    benchmark_path = os.path.join(BENCHMARK_DIR, benchmark_name)
    with open(benchmark_path, "r") as f:
        return [line.strip() for line in f if line.strip()]