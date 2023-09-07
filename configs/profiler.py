profiler = dict(
    torch_active_count=50,        # Control the frequency of torch profiler
    memory_active_count=50,       # Control the frequency of memory profiler
    bench_active_count=50,        # Control the frequency of cuda burn test and nccl-test
    diagnosis_active_count=50,    # Control the frequency of diagnosis
    
    diagnosis_start=10,           # Control the start batch count of diagnosis
    diagnosis_slower_check=20,    # Control the frequency of checking continuous slower step
)