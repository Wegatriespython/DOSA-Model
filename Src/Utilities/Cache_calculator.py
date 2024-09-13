def calculate_cache_size(n_workers, n_firms, time_steps, periods):
    entries = n_workers * n_firms * time_steps
    entry_size = 4 * periods * 8  # 4 lists, each with 'periods' floats
    total_bytes = entries * entry_size
    total_mb = total_bytes
    return total_mb

n_workers = 30
n_firms = 4
time_steps = 200
periods = 8

cache_size_bytes = calculate_cache_size(n_workers, n_firms, time_steps, periods)
print(f"Estimated cache size: {cache_size_bytes:.2f}")
