# %%
import sys
sys.path.append('/home/steveyang/.jupyternb/hw3/python')
import numpy as np
import needle as ndl
from needle import backend_ndarray as nd
import timeit
import statistics
import matplotlib.pyplot as plt

# %%
shape = np.arange(5, 12)
shapes = [(2**s, 2**s) for s in shape]
devices = [nd.cuda(), nd.cpu()]
n_runs = 100
results = {str(device): {} for device in devices}

# %%
for device in devices:
    for shape in shapes:
        if device == nd.cpu() and shape[0] > 256:
            continue
        _A = np.random.randint(low=0, high=10, size=shape)
        _B = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=device)
        B = nd.array(_B, device=device)

        def matmul():
            return A @ B

        times = timeit.repeat(matmul, number=n_runs)

        print("Device:", device)
        print("Shape:", shape)
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times)
        print("Average time:", avg_time)
        print("Standard deviation:", std_dev)
        print()

        results[str(device)][str(shape)] = (avg_time, std_dev)

# %%
# Plotting
fig, axs = plt.subplots(2, 1, figsize=(8, 12))

ax = axs[0]
for device, device_results in results.items():
    labels = list(device_results.keys())
    avg_times = [res[0] for res in device_results.values()]
    std_devs = [res[1] for res in device_results.values()]
    ax.errorbar(labels, avg_times, yerr=std_devs, fmt='o', label=str(device))

ax.legend()
ax.set_title('Matrix multiplication time by shape and device')
ax.set_xlabel('Shape')
ax.set_ylabel('Time (s)')
ax.tick_params(axis='x', rotation='vertical')

ax = axs[1]
cpu_times = [res[0] for res in results[str(nd.cpu())].values()]
gpu_times = [res[0] for res in results[str(nd.cuda())].values()]
ratios = [cpu / gpu for cpu, gpu in zip(cpu_times, gpu_times)]
ax.plot(labels, ratios, 'o-')
ax.set_title('CPU/GPU time ratio')
ax.set_xlabel('Shape')
ax.set_ylabel('Ratio')

plt.tight_layout()
plt.show()