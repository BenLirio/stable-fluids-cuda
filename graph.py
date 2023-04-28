from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle

output_dir = 'graphs'

def create_kernel_feature_line_graph(timings, kernel_features):
  ns = list(set([ int(x['VALUES']['WIDTH']) for x in timings ]))
  ns = list(set([ int(x['VALUES']['WIDTH']) for x in timings ]))
  ns.sort()
  idx_of_n = {}
  for i in range(len(ns)):
    idx_of_n[ns[i]] = i
  
  num_ns = len(ns)
  num_features = len(kernel_features)
  data = np.zeros((num_features, num_ns), dtype=float)

  for timing in timings:
    n = int(timing['VALUES']['WIDTH'])
    idx = idx_of_n[n]
    feature = 'NAIVE'
    for f in kernel_features:
      if f in timing['TAGS']:
        feature = f
    data[kernel_features.index(feature), idx] += timing['VALUES']['TIME']
  
  for i in range(num_features):
    for j in range(num_ns):
      assert data[i,j] > 0, f'No data for {kernel_features[i]} {ns[j]}'
  
  fig, ax = plt.subplots()
  for i in range(num_features):
    ax.plot(ns, data[i,:], label=kernel_features[i])
  ax.set_title('Kernel Feature Comparison')
  ax.set_xlabel('Width')
  ax.set_ylabel('Time (ms)')
  ax.legend()
  fig.savefig(f'{output_dir}/kernel_feature_line_graph.png')


def create_kernel_feature_bar_graph(timings, kernel_features):
  kernel_functions = ['ADVECT', 'DIFFUSE', 'PROJECT']
  feature_function_timing_list = {}
  for feature in kernel_features:
    feature_function_timing_list[feature] = {}
    for function in kernel_functions:
      feature_function_timing_list[feature][function] = []

  for timing in timings:

    for feature in kernel_features:
      if feature in timing['TAGS']:
        timing_feature = feature

    timing_function = 'UNKNOWN'
    for function in kernel_functions:
      if function in timing['TAGS']:
        timing_function = function
    
    assert timing_function != 'UNKNOWN', f'Could not find function in timing: {timing}'

    feature_function_timing_list[timing_feature][timing_function].append(timing['VALUES']['TIME'])
  
  for feature in kernel_features:
    for function in kernel_functions:
      assert len(feature_function_timing_list[feature][function]) > 0, f'No timings for {feature} {function}'

  feature_function_timing_totals = {}
  for feature in kernel_features:
    feature_function_timing_totals[feature] = {}
    for function in kernel_functions:
      feature_function_timing_totals[feature][function] = sum(feature_function_timing_list[feature][function])

  xs = np.arange(len(kernel_functions))
  box_width = 0.25
  fig, ax = plt.subplots()
  for i, feature in enumerate(kernel_features):
    x = xs + (i * box_width)
    ax.bar(x, feature_function_timing_totals[feature].values(), width=box_width, label=feature)
  ax.set_title('Kernel Feature Comparison')
  ax.set_xlabel('Kernel Function')
  ax.set_ylabel('Time (s)')
  ax.set_xticks(xs)
  ax.set_xticklabels(kernel_functions)
  ax.legend()
  fig.savefig(f'{output_dir}/kernel_feature_comparison.png')

  
def graph_change_in_n(timings):
  ns = list(set([ int(x['VALUES']['WIDTH']) for x in timings ]))
  ns.sort()
  n_map = {}
  for i in range(len(ns)):
    n_map[ns[i]] = i

  cpu_times = np.zeros(len(ns), dtype=float)
  kernel_times = np.zeros(len(ns), dtype=float)

  for timing in timings:
    n = int(timing['VALUES']['WIDTH'])
    idx = n_map[n]
    time = timing['VALUES']['TIME']
    if 'CPU' in timing['TAGS']:
      cpu_times[idx] += time
    if 'GPU' in timing['TAGS']:
      kernel_times[idx] += time

  fig, ax = plt.subplots()

  ax.set_title('CPU vs GPU Timings by N')
  ax.set_yscale('log')
  ax.set_xlabel('N')
  ax.set_ylabel('Time (ms)')
  ax.plot(ns, cpu_times, label='CPU')
  ax.plot(ns, kernel_times, label='Kernel')
  ax.grid(True, linestyle='--')

  ax.legend()
  fig.savefig('timings/change_in_n.png')

def graph_shared_memory(timings):
  ns = list(set([ int(x['VALUES']['WIDTH']) for x in timings ]))
  ns.sort()
  n_map = {}
  for i in range(len(ns)):
    n_map[ns[i]] = i

  naive_times = np.zeros(len(ns), dtype=float)
  shared_memory_times = np.zeros(len(ns), dtype=float)

  for timing in timings:
    n = int(timing['VALUES']['WIDTH'])
    idx = n_map[n]
    time = timing['VALUES']['TIME']
    if 'SHARED_MEMORY' in timing['TAGS']:
      shared_memory_times[idx] += time
    else:
      naive_times[idx] += time

  fig, ax = plt.subplots()

  ax.set_title('Using Shared Memory vs Naive')
  ax.set_yscale('log')
  ax.set_xlabel('N')
  ax.set_ylabel('Time (ms)')
  ax.plot(ns, naive_times, label='Naive')
  ax.plot(ns, shared_memory_times, label='Shared Memory')
  ax.grid(True, linestyle='--')

  ax.legend()
  fig.savefig('timings/shared_memory.png')


if __name__ == '__main__':
  Path(f'{output_dir}').mkdir(parents=True, exist_ok=True)
  with open('timings/timings.pkl', 'rb') as f:
    timings = pickle.load(f)

  # create_kernel_feature_bar_graph(timings, ['NAIVE', 'SHARED_MEMORY', 'ROW_COARSENING'])
  # create_kernel_feature_line_graph(timings, ['NAIVE', 'SHARED_MEMORY', 'ROW_COARSENING'])
  create_kernel_feature_bar_graph(timings, ['NAIVE', 'CPU'])
  create_kernel_feature_line_graph(timings, ['NAIVE', 'CPU'])
  # graph_shared_memory_functions(timings)
  # graph_functions(timings)
  # graph_change_in_n(timings)
  # graph_shared_memory(timings)