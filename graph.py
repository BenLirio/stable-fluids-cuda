from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sfc
import plotly.figure_factory as ff

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

def create_gauss_solve_error_graph(timings, kernel_flags=None):
  fig, ax = plt.subplots()

  timings = [ x for x in timings if 'SOLVE' in x['TAGS'] ]

  if kernel_flags is None:
    kernel_flags = list(set([ x['VALUES']['KERNEL_FLAGS'] for x in timings ]))
  kernel_flags.sort()


  for kernel_flag in kernel_flags:
    kernel_flag_timings = [ x for x in timings if (x['VALUES']['KERNEL_FLAGS'] == kernel_flag) ]
    diffuse_timings = [ x for x in kernel_flag_timings if 'DIFFUSE' in x['TAGS'] ]
    project_timings = [ x for x in kernel_flag_timings if 'PROJECT' in x['TAGS'] ]

    for [kernel_function_name, kernel_flag_function_timings] in [('project', project_timings)]:
      gauss_steps = list(set([ int(x['VALUES']['GAUSS_STEP']) for x in kernel_flag_function_timings ]))
      reverse_gauss_step_map = {}
      for i in range(len(gauss_steps)):
        reverse_gauss_step_map[gauss_steps[i]] = i
      
      errors = np.zeros(len(gauss_steps), dtype=float)
      idx_count = np.zeros(len(gauss_steps), dtype=int)
      for timing in kernel_flag_function_timings:
        gauss_step = int(timing['VALUES']['GAUSS_STEP'])
        idx = reverse_gauss_step_map[gauss_step]
        idx_count[idx] += 1
        if 'ERROR' not in timing['VALUES']:
          print("WARNING: No error in timing")
        errors[idx] = timing['VALUES'].get('ERROR', 0)
      # kernel_flag_function_timings.sort(key=lambda x: int(x['VALUES']['TIME']))
      # times = [ x['VALUES']['TIME'] for x in kernel_flag_function_timings ]
      # errors = [ x['VALUES']['ERROR'] for x in kernel_flag_function_timings ]

      
      label = f'{sfc.string_of_kernel_flags(kernel_flag)} ({kernel_function_name})'
      ax.plot(gauss_steps, errors, label=label)
      # ax.plot(times, errors, label=label)

  ax.legend()
  fig.savefig(f'{output_dir}/gauss_solve_error.png')




def create_gantt_chart(timings, kernel_flags=None):
  data = []
  for kernel_flag in kernel_flags:
    kernel_flag_timings = [ x for x in timings if (x['VALUES']['KERNEL_FLAGS'] == kernel_flag) ]
    timings_by_id = {}
    for timing in kernel_flag_timings:
      if timings_by_id.get(timing['VALUES']['ID']) is None:
        timings_by_id[timing['VALUES']['ID']] = []
      timings_by_id[timing['VALUES']['ID']].append(timing)
    timing_pairs = list(timings_by_id.values())
    for timing_pair in timing_pairs:
      timing_pair.sort(key=lambda x: x['VALUES']['TIME'])
      assert(len(timing_pair) == 2)

    def timing_to_gantt_entry(timing_pair):
      [start_timing, end_timing] = timing_pair
      task = 'Unknown'
      if 'DIFFUSE' in start_timing['TAGS']:
        task = 'Diffuse'
      elif 'PROJECT' in start_timing['TAGS']:
        task = 'Project'
      elif 'ADVECT' in start_timing['TAGS']:
        task = 'Advect'

      return {
        'Task': task,
        'Start': start_timing['VALUES']['TIME'],
        'Finish': end_timing['VALUES']['TIME'],
        'Resource': sfc.string_of_kernel_flag(kernel_flag)
      }
    
    data = data + [ timing_to_gantt_entry(timing_pair) for timing_pair in timing_pairs ]

  # # Create a color scale for resources (optional)
  # colors = {
  #     "Resource 1": "rgb(220, 0, 0)",
  #     "Resource 2": "rgb(0, 0, 220)",
  # }

  # # Create the Gantt chart
  fig = ff.create_gantt(data, index_col="Resource", show_colorbar=True, group_tasks=True)

  # # Show the Gantt chart
  fig.show()


if __name__ == '__main__':
  Path(f'{output_dir}').mkdir(parents=True, exist_ok=True)
  with open('timings/timings.pkl', 'rb') as f:
    timings = pickle.load(f)

  features = [
    sfc.USE_THREAD_FENCE|sfc.USE_SHARED_MEMORY|sfc.USE_NO_IDX,
    sfc.USE_THREAD_FENCE|sfc.USE_SHARED_MEMORY,
  ]
  # create_gantt_chart(timings, features)

  create_gauss_solve_error_graph(timings, features) # This one is GOOD

  # create_kernel_feature_bar_graph(timings, ['NAIVE', 'SHARED_MEMORY', 'ROW_COARSENING'])
  # create_kernel_feature_line_graph(timings, ['NAIVE', 'SHARED_MEMORY', 'ROW_COARSENING'])
  # create_kernel_feature_bar_graph(timings, ['NAIVE', 'CPU'])
  # create_kernel_feature_line_graph(timings, ['NAIVE', 'CPU'])
  # graph_shared_memory_functions(timings)
  # graph_functions(timings)
  # graph_change_in_n(timings)
  # graph_shared_memory(timings)