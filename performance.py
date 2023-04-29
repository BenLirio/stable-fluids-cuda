import glob
import numpy as np
import os
import pickle
import re
from subprocess import run, Popen, PIPE
from tempfile import TemporaryDirectory
from sfc import get_config, OUTPUT_PERFORMANCE, build, USE_SHARED_MEMORY, USE_THREAD_COARSENING, USE_NAIVE, USE_ROW_COARSENING, OUTPUT_SOLVE_ERROR, USE_RED_BLACK, USE_NO_BLOCK_SYNC, USE_THREAD_FENCE
from pathlib import Path
from uuid import uuid4
import shutil

tags = [
  'TOTAL',
  'ADVECT',
  'DIFFUSE',
  'PROJECT',
  'COLOR',
  'VELOCITY',
  'SOLVE',
]

def parse_timing_line(line):
  fields = {
    'TAGS': [],
    'VALUES': {}
  }
  for tag in tags:
    if f'[{tag}]' in line:
      fields['TAGS'].append(tag)

  patterns = [
    [r'\[time=(\d+\.\d+)\]', float, 'TIME'],
    [r'\[error=(\d+\.\d+)\]', float, 'ERROR'],
    [r'\[step=(\d+)\]', int, 'STEP'],
    [r'\[gauss_step=(\d+)\]', int, 'GAUSS_STEP'],
  ]
  for [pattern, typecast, label] in patterns:
    match = re.search(pattern, line)
    if match:
      value = typecast(match.group(1))
      fields['VALUES'][label] = value

  return [fields]

def parse_timings(performance_output):
  performance_data = []
  for line in performance_output.splitlines():
    performance_data += parse_timing_line(line)
  return performance_data


def generate_timings(config):
  with TemporaryDirectory() as build_dir:
    build(config, build_dir)
    stable_fluids_process = run([
      f'{build_dir}/src/stable-fluids-cuda',
    ], capture_output=True, text=True)
    print(stable_fluids_process.stderr)

    timings = parse_timings(stable_fluids_process.stdout)
    for timing in timings:
      timing['VALUES']['WIDTH'] = config['WIDTH']
      timing['VALUES']['HEIGHT'] = config['HEIGHT']
      timing['VALUES']['GAUSS_SEIDEL_ITERATIONS'] = config['GAUSS_SEIDEL_ITERATIONS']
      timing['VALUES']['KERNEL_FLAGS'] = config['KERNEL_FLAGS']

      # if config['USE_GOLD']:                            timing['TAGS'] += ['CPU']
      # else:                                             timing['TAGS'] += ['GPU']
      # if config['KERNEL_FLAGS']&USE_SHARED_MEMORY:      timing['TAGS'] += ['SHARED_MEMORY']
      # if config['KERNEL_FLAGS']&USE_THREAD_COARSENING:  timing['TAGS'] += ['THREAD_COARSENING']
      # if config['KERNEL_FLAGS']&USE_ROW_COARSENING:     timing['TAGS'] += ['ROW_COARSENING']
      # if config['USE_GOLD'] == 0 and \
      #    config['KERNEL_FLAGS']==USE_NAIVE:             timing['TAGS'] += ['NAIVE']
    return timings


output_dir = 'timings'
if __name__ == '__main__':
  Path(f'{output_dir}/old').mkdir(parents=True, exist_ok=True)

  for path in glob.glob(f'{output_dir}/*.pkl'):
    uid = uuid4().hex
    shutil.move(path, f'{output_dir}/old/{uid}.pkl')

  config = get_config(output=OUTPUT_PERFORMANCE|OUTPUT_SOLVE_ERROR)

  timings = []
  # for feature in [USE_NAIVE, USE_SHARED_MEMORY, USE_THREAD_COARSENING, USE_ROW_COARSENING]:

  ns = [64]

  # for feature in [USE_NO_BLOCK_SYNC, USE_RED_BLACK, USE_THREAD_FENCE, USE_RED_BLACK|USE_THREAD_FENCE, USE_RED_BLACK|USE_THREAD_FENCE|USE_SHARED_MEMORY]:
  for feature in [USE_RED_BLACK, USE_THREAD_FENCE|USE_SHARED_MEMORY]:

    for n in ns:
      current_config = config.copy()
      current_config['WIDTH'] = n
      current_config['HEIGHT'] = n

      current_config['KERNEL_FLAGS'] |= feature

      timings += generate_timings(current_config)

  with open(f'{output_dir}/timings.pkl', 'wb') as performance_file:
    pickle.dump(timings, performance_file)

  exit(0)

  # list(np.logspace(5, 11, num=20, base=2, dtype=int))