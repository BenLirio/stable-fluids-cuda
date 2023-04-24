import glob
import numpy as np
import os
import pickle
import re
from subprocess import run, Popen, PIPE
from tempfile import TemporaryDirectory
from sfc import get_config, OUTPUT_PERFORMANCE, build, USE_SHARED_MEMORY
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
]

def parse_timing_line(line):
  fields = {
    'TAGS': [],
    'VALUES': {}
  }
  for tag in tags:
    if f'[{tag}]' in line:
      fields['TAGS'].append(tag)

  step_pattern = r'\[step=(\d+)\]'
  step_match = re.search(step_pattern, line)
  if step_match:
    step = int(step_match.group(1))
    fields['VALUES']['STEP'] = step
  else: raise Exception(f'Could not find step in line: {line}')

  time_pattern = r'\[time=(\d+\.\d+)\]'
  time_match = re.search(time_pattern, line)
  if time_match:
    time = float(time_match.group(1))
    fields['VALUES']['TIME'] = time
  else: raise Exception(f'Could not find time in line: {line}')

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

    timings = parse_timings(stable_fluids_process.stdout)
    for timing in timings:
      timing['VALUES']['WIDTH'] = config['WIDTH']
      timing['VALUES']['HEIGHT'] = config['HEIGHT']
      timing['VALUES']['GAUSS_SEIDEL_ITERATIONS'] = config['GAUSS_SEIDEL_ITERATIONS']
      if config['USE_GOLD']:
        timing['TAGS'] += ['CPU']
      else:
        timing['TAGS'] += ['GPU']
      if config['KERNEL_FLAGS']&USE_SHARED_MEMORY:
        timing['TAGS'] += ['SHARED_MEMORY']
    return timings


output_dir = 'timings'
if __name__ == '__main__':
  Path(f'{output_dir}/old').mkdir(parents=True, exist_ok=True)

  for path in glob.glob(f'{output_dir}/*.pkl'):
    uid = uuid4().hex
    shutil.move(path, f'{output_dir}/old/{uid}.pkl')

  config = get_config(output=OUTPUT_PERFORMANCE)

  timings = []
  # for n in list(np.logspace(5, 10, num=20, base=2, dtype=int)):
  for use_shared_memory in [True, False]:
    for n in [1024]:
      current_config = config.copy()

      if use_shared_memory: current_config['KERNEL_FLAGS'] |= USE_SHARED_MEMORY
      current_config['WIDTH'] = n
      current_config['HEIGHT'] = n

      timings += generate_timings(current_config)

  with open(f'{output_dir}/timings.pkl', 'wb') as performance_file:
    pickle.dump(timings, performance_file)

  exit(0)