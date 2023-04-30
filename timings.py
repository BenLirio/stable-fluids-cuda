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
import sfc

TIMING_DIR = 'timings'

def parse_timing_line(line):
  fields = {
  }
  for tag in sfc.tags:
    if f'[{tag}]' in line:
      fields[tag] = True

  patterns = [
    [r'\[time=(\d+\.\d+)\]', float, 'TIME'],
    [r'\[error=(\d+\.\d+)\]', float, 'ERROR'],
    [r'\[step=(\d+)\]', int, 'STEP'],
    [r'\[gauss_step=(\d+)\]', int, 'GAUSS_STEP'],
    [r'\[id=(\d+)\]', int, 'ID'],
    [r'\[depth=(\d+)\]', int, 'DEPTH'],
  ]
  for [pattern, typecast, label] in patterns:
    match = re.search(pattern, line)
    if match:
      value = typecast(match.group(1))
      fields[label] = value

  return [fields]

def parse_timings(performance_output):
  performance_data = []
  for line in performance_output.splitlines():
    performance_data += parse_timing_line(line)
  return performance_data


def generate_timings(config):
  config_hash = sfc.hash_of_config(config)
  if os.path.exists(f'{TIMING_DIR}/{config_hash}.pkl'):
    print(f'Found existing timings for {config}')
    with open(f'{TIMING_DIR}/{config_hash}.pkl', 'rb') as performance_file:
      return pickle.load(performance_file)

  with TemporaryDirectory() as build_dir:
    build(config, build_dir)
    stable_fluids_process = run([
      f'{build_dir}/src/stable-fluids-cuda',
    ], capture_output=True, text=True)
    print(stable_fluids_process.stderr)

    timings = parse_timings(stable_fluids_process.stdout)
    for timing in timings:
      timing['WIDTH'] = config['WIDTH']
      timing['HEIGHT'] = config['HEIGHT']
      timing['GAUSS_SEIDEL_ITERATIONS'] = config['GAUSS_SEIDEL_ITERATIONS']
      timing['KERNEL_FLAGS'] = config['KERNEL_FLAGS']
    
    assert len(timings) > 0, 'No timings found'

    with open(f'{TIMING_DIR}/{config_hash}.pkl', 'wb') as performance_file:
      pickle.dump(timings, performance_file)
    return timings


def as_log_pairs(logs):
  log_map = {}
  for log in logs:
    log_map[log['ID']] = log_map.get(log['ID'], []) + [log]
  log_pairs = list(log_map.values())
  for log_pair in log_pairs:
    assert len(log_pair) == 2
    log_pair.sort(key=lambda log: log['TIME'])
  return log_pairs