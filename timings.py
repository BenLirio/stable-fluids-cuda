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
    [r'\[id=(\d+)\]', int, 'ID'],
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

    return timings
