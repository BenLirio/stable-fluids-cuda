import glob
import os
import pickle
import re
from subprocess import run, Popen, PIPE
from tempfile import TemporaryDirectory
from sfc import get_config, cmake
from pathlib import Path

current_uid = 0

tags = [
  'PERFORMANCE',
  'CPU',
  'GPU',
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

  if 'PERFORMANCE' not in fields['TAGS']:
    return []

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


def generate_performance(config):
  global current_uid
  current_uid += 1
  with TemporaryDirectory() as build_dir:
    cmake(config, build_dir)
    run([
      'make',
      '-C',
      build_dir,
      'performance_cpu_test',
      'performance_kernel_test',
    ])

    cpu_test_process  = run([
      f'{build_dir}/tests/performance_cpu_test',
      '--verbose'
    ], capture_output=True, text=True)
    kernel_test_process = run([
      f'{build_dir}/tests/performance_kernel_test',
      '--verbose'
    ], capture_output=True, text=True)

    test_output = '\n'.join([cpu_test_process.stdout, kernel_test_process.stdout])
    timings = parse_timings(test_output)
    for timing in timings:
      timing['VALUES']['WIDTH'] = config['WIDTH']
      timing['VALUES']['HEIGHT'] = config['HEIGHT']
      timing['VALUES']['GAUSS_SEIDEL_ITERATIONS'] = config['GAUSS_SEIDEL_ITERATIONS']
      timing['VALUES']['UID'] = current_uid
    return timings


output_dir = 'timings'
if __name__ == '__main__':
  for path in glob.glob(f'{output_dir}/*.pkl'):
    os.remove(path)

  timings = generate_performance(get_config())
  Path(output_dir).mkdir(parents=True, exist_ok=True)
  with open(f'{output_dir}/timings.pkl', 'wb') as performance_file:
    pickle.dump(timings, performance_file)

  exit(0)