from subprocess import run
import argparse
import numpy as np
import argparse
from random import choice
import hashlib

GRAPH_DIR = 'graphs'

tags = [
  'TOTAL',
  'ADVECT',
  'DIFFUSE',
  'PROJECT',
  'COLOR',
  'VELOCITY',
  'SOLVE',
  'STEP',
  'SOURCE',
  'SINK'
]
def string_of_log_tags(log):
  active_tags = [ tag for tag in tags if tag in log ]
  return ' + '.join(active_tags)

# OUTPUT_FLAGS
OUTPUT_PERFORMANCE = 1<<0
OUTPUT_GIF = 1<<1
OUTPUT_SOLVE_ERROR = 1<<2

# KERNEL_FLAGS
USE_NAIVE = 0
USE_SHARED_MEMORY = 1<<0
USE_THREAD_COARSENING = 1<<1
USE_ROW_COARSENING = 1<<2
USE_NO_BLOCK_SYNC = 1<<3
USE_RED_BLACK = 1<<4
USE_THREAD_FENCE = 1<<5
USE_NO_IDX = 1<<6

all_kernel_flags = [
  USE_NAIVE,
  USE_SHARED_MEMORY,
  USE_THREAD_COARSENING,
  USE_ROW_COARSENING,
  USE_NO_BLOCK_SYNC,
  USE_RED_BLACK,
  USE_THREAD_FENCE,
  USE_NO_IDX,
]

def string_of_kernel_flag(kernel_flag):
  if kernel_flag == USE_NAIVE: return 'Naive'
  if kernel_flag == USE_SHARED_MEMORY: return 'Shared Memory'
  if kernel_flag == USE_THREAD_COARSENING: return 'Thread Coarsening'
  if kernel_flag == USE_ROW_COARSENING: return 'Row Coarsening'
  if kernel_flag == USE_NO_BLOCK_SYNC: return 'No Block Sync'
  if kernel_flag == USE_RED_BLACK: return 'Red-Black'
  if kernel_flag == USE_THREAD_FENCE: return 'Thread Fence'
  if kernel_flag == USE_NO_IDX: return 'No Idx'


def string_of_kernel_flags(kernel_flag):
  ss = []
  for flag in all_kernel_flags:
    if kernel_flag & flag:
      ss.append(string_of_kernel_flag(flag))
  return ', '.join(ss)


DEFAULT_STEPS = 3
DEFAULT_WIDTH = 16
DEFAULT_HEIGHT = 16
DEFAULT_DIFFUSION_RATE = 0.0001
DEFAULT_VISCOSITY = 0.0005
DEFAULT_GAUSS_SEIDEL_ITERATIONS = 20
DEFAULT_TIME_STEP = 0.01



def cmake(config, build_dir):
  compile_flags = [ f'-D{KEY}={config[KEY]}' for KEY in config ]
  cmd_options = [ '-B', build_dir, '-S', '.']
  run(['cmake'] + compile_flags + cmd_options)

def make(build_dir):
  run([ 'make', '-C', build_dir, 'stable-fluids-cuda' ])

def build(config, build_dir):
  cmake(config, build_dir)
  make(build_dir)


def get_config(output):
  parser = argparse.ArgumentParser()
  parser.add_argument('--num-steps', type=int, default=DEFAULT_STEPS)
  parser.add_argument('--width', type=int, default=DEFAULT_WIDTH)
  parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT)
  parser.add_argument('--diffusion-rate', type=float, default=DEFAULT_DIFFUSION_RATE)
  parser.add_argument('--viscosity', type=float, default=DEFAULT_VISCOSITY)
  parser.add_argument('--gauss-seidel-iterations', type=int, default=DEFAULT_GAUSS_SEIDEL_ITERATIONS)
  parser.add_argument('--time-step', type=float, default=DEFAULT_TIME_STEP)
  parser.add_argument('--use-gold', action='store_true')
  parser.add_argument('--use-shared-memory', action='store_true')
  parser.add_argument('--use-thread-coarsening', action='store_true')
  parser.add_argument('--use-row-coarsening', action='store_true')
  parser.add_argument('--use-no-block-sync', action='store_true')
  parser.add_argument('--use-red-black', action='store_true')
  parser.add_argument('--use-thread-fence', action='store_true')
  parser.add_argument('--use-no-idx', action='store_true')

  args = parser.parse_args()
  config = {
    'NUM_STEPS': args.num_steps,
    'WIDTH': args.width,
    'HEIGHT': args.height,
    'DIFFUSION_RATE': args.diffusion_rate,
    'VISCOSITY': args.viscosity,
    'GAUSS_SEIDEL_ITERATIONS': args.gauss_seidel_iterations,
    'TIME_STEP': args.time_step,
    'USE_GOLD': 1 if args.use_gold else 0,
    'OUTPUT': output,
    'KERNEL_FLAGS': 0
  }
  if args.use_shared_memory: config['KERNEL_FLAGS'] |= USE_SHARED_MEMORY
  if args.use_thread_coarsening: config['KERNEL_FLAGS'] |= USE_THREAD_COARSENING
  if args.use_row_coarsening: config['KERNEL_FLAGS'] |= USE_ROW_COARSENING
  if args.use_no_block_sync: config['KERNEL_FLAGS'] |= USE_NO_BLOCK_SYNC
  if args.use_red_black: config['KERNEL_FLAGS'] |= USE_RED_BLACK
  if args.use_thread_fence: config['KERNEL_FLAGS'] |= USE_THREAD_FENCE
  if args.use_no_idx: config['KERNEL_FLAGS'] |= USE_NO_IDX

  return config


def hash_of_config(config):
  return hashlib.md5(f"""
{config["NUM_STEPS"]}
{config["WIDTH"]}
{config["HEIGHT"]}
{config["DIFFUSION_RATE"]}
{config["VISCOSITY"]}
{config["GAUSS_SEIDEL_ITERATIONS"]}
{config["TIME_STEP"]}
{config["USE_GOLD"]}
{config["KERNEL_FLAGS"]}
""".encode()).hexdigest()

def string_of_config(config):
  return ' | '.join([
    f'Kernel Flags: {string_of_kernel_flags(config["KERNEL_FLAGS"])}',
    f'Dim: ({config["WIDTH"]}x{config["HEIGHT"]})',
    f'GS Steps: {config["GAUSS_SEIDEL_ITERATIONS"]}',
  ])
