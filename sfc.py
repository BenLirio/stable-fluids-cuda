from subprocess import run
import argparse
import numpy as np
import argparse
from random import choice

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

def string_of_kernel_flag(kernel_flag):
  if kernel_flag == USE_NAIVE: return 'Naive'
  if kernel_flag == USE_SHARED_MEMORY: return 'Shared Memory'
  if kernel_flag == USE_THREAD_COARSENING: return 'Thread Coarsening'
  if kernel_flag == USE_ROW_COARSENING: return 'Row Coarsening'
  if kernel_flag == USE_NO_BLOCK_SYNC: return 'No Block Sync'
  if kernel_flag == USE_RED_BLACK: return 'Red Black'
  if kernel_flag == USE_THREAD_FENCE: return 'Thread Fence'
  if kernel_flag == (USE_RED_BLACK|USE_THREAD_FENCE): return 'Red Black + Thread Fence'
  if kernel_flag == (USE_RED_BLACK|USE_THREAD_FENCE|USE_SHARED_MEMORY): return 'Red Black + Thread Fence + Shared Memory'
  if kernel_flag == (USE_THREAD_FENCE|USE_SHARED_MEMORY): return 'Thread Fence + Shared Memory'
  if kernel_flag == (USE_THREAD_FENCE|USE_SHARED_MEMORY|USE_NO_IDX): return 'Thread Fence + Shared Memory + No Idx'
  return 'Unknown'

NUM_VARIATIONS = 8
NUM_SAMPLES = 100
positive_int_normal = lambda mean, std: np.abs(np.random.normal(mean, std, NUM_SAMPLES).astype(int))
positive_float_normal = lambda mean, std: np.round(np.abs(np.random.normal(mean, std, NUM_SAMPLES)), 4)

NUM_STEPS_VARIATIONS = positive_int_normal(30, 10)
WIDTH_VARIATIONS = positive_int_normal(100, 30)
HEIGHT_VARIATIONS = WIDTH_VARIATIONS
DIFFUSION_RATE_VARIATIONS = positive_float_normal(0.001, 0.001)
VISCOSITY_VARIATIONS = positive_float_normal(0.05, 0.01)
GAUSS_SEIDEL_ITERATIONS_VARIATIONS = positive_int_normal(20, 20)
TIME_STEP_VARIATIONS = positive_float_normal(0.01, 0.001)

def cmake(config, build_dir):
  compile_flags = [ f'-D{KEY}={config[KEY]}' for KEY in config ]
  cmd_options = [ '-B', build_dir, '-S', '.']
  run(['cmake'] + compile_flags + cmd_options)

def make(build_dir):
  run([ 'make', '-C', build_dir, 'stable-fluids-cuda' ])

def build(config, build_dir):
  cmake(config, build_dir)
  make(build_dir)


def config_to_string(config):
  return '_'.join([
    f'steps={config["NUM_STEPS"]}',
    f'w={config["WIDTH"]}',
    f'h={config["HEIGHT"]}',
    f'diff={config["DIFFUSION_RATE"]}',
    f'visc={config["VISCOSITY"]}',
    f'gs={config["GAUSS_SEIDEL_ITERATIONS"]}',
    f'ts={config["TIME_STEP"]}',
  ])

def get_config(output):
  parser = argparse.ArgumentParser()
  parser.add_argument('--vary-on', type=str, default='None')
  parser.add_argument('--num-steps', type=int, default=choice(NUM_STEPS_VARIATIONS))
  parser.add_argument('--width', type=int, default=choice(WIDTH_VARIATIONS))
  parser.add_argument('--height', type=int, default=choice(HEIGHT_VARIATIONS))
  parser.add_argument('--diffusion-rate', type=float, default=choice(DIFFUSION_RATE_VARIATIONS))
  parser.add_argument('--viscosity', type=float, default=choice(VISCOSITY_VARIATIONS))
  parser.add_argument('--gauss-seidel-iterations', type=int, default=choice(GAUSS_SEIDEL_ITERATIONS_VARIATIONS))
  parser.add_argument('--time-step', type=float, default=choice(TIME_STEP_VARIATIONS))
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