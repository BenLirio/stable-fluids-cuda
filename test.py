import glob
import os
from subprocess import run, Popen, PIPE, STDOUT, DEVNULL
from random import choice
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import argparse

VIZ_PATH = '/usr/bin/viz'
ANIMATION_DURATION_SECONDS = 6

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

def generate_gif(config):
  with TemporaryDirectory() as build_dir:
    run([
      'cmake',
      f'-DNUM_STEPS={config["NUM_STEPS"]}',
      f'-DWIDTH={config["WIDTH"]}',
      f'-DHEIGHT={config["HEIGHT"]}',
      f'-DDIFFUSION_RATE={config["DIFFUSION_RATE"]}',
      f'-DVISCOSITY={config["VISCOSITY"]}',
      f'-DGAUSS_SEIDEL_ITERATIONS={config["GAUSS_SEIDEL_ITERATIONS"]}',
      f'-DTIME_STEP={config["TIME_STEP"]}',
      '-B',
      build_dir,
      '-S',
      '.',
    ])
    run([
      'make',
      '-C',
      build_dir,
      'stable-fluids-cuda'
    ])
    stable_fluids_process = Popen([
      f'{build_dir}/src/stable-fluids-cuda',
    ], stdout=PIPE, stderr=sys.stderr)
    Path('gifs').mkdir(parents=True, exist_ok=True)
    with open(f'gifs/{config_to_string(config)}.gif', 'w') as gif_file:
      viz_process = Popen([
        'python',
        VIZ_PATH,
        f'--num-frames={config["NUM_STEPS"]}',
        f'--width={config["WIDTH"]}',
        f'--height={config["HEIGHT"]}',
        '--input-format=csv',
        '--output-format=gif',
        '--encoding=scalar',
        f'--duration={int(ANIMATION_DURATION_SECONDS*1000/config["NUM_STEPS"])}',
      ], stdin=stable_fluids_process.stdout, stdout=gif_file)
      stable_fluids_process.stdout.close()
      viz_process.communicate()


parser = argparse.ArgumentParser()
parser.add_argument('--vary-on', type=str, default='None')
parser.add_argument('--num-steps', type=int, default=choice(NUM_STEPS_VARIATIONS))
parser.add_argument('--width', type=int, default=choice(WIDTH_VARIATIONS))
parser.add_argument('--height', type=int, default=choice(HEIGHT_VARIATIONS))
parser.add_argument('--diffusion-rate', type=float, default=choice(DIFFUSION_RATE_VARIATIONS))
parser.add_argument('--viscosity', type=float, default=choice(VISCOSITY_VARIATIONS))
parser.add_argument('--gauss-seidel-iterations', type=int, default=choice(GAUSS_SEIDEL_ITERATIONS_VARIATIONS))
parser.add_argument('--time-step', type=float, default=choice(TIME_STEP_VARIATIONS))


if __name__ == '__main__':
  for path in glob.glob('gifs/*.gif'):
    os.remove(path)

  args = parser.parse_args()
  config = {
    "NUM_STEPS": args.num_steps,
    "WIDTH": args.width,
    "HEIGHT": args.height,
    "DIFFUSION_RATE": args.diffusion_rate,
    "VISCOSITY": args.viscosity,
    "GAUSS_SEIDEL_ITERATIONS": args.gauss_seidel_iterations,
    "TIME_STEP": args.time_step,
  }
  if args.vary_on == 'None':
    generate_gif(config)
    exit(0)
  for i in range(NUM_VARIATIONS):
    config[args.vary_on] = choice(locals()[f'{args.vary_on}_VARIATIONS'])
    generate_gif(config)