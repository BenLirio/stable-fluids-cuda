import glob
import os
from subprocess import run, Popen, PIPE, STDOUT, DEVNULL
from random import choice
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

VIZ_PATH = '/usr/bin/viz'


NUM_STEPS_VARIATIONS = [ 89, 1, 7, 30 ]
WIDTH_VARIATIONS = [ 32, 3, 7, 19, 31, 72 ]
HEIGHT_VARIATIONS = WIDTH_VARIATIONS
DIFFUSION_RATE_VARIATIONS = [ 0.01, 0.005, 0.05, 0.1 ]
VISCOSITY_VARIATIONS = [ 0.005, 0.01, 0.05, 0.1 ]
GAUSS_SEIDEL_ITERATIONS_VARIATIONS = [ 20, 3, 10, 30, 40, 50, 100 ]
TIME_STEP_VARIATIONS = [ 0.01, 0.1, 1.0, 2, 0.0001 ]

def default_config():
  return {
      "NUM_STEPS": NUM_STEPS_VARIATIONS[0],
      "WIDTH": WIDTH_VARIATIONS[0],
      "HEIGHT": HEIGHT_VARIATIONS[0],
      "DIFFUSION_RATE": DIFFUSION_RATE_VARIATIONS[0],
      "VISCOSITY": VISCOSITY_VARIATIONS[0],
      "GAUSS_SEIDEL_ITERATIONS": GAUSS_SEIDEL_ITERATIONS_VARIATIONS[0],
      "TIME_STEP": TIME_STEP_VARIATIONS[0],
  }
def random_config():
  width = choice(WIDTH_VARIATIONS)
  height = choice(HEIGHT_VARIATIONS)
  diffusion_rate = choice(DIFFUSION_RATE_VARIATIONS)
  time_step = choice(TIME_STEP_VARIATIONS)
  num_steps = choice(NUM_STEPS_VARIATIONS)
  # overrides
  height = width
  diffusion_rate = 0.001
  time_step = 0.01
  num_steps = 5
  return {
      "NUM_STEPS": num_steps,
      "WIDTH": width,
      "HEIGHT": height,
      "DIFFUSION_RATE": diffusion_rate,
      "VISCOSITY": choice(VISCOSITY_VARIATIONS),
      "GAUSS_SEIDEL_ITERATIONS": choice(GAUSS_SEIDEL_ITERATIONS_VARIATIONS),
      "TIME_STEP": time_step,
  }
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
    ])
    stable_fluids_process = Popen([
      f'{build_dir}/stable-fluids-cuda',
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
        f'--duration={int(3000/config["NUM_STEPS"])}',
      ], stdin=stable_fluids_process.stdout, stdout=gif_file)
      stable_fluids_process.stdout.close()
      viz_process.communicate()




if __name__ == '__main__':
  for path in glob.glob('gifs/*'):
    print(path)
    os.remove(path)
  if len(sys.argv) != 2:
    print('Usage: python test.py <num-gifs-to-generate>')
    exit(1)
  generate_gif(default_config())
  for i in range(int(sys.argv[1])-1):
    generate_gif(random_config())