import glob
import os
import shutil
from subprocess import run, Popen, PIPE
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
from sfc import get_config, build, OUTPUT_GIF
import sfc
from uuid import uuid4

VIZ_PATH = '/usr/bin/viz'
ANIMATION_DURATION_SECONDS = 6

def generate_gif(config):
  with TemporaryDirectory() as build_dir:
    build(config, build_dir)

    stable_fluids_process = Popen([
      f'{build_dir}/src/stable-fluids-cuda',
    ], stdout=PIPE, stderr=sys.stderr)

    Path('gifs').mkdir(parents=True, exist_ok=True)
    with open(f'gifs/{sfc.string_of_config(config)}.gif', 'w') as gif_file:
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

output_dir = 'gifs'
if __name__ == '__main__':
  Path(f'{output_dir}/old').mkdir(parents=True, exist_ok=True)
  for path in glob.glob(f'{output_dir}/*.gif'):
    uid = uuid4().hex
    shutil.move(path, f'{output_dir}/old/{uid}.gif')

  config = get_config(output=OUTPUT_GIF)
  generate_gif(config)
  exit(0)

  # for i in range(NUM_VARIATIONS):
  #   config[args.vary_on] = choice(locals()[f'{args.vary_on}_VARIATIONS'])
  #   generate_gif(config)