import glob
import pickle
from sfc import get_config, OUTPUT_PERFORMANCE, build, USE_SHARED_MEMORY, USE_THREAD_COARSENING, USE_NAIVE, USE_ROW_COARSENING, OUTPUT_SOLVE_ERROR, USE_RED_BLACK, USE_NO_BLOCK_SYNC, USE_THREAD_FENCE
from pathlib import Path
from uuid import uuid4
import shutil

from timings import generate_timings


output_dir = 'timings'
if __name__ == '__main__':
  Path(f'{output_dir}/old').mkdir(parents=True, exist_ok=True)

  for path in glob.glob(f'{output_dir}/*.pkl'):
    uid = uuid4().hex
    shutil.move(path, f'{output_dir}/old/{uid}.pkl')

  config = get_config(output=OUTPUT_PERFORMANCE)

  timings = []

  ns = [1024]

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