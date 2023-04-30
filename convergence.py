import glob
import pickle
from sfc import get_config, OUTPUT_PERFORMANCE, build, USE_SHARED_MEMORY, USE_THREAD_COARSENING, USE_NAIVE, USE_ROW_COARSENING, OUTPUT_SOLVE_ERROR, USE_RED_BLACK, USE_NO_BLOCK_SYNC, USE_THREAD_FENCE, USE_NO_IDX
from pathlib import Path
from uuid import uuid4
import shutil

from timings import generate_timings


output_dir = 'timings'
if __name__ == '__main__':

  config = get_config(output=OUTPUT_PERFORMANCE|OUTPUT_SOLVE_ERROR)

  timings = []

  ns = [16]

  for feature in [USE_THREAD_FENCE|USE_SHARED_MEMORY]:

    for n in ns:
      current_config = config.copy()
      current_config['WIDTH'] = n
      current_config['HEIGHT'] = n

      current_config['KERNEL_FLAGS'] |= feature

      timings += generate_timings(current_config)

  exit(0)

  # list(np.logspace(5, 11, num=20, base=2, dtype=int))