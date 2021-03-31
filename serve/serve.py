import os
from pathlib import Path
import sys
import supervisely_lib as sly

root_source_path = str(Path(sys.argv[0]).parents[1])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

import unet
from unet import construct_unet

