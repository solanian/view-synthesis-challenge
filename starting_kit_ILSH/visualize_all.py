from pathlib import Path
from itertools import chain

import subprocess


if __name__ == '__main__':
	root_dir = Path('data/devPhase')
	
	# Combine the results of glob for both directories
	input_paths = chain(
		(root_dir / "devPhase_01").glob("*"),
		(root_dir / "devPhase_02").glob("*")
	)

	# Iterate through the combined input paths
	for input_path in input_paths:
		if input_path.suffix == '.json':
			continue
		subprocess.run(['python3', 
			'starting_kit_ILSH/tip1_visualise_camera_poses.py',
			'--in_dir', input_path,
			'--out_path', input_path.name,
			'--save_jpg', 'True'], capture_output=True, text=True)

