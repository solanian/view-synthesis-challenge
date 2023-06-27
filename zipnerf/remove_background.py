from pathlib import Path
from tqdm import tqdm
from rembg import remove, new_session

if __name__ == '__main__':
	root_dir = Path('data/devPhase')
	
	# session = new_session("u2net")
	# session = new_session("u2net_human_seg")
	session = new_session("isnet-general-use")
	# session = new_session("sam")

	# file_list = list(root_dir.glob('*'))
	for input_path in tqdm(root_dir.rglob('*')):
		if input_path.is_file() and input_path.suffix == '.jpg' and input_path.parent.name == 'images':
			output_path = Path(str(input_path.parent).replace('images', 'images_rembg'))
			if not output_path.exists():
				output_path.mkdir(parents=True, exist_ok=True)
			output_path = output_path / input_path.name
			with open(input_path, 'rb') as i:
				with open(output_path, 'wb') as o:
					input = i.read()
					output = remove(
						input, 
						session=session,
						bgcolor=tuple([0, 0, 0, 255])
					)
					o.write(output)
					print(f"write background removed image {output_path}")
			# print(input_path.parent)
			# print(input_path.name)
