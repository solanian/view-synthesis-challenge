import os

def gen_config(dataset):
	content = f'''dataset_name = ilsh
datadir = ../data_root/chaPhase/{dataset}
expname = ilsh_{dataset}
basedir = ./log

n_iters = 50000
batch_size = 4096

N_voxel_init = 2097156 # 128**3
N_voxel_final = 64000000 # 400**3
upsamp_list = [2000,3000,4000,5500,7000,8500,10000,12000,14000,16000]
update_AlphaMask_list = [2000,4000]

N_vis = 1
vis_every = 25000
hold_every = -1

n_lamb_sigma = [16,4,4]
n_lamb_sh = [48,12,12]
model_name = TensorVMSplit_ZipNeRF

shadingMode = mixed
fea2denseAct = relu

view_pe = 0
fea_pe = 0

view_pe = 0
fea_pe = 0

TV_weight_density = 1
TV_weight_app = 1.0

rm_weight_mask_thre = 1e-4
	'''

	with open(f'ilsh_{dataset}.txt', 'w') as f:
		f.write(content.strip())

dataset_list = os.listdir('../../../data_root/chaPhase')

for dataset in dataset_list:
	if '.json' in dataset:
		continue
	gen_config(dataset)