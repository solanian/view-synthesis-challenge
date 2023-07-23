# setup dataset

# setup requirments

```
./setup_requirement.sh
```

# train our Model "Mixed Renderer" 
## Change the number to training according to the desired subject num eg) ilsh_0XX_00.txt

```
cd TensoRF
python train.py --config configs/submit_result/ilsh_002_00.txt
```

# Rendering
### notice we rendering near_far = [3.6, 3.8] plz change (TensoRF/models/tensorBase.py#L275C1-L275C33)
```
python train.py --config <path the config> --ckpt <path the ckpt>  --render_visual 1

eg.)  python train.py --config /home/ubuntu/view-synthesis-challenge/TensoRF/configs/submit_result/ilsh_033_00.txt --ckpt /home/ubuntu/view-synthesis-challenge/TensoRF/log/ilsh_033_00/ilsh_033_00.th  --render_visual 1
```
###or If you want rendering all things in configs
```
./render.sh
```
