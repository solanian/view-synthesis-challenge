# setup dataset

```
./setup_challenge_dataset.sh
```

# train TensoRF

```
cd TensoRF
python train.py --config configs/ilsh.txt
```

# train GeoNeRF

```
cd GeoNeRF
python train.py --config configs/config_ilsh_finetune.txt
```