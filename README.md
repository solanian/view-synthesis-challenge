# setup dataset

```
./setup_challenge_dataset.sh
```

# train TensoRF

```
python TensoRF/train.py --config TensoRF/configs/ilsh.txt
```

# train GeoNeRF

```
python GeoNeRF/train.py --config GeoNeRF/configs/config_ilsh_finetune.txt
```