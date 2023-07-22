# setup dataset

```
./setup_challenge_dataset.sh
```

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

