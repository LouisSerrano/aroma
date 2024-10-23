# 0. Set your local dir
```bash
export LOCAL_DIR=YOUR_PATH_TO_LOCAL_DIR
```

# 1. Burgers 

```bash
python3 download_data_hugging_face.py local_dir=$LOCAL_DIR
```

# 2. NS 1e-4
```bash
python3 download_data_hugging_face.py local_dir=$LOCAL_DIR repo_id=sogeeking/navier-stokes-1e-4 file_list=[ns_V1e-4_N10000_T30.mat]
```

# 3. NS 1e-5
```bash
python3 download_data_hugging_face.py local_dir=$LOCAL_DIR repo_id=sogeeking/navier-stokes-1e-5 file_list=[NavierStokes_V1e-5_N1200_T20.mat]
```

# 4. SW
```bash
python3 download_data_hugging_face.py local_dir=$LOCAL_DIR repo_id=sogeeking/shallow-water file_list=[shallow_water_16_160_128_256_train.h5, shallow_water_2_160_128_256_test.h5]
```

# 5. NS
```bash
python3 download_data_hugging_face.py local_dir=$LOCAL_DIR repo_id=sogeeking/navier-stokes file_list=[navier_1e-3_256_2_test.shelve.dat,navier_1e-3_256_2_train.shelve.dat,navier_1e-3_256_2_test.shelve.dir,navier_1e-3_256_2_train.shelve.dir]
```

# 6. Cylinder Flow
```bash
sh download_deepmind_dataset.sh cylinder_flow $LOCAL_DIR
```

# 7. Airfoil Flow
```bash
sh download_deepmind_dataset.sh airfoil $LOCAL_DIR
```
