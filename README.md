# GIFT: Learning Transformation-Invariant Dense Visual Descriptors via Group CNNs

Still under construction, please stay tunned~

## Download dataset

```shell script
mkdir data
cd data
wget https://1drv.ms/u/s!AoUMOA44sUHpbDlJ6ppgRm3_RYo?e=MIbyCz
wget https://1drv.ms/u/s!AoUMOA44sUHpawiL-9MF0gZRDxo?e=Uugchw
unzip pretrain_model.zip
unzip eval_data_hpatches.zip
```

## Compilation

1. Compile hard example mining

```shell script
cd hard_mining
python setup.py build_ext --inplace
```

2. Compile extend utils

```shell script
cd utils/extend_utils
python build_extend_utils.py
```

3. Run the evaluation on hpatches

```shell script
python run.py --task=eval --dataset=hv --cfg=configs/pretrain.yaml
```

4. If all things are correct, you will see a pck 67.14 on the HP-View dataset.