# Experiment all tricks without center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# without center loss
python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('occduke')" DATASETS.ROOT_DIR "('/home/zdm/rxn/myproject/dataset')" OUTPUT_DIR "('/home/zdm/rxn/myproject/baseline_with_relation/logs/occduke/Experiment-all-tricks-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on')"