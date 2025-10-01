
FEATURE_PATH='resnet18_CIFAR100_dict.pickle'
MODEL_TYPE='ResNet18'
MODEL_PATH='models/resnet18_CIFAR100/checkpoint_best.pth'
DATA_PATH='datasets'
DATASET='CIFAR100'


CUDA_VISIBLE_DEVICES=0 python3 get_feature_maps.py --feature-maps-path $FEATURE_PATH --model-type $MODEL_TYPE --model-path $MODEL_PATH --dataset $DATASET --data-root-path $DATA_PATH 
