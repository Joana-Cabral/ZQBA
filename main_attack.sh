
FEATURE_PATH='feature_maps/resnet18_CIFAR10_dict.pickle'
MODEL_TYPE='ResNet18'
MODEL_PATH='models/resnet18_CIFAR10/checkpoint_best.pth'
DATA_PATH='datasets'
DATASET='CIFAR10'


CUDA_VISIBLE_DEVICES=0 python3 main_attack.py --feature-maps-path $FEATURE_PATH --model-type $MODEL_TYPE --model-path $MODEL_PATH --dataset $DATASET --data-root-path $DATA_PATH 
