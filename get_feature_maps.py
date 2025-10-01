import argparse
import pickle
import os

import torch
import torchvision.transforms as transforms
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    deprocess_image
)
import torchvision
import torch.nn as nn

from tqdm import tqdm
from Tiny_dataset_loader import TinyImageNetDataset
from wideresnet import WideResNet

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-maps-path', type=str, default='feature_maps_resnet18_dict.pickle', help='Path to the pickle containing the feature maps')
    parser.add_argument('--model-type', type=str, default='ResNet18', help='Model architecture to evaluate')
    parser.add_argument('--model-path', type=str, default='../Adversarial_Attack/models/resnet18_CIFAR10_not_attacked/checkpoint_best.pth', help='Path to the previously trained model to evaluate')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to evaluate')
    parser.add_argument('--data-root-path', type=str, default='/home/joana/Adversarial_Attack/datasets', help='Path to the data directory')
    args = parser.parse_args()

    return args


def save_feat_maps(res_img_dict, dest_file):
    with open(dest_file, 'wb') as handle:
        pickle.dump(res_img_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = get_args()

    batch_size = 1

    dataset = args.dataset
    data_root_path = args.data_root_path
    
    model_type = args.model_type
    model_path = args.model_path

    feature_maps_root = 'feature_maps_teste'
    os.makedirs(feature_maps_root, exist_ok=True)
    feature_maps_path = os.path.join(feature_maps_root, args.feature_maps_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_to_image = transforms.ToPILImage()

    if dataset == 'CIFAR10':
        num_labels = 10
        testset = torchvision.datasets.CIFAR10(root=data_root_path, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    elif dataset == 'CIFAR100':
        num_labels = 100
        testset = torchvision.datasets.CIFAR100(root=data_root_path, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    elif dataset == 'Tiny':
        num_labels = 200
        data_path = os.path.join(data_root_path, 'tiny-imagenet-200')
        testset_path = os.path.join(data_path, 'val')
        test_csv_path = os.path.join(testset_path, 'labels_test.csv')

        testset = TinyImageNetDataset(csv_file=test_csv_path, root_dir=testset_path, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    if model_type == 'ResNet18':
        model = torchvision.models.resnet18()
        if dataset in ['CIFAR10', 'CIFAR100']:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_labels)

    elif model_type == 'ResNet50':
        model = torchvision.models.resnet50()
        if dataset in ['CIFAR10', 'CIFAR100']:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_labels)

    elif model_type == 'ResNet101':
        model = torchvision.models.resnet101()
        if dataset in ['CIFAR10', 'CIFAR100']:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_labels)
    
    elif model_type == 'MobileNetv2':
        model = torchvision.models.mobilenet_v2(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_labels)

    elif model_type == 'VGG19':
        model = torchvision.models.vgg19(pretrained=False)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_labels)

    elif model_type == 'EfficientNetB2':
        model = torchvision.models.efficientnet_b2(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_labels)

    elif model_type == 'WideResNet28_10':
        model = WideResNet(depth=28, num_classes=num_labels, widen_factor=10)

    else:
        assert False

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to('cuda')
    model.eval()

    targets = None
    gb_model = GuidedBackpropReLUModel(model=model, device='cuda')

    feature_map_dict = {}

    img_count = 0
    orig_test_total = 0
    orig_test_correct = 0
    for step, (images, labels) in enumerate(tqdm(test_loader), 0):
        images, labels = images.to('cuda'), labels.to('cuda')

        gb = gb_model(images, target_category=None)
        gb = deprocess_image(gb)

        if dataset == 'Tiny':
            img_name = 'val_{}.JPEG'.format(img_count)
        elif dataset in ['CIFAR10', 'CIFAR100']:
            img_name = '{}.png'.format(img_count)
        feature_map_dict[img_name] = gb
        img_count += 1

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        orig_test_total += labels.size(0)
        orig_test_correct += (predicted == labels).sum().item()

    test_accuracy = 100.0 * orig_test_correct / orig_test_total
    print('Clean Accuracy', test_accuracy)

    save_feat_maps(feature_map_dict, feature_maps_path)