# integer grid
## infinite
CUDA_VISIBLE_DEVICES=2 python test.py -c ./configs/PGDTimeShift_vggsnn_dvsgesture.yaml --attack_eps 1 --T 10
CUDA_VISIBLE_DEVICES=2 python test.py -c ./configs/PGDTimeShift_vggsnn_dvsgesture.yaml --attack_eps 2 --T 10
CUDA_VISIBLE_DEVICES=3 python test.py -c ./configs/PGDTimeShift_vggsnn_dvsgesture.yaml --attack_eps 3 --T 10


CUDA_VISIBLE_DEVICES=1 python test.py -c ./configs/PGDTimeShift_resnet18_cifar10dvs.yaml --attack_eps 1 --T 10
CUDA_VISIBLE_DEVICES=0 python test.py -c ./configs/PGDTimeShift_resnet18_cifar10dvs.yaml --attack_eps 2 --T 10
CUDA_VISIBLE_DEVICES=1 python test.py -c ./configs/PGDTimeShift_resnet18_cifar10dvs.yaml --attack_eps 3 --T 10

## L1
CUDA_VISIBLE_DEVICES=2 python test.py -c ./configs/PGDTimeShiftL1_vggsnn_dvsgesture.yaml --attack_eps 2000 --T 10
CUDA_VISIBLE_DEVICES=2 python test.py -c ./configs/PGDTimeShiftL1_vggsnn_dvsgesture.yaml --attack_eps 4000 --T 10
CUDA_VISIBLE_DEVICES=3 python test.py -c ./configs/PGDTimeShiftL1_vggsnn_dvsgesture.yaml --attack_eps 8000 --T 10
CUDA_VISIBLE_DEVICES=3 python test.py -c ./configs/PGDTimeShiftL1_vggsnn_dvsgesture.yaml --attack_eps 16000 --T 10


CUDA_VISIBLE_DEVICES=1 python test.py -c ./configs/PGDTimeShiftL1_resnet18_cifar10dvs.yaml --attack_eps 2000 --T 10
CUDA_VISIBLE_DEVICES=0 python test.py -c ./configs/PGDTimeShiftL1_resnet18_cifar10dvs.yaml --attack_eps 4000 --T 10
CUDA_VISIBLE_DEVICES=1 python test.py -c ./configs/PGDTimeShiftL1_resnet18_cifar10dvs.yaml --attack_eps 8000 --T 10
CUDA_VISIBLE_DEVICES=1 python test.py -c ./configs/PGDTimeShiftL1_resnet18_cifar10dvs.yaml --attack_eps 16000 --T 10

## L0
CUDA_VISIBLE_DEVICES=2 python test.py -c ./configs/PGDTimeShiftL0_vggsnn_dvsgesture.yaml --attack_eps 1000 --T 10
CUDA_VISIBLE_DEVICES=2 python test.py -c ./configs/PGDTimeShiftL0_vggsnn_dvsgesture.yaml --attack_eps 2000 --T 10
CUDA_VISIBLE_DEVICES=3 python test.py -c ./configs/PGDTimeShiftL0_vggsnn_dvsgesture.yaml --attack_eps 4000 --T 10
CUDA_VISIBLE_DEVICES=3 python test.py -c ./configs/PGDTimeShiftL0_vggsnn_dvsgesture.yaml --attack_eps 8000 --T 10


CUDA_VISIBLE_DEVICES=1 python test.py -c ./configs/PGDTimeShiftL0_resnet18_cifar10dvs.yaml --attack_eps 1000 --T 10
CUDA_VISIBLE_DEVICES=0 python test.py -c ./configs/PGDTimeShiftL0_resnet18_cifar10dvs.yaml --attack_eps 2000 --T 10
CUDA_VISIBLE_DEVICES=1 python test.py -c ./configs/PGDTimeShiftL0_resnet18_cifar10dvs.yaml --attack_eps 4000 --T 10
CUDA_VISIBLE_DEVICES=1 python test.py -c ./configs/PGDTimeShiftL0_resnet18_cifar10dvs.yaml --attack_eps 8000 --T 10


# binary grid
## infinite
CUDA_VISIBLE_DEVICES=2 python test.py -c ./configs/PGDTimeShift_vggsnn_dvsgesture_binary.yaml --attack_eps 1 --T 10
CUDA_VISIBLE_DEVICES=2 python test.py -c ./configs/PGDTimeShift_vggsnn_dvsgesture_binary.yaml --attack_eps 2 --T 10
CUDA_VISIBLE_DEVICES=3 python test.py -c ./configs/PGDTimeShift_vggsnn_dvsgesture_binary.yaml --attack_eps 3 --T 10


CUDA_VISIBLE_DEVICES=1 python test.py -c ./configs/PGDTimeShift_resnet18_cifar10dvs_binary.yaml --attack_eps 1 --T 10
CUDA_VISIBLE_DEVICES=0 python test.py -c ./configs/PGDTimeShift_resnet18_cifar10dvs_binary.yaml --attack_eps 2 --T 10
CUDA_VISIBLE_DEVICES=1 python test.py -c ./configs/PGDTimeShift_resnet18_cifar10dvs_binary.yaml --attack_eps 3 --T 10

## L1
CUDA_VISIBLE_DEVICES=2 python test.py -c ./configs/PGDTimeShiftL1_vggsnn_dvsgesture_binary.yaml --attack_eps 2000 --T 10
CUDA_VISIBLE_DEVICES=2 python test.py -c ./configs/PGDTimeShiftL1_vggsnn_dvsgesture_binary.yaml --attack_eps 4000 --T 10
CUDA_VISIBLE_DEVICES=3 python test.py -c ./configs/PGDTimeShiftL1_vggsnn_dvsgesture_binary.yaml --attack_eps 8000 --T 10


CUDA_VISIBLE_DEVICES=1 python test.py -c ./configs/PGDTimeShiftL1_resnet18_cifar10dvs_binary.yaml --attack_eps 2000 --T 10
CUDA_VISIBLE_DEVICES=0 python test.py -c ./configs/PGDTimeShiftL1_resnet18_cifar10dvs_binary.yaml --attack_eps 4000 --T 10
CUDA_VISIBLE_DEVICES=1 python test.py -c ./configs/PGDTimeShiftL1_resnet18_cifar10dvs_binary.yaml --attack_eps 8000 --T 10

## L0
CUDA_VISIBLE_DEVICES=2 python test.py -c ./configs/PGDTimeShiftL0_vggsnn_dvsgesture_binary.yaml --attack_eps 1000 --T 10
CUDA_VISIBLE_DEVICES=2 python test.py -c ./configs/PGDTimeShiftL0_vggsnn_dvsgesture_binary.yaml --attack_eps 2000 --T 10
CUDA_VISIBLE_DEVICES=3 python test.py -c ./configs/PGDTimeShiftL0_vggsnn_dvsgesture_binary.yaml --attack_eps 4000 --T 10


CUDA_VISIBLE_DEVICES=1 python test.py -c ./configs/PGDTimeShiftL0_resnet18_cifar10dvs_binary.yaml --attack_eps 1000 --T 10
CUDA_VISIBLE_DEVICES=0 python test.py -c ./configs/PGDTimeShiftL0_resnet18_cifar10dvs_binary.yaml --attack_eps 2000 --T 10
CUDA_VISIBLE_DEVICES=1 python test.py -c ./configs/PGDTimeShiftL0_resnet18_cifar10dvs_binary.yaml --attack_eps 4000 --T 10

