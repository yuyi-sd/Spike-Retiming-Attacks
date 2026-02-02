import numpy as np
import torch
import argparse
import yaml
from utils.attack import set_PDSG, FGSM, PGD, SDA, PGDTimeShiftAfterEncoder, PGDTimeShiftAfterEncoder_Lowgpu, PGDTimeShiftAfterEncoder_L1, PGDTimeShiftAfterEncoder_L0
from utils.encoder import get_encoder
import torchvision
import os
from timm.data import create_loader
from torchvision import transforms
from utils.utils import DatasetSplitter, DatasetWarpper, DVStransform
import logging
from timm.models import create_model
import models.resnet, models.vgg, models.spikingresformer, models.simplenet
import errno
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    config_parser = argparse.ArgumentParser(description="Attack Config", add_help=False)

    config_parser.add_argument(
        "-c",
        "--config",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(description='Attacking')

    # testing options
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--encoding', default='direct', type=str, help='encoding scheme')
    parser.add_argument('--model', default='spiking_resnet18', help='model name')
    parser.add_argument('--dataset', default='CIFAR10', help='dataset name')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers')

    parser.add_argument('--data-path', default='./datasets')
    parser.add_argument('--output-dir', default='./logs/temp')
    parser.add_argument('--resume', type=str, help='model checkpoint')
    
    # attacking options
    parser.add_argument('--attack', default='FGSM', type=str)
    parser.add_argument('--attack_eps', default=8, type=int)
    parser.add_argument('--store_adv', action='store_true', default=False)
    parser.add_argument('--store_disp', action='store_true', default=False)

    parser.add_argument('--no_PIL', action='store_true', default=False) 
    parser.add_argument('--no_cap', action='store_true', default=False)  
    parser.add_argument('--no_penalty', action='store_true', default=False)

    parser.add_argument('--target_label', default=-1, type=int)
    parser.add_argument('--targeted', action='store_true', default=False)

    parser.add_argument('--defense', type=str, default=None)
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)

    return args

def setup_logger(args):
    output_dir = args.output_dir
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s',
                                  datefmt=r'%Y-%m-%d %H:%M:%S')

    log_path = os.path.join(output_dir, '{}_eps{}_T{}.log'.format(args.attack,args.attack_eps,args.T))
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    return logger

def load_data(
    dataset_dir: str,
    batch_size: int,
    workers: int,
    dataset_type: str,
    T: int,
):
    if dataset_type == 'CIFAR10':
        num_classes = 10
        input_size = (3, 32, 32)
        dataset_test = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=False,
                                                    download=True)
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            num_workers=workers,
            crop_pct=1.0,
            pin_memory=True,
        )
        data_loader_attack = data_loader_test
    elif dataset_type == 'CIFAR100':
        num_classes = 100
        input_size = (3, 32, 32)
        dataset_test = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=False,
                                                     download=True)
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=[n / 255. for n in [129.3, 124.1, 112.4]],
            std=[n / 255. for n in [68.2, 65.4, 70.4]],
            num_workers=workers,
            crop_pct=1.0,
            pin_memory=True,
        )
        data_loader_attack = data_loader_test
    
    elif dataset_type == 'CIFAR10DVS':
        num_classes = 10
        input_size = (2, 128, 128)
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        transform_test = DVStransform(
            transform=transforms.Resize(size=input_size[-2:], antialias=True))
        dataset = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=T, split_by='number')
        dataset_test = DatasetSplitter(dataset, 0.1, False)
        dataset_test = DatasetWarpper(dataset_test, transform_test)
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=int(batch_size*10/T),
                                                       shuffle=True, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
        data_loader_attack = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                       shuffle=False, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
    elif dataset_type == 'DVSGesture':
        num_classes = 11
        input_size = (2, 128, 128)
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        transform_test = DVStransform(
            transform=transforms.Resize(size=input_size[-2:], antialias=True))
        dataset_test = DVS128Gesture(dataset_dir, train=False, data_type='frame', frames_number=T,
                                     split_by='number')
        dataset_test = DatasetWarpper(dataset_test, transform_test)
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                       shuffle=True, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
        data_loader_attack = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                       shuffle=False, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
    elif dataset_type == 'NMNIST':
        num_classes = 10
        input_size = (2, 34, 34)
        from spikingjelly.datasets.n_mnist import NMNIST
        transform_test = DVStransform(
            transform=transforms.Resize(size=input_size[-2:], antialias=True))
        dataset_test = NMNIST(dataset_dir, train=False, data_type='frame', frames_number=T,
                                     split_by='number')
        dataset_test = DatasetWarpper(dataset_test, transform_test)
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                       shuffle=True, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
        data_loader_attack = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                       shuffle=False, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
    elif dataset_type == 'ImageNet':
        num_classes = 1000
        input_size = (3, 224, 224)
        valdir = os.path.join(dataset_dir, 'val')
        dataset_test = torchvision.datasets.ImageFolder(valdir)
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_workers=workers,
            crop_pct=0.95,
            pin_memory=True,
        )
        data_loader_attack = data_loader_test
    else:
        raise ValueError(dataset_type)

    return num_classes, input_size, data_loader_test, data_loader_attack

def main():    
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.defense:
        if 'trades' in args.defense:
            args.resume = args.resume.replace("checkpoints/", "../train_snn_release/outputs/{}/".format(args.defense))
            args.resume = args.resume.replace("VGGSNN","vggsnn")
            args.resume = args.resume.replace("DVSGesture","dvsgesture")
            args.resume = args.resume.replace(".pth","/checkpoint_max_acc1.pth")
            args.output_dir = args.output_dir.replace("logs/","rebuttal/logs/{}/".format(args.defense))
    
    if args.no_PIL:
        args.output_dir = args.output_dir.replace(args.attack, '{}_noPIL'.format(args.attack))
    if args.no_cap:
        args.output_dir = args.output_dir.replace(args.attack, '{}_noCap'.format(args.attack))
    if args.no_penalty:
        args.output_dir = args.output_dir.replace(args.attack, '{}_noPenalty'.format(args.attack))

    if args.target_label >= 0:
        args.output_dir = args.output_dir.replace(args.attack, '{}_Targeted_Label{}'.format(args.attack, args.target_label)) 
    
    if args.targeted:
        args.output_dir = args.output_dir.replace(args.attack, '{}_Targeted'.format(args.attack)) 
    print (args.output_dir)

    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    logger = setup_logger(args)
    logger.info(str(args))
    
    logger.info('------SNN TESTING------')
    dataset_type = args.dataset

    num_classes, input_size, data_loader_test, data_loader_attack = load_data(
        args.data_path, args.batch_size, args.workers, dataset_type, args.T)
     
    net = create_model(
        args.model,
        T=args.T,
        num_classes=num_classes,
        img_size=input_size,
    ).cuda()

    encoder = get_encoder(args.encoding, args.T)
    
    if args.attack == 'FGSM':
        attack_generator = FGSM(net, encoder, args.attack_eps / 255)
    elif args.attack == 'PGD':
        attack_generator = PGD(net, encoder, args.attack_eps / 255)
    elif args.attack == 'SDA':
        attack_generator = SDA(net, batch_limit=args.batch_size)
    elif args.attack == 'PGDTimeShift':
        # attack_generator = PGDTimeShiftAfterEncoder(device = device, model_without_encoder = net, D = args.attack_eps)
        attack_generator = PGDTimeShiftAfterEncoder_Lowgpu(device = device, model_without_encoder = net, D = args.attack_eps)
    elif args.attack == 'PGDTimeShiftL1':
        attack_generator = PGDTimeShiftAfterEncoder_L1(device = device, model_without_encoder = net, l1_steps_budget = args.attack_eps)
    elif args.attack == 'PGDTimeShiftL0':
        attack_generator = PGDTimeShiftAfterEncoder_L0(device = device, model_without_encoder = net, l0_moves_budget = args.attack_eps)
    # else:
    #     raise NotImplementedError(args.attack)
    
    if args.attack == 'FGSM' or args.attack == 'PGD':
        if dataset_type == 'CIFAR10':
            attack_generator.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        elif dataset_type == 'CIFAR100':
            attack_generator.set_normalization_used(mean=[n / 255. for n in [129.3, 124.1, 112.4]], std=[n / 255. for n in [68.2, 65.4, 70.4]])
        elif dataset_type == 'ImageNet':
            attack_generator.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if dataset_type == 'DVSGesture' or dataset_type == 'CIFAR10DVS' or dataset_type == 'NMNIST':
            attack_generator.max_val = None
    
    if args.T != 10:
        args.resume = args.resume.replace(".pt", "_T{}.pt".format(args.T))
    state = torch.load(args.resume, map_location=device)
    try:
        net.load_state_dict(state["model"])
    except:
        net.load_state_dict(state)

    # adopt PDSG surrogate function        
    net = set_PDSG(net)
    
    torch.cuda.empty_cache()
    net.eval()
    
    if 'PGDTimeShift' in args.attack:
        total = 0
        correct = 0
        
        # -------- 1) 原始准确率（保留你原来的 mean(0) 逻辑） --------
        for batch_idx, (images, labels) in enumerate(data_loader_test):
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                encoded_images = encoder(images.detach())     # [T,B,C,H,W]
                outputs = net(encoded_images).mean(0)         # [B,C]
                _, predicted = outputs.max(1)
                
                total += float(labels.size(0))
                correct += float(predicted.eq(labels).sum().item())
        logger.info(f'Accuracy: {100.0 * correct / total:.2f}')
        
        # -------- 2) 攻击部分（去掉 L0，改成 Fooling Rate） --------
        total_attacked = 0
        total_success  = 0
        if args.store_adv:
            adv_images = torch.tensor([])
            labels_tensor = torch.tensor([])
        if args.store_disp:
            adv_deltas = torch.tensor([])

        for batch_idx, (images, labels) in enumerate(data_loader_attack):
            images, labels = images.to(device), labels.to(device)
            
            # batch_size=1 in DVS attack
            for b in range(images.shape[0]):
                image, label = images[b].unsqueeze(0), labels[b].unsqueeze(0)  # [1,T,C,H,W], [1]
                image_enc = encoder(image)                                     # [T,1,C,H,W]
                
                # 只攻击干净样本分类正确的
                with torch.no_grad():
                    out_clean = net(image_enc).mean(0)                         # [1,C]
                    pred_clean = out_clean.argmax(1)
                if pred_clean.item() != label.item():
                    # if args.store_adv:
                    #     adv_images = torch.cat([adv_images, image_enc.detach().clone().cpu()], dim = 1)
                    # if args.store_disp:
                    #     adv_deltas = torch.cat([adv_deltas, torch.zeros_like(image_enc).detach().clone().cpu()], dim = 1)
                    continue

                if args.targeted:
                    target_label = np.random.randint(num_classes)
                else:
                    target_label = args.target_label

                if pred_clean.item() == target_label:
                    continue

                total_attacked += 1

                # 生成对抗样本（after-encoder：输入/输出都是 [T,B,C,H,W]）
                if args.store_disp:
                    adv_image_enc, delta_tbchw = attack_generator(image_enc, label, args.store_disp, use_PIL = not args.no_PIL, use_cap = not args.no_cap, use_penalty = not args.no_penalty, target_label = args.target_label)             # [T,1,C,H,W]
                    print("total shfted:", (delta_tbchw!=0).sum().item())
                    adv_deltas = torch.cat([adv_deltas, delta_tbchw.detach().clone().cpu()], dim = 1)
                else:
                    adv_image_enc = attack_generator(image_enc, label, use_PIL = not args.no_PIL, use_cap = not args.no_cap, use_penalty = not args.no_penalty, target_label = args.target_label)
                if args.store_adv:
                    labels_tensor = torch.cat([labels_tensor, label.cpu()], dim = 0)
                    adv_images = torch.cat([adv_images, adv_image_enc.detach().clone().cpu()], dim = 1)
                
                print("clean spikes:", image_enc.sum().item())
                print("adv spikes:",   adv_image_enc.sum().item())
                print("same ratio:",   (image_enc == adv_image_enc).float().mean().item())
                # print("loose ratio:",   attack_generator.measure_bound_looseness(image_enc, adv_image_enc, args.attack_eps))

                # 评估是否成功
                with torch.no_grad():
                    out_adv = net(adv_image_enc).mean(0)                        # [1,C]
                    pred_adv = out_adv.argmax(1)

                if target_label < 0:
                    ce = torch.nn.CrossEntropyLoss()(out_adv, label)
                    if pred_adv.item() != label.item():
                        total_success += 1
                else:
                    ce = torch.nn.CrossEntropyLoss()(out_adv, target_label * torch.ones_like(label))
                    if pred_adv.item() == target_label:
                        total_success += 1
                print (ce)

            # —— 批内日志 —— #
            fr = 100.0 * total_success / (total_attacked if total_attacked > 0 else 1)
            logger.info(
                f'Batch:[{batch_idx+1}/{len(data_loader_attack)}], '
                f'FoolingRate: {fr:.2f}%'
            )

            # # 跟你原逻辑一致：攻击到 100 个样本就停
            if dataset_type == 'CIFAR10DVS' and total_attacked >= 100:
                break
            if dataset_type == 'NMNIST' and total_attacked >= 1000:
                break
        if args.store_adv:
            torch.save(labels_tensor, os.path.join(args.output_dir, 'Labels.pt'))
            torch.save(adv_images, os.path.join(args.output_dir, 'AdvImages_{}_eps{}_T{}.pt'.format(args.attack,args.attack_eps,args.T)))
        if args.store_disp:
            torch.save(adv_deltas, os.path.join(args.output_dir, 'AdvDeltas_{}_eps{}_T{}.pt'.format(args.attack,args.attack_eps,args.T)))

    elif args.attack == 'Clean':
        total = 0
        correct = 0
        
        # -------- 1) 原始准确率（保留你原来的 mean(0) 逻辑） --------
        for batch_idx, (images, labels) in enumerate(data_loader_test):
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                encoded_images = encoder(images.detach())     # [T,B,C,H,W]
                outputs = net(encoded_images).mean(0)         # [B,C]
                _, predicted = outputs.max(1)
                
                total += float(labels.size(0))
                correct += float(predicted.eq(labels).sum().item())
        logger.info(f'Accuracy: {100.0 * correct / total:.2f}')
        
        # -------- 2) 攻击部分（去掉 L0，改成 Fooling Rate） --------
        images_tensor = torch.tensor([])
        labels_tensor = torch.tensor([])

        for batch_idx, (images, labels) in enumerate(data_loader_attack):
            images, labels = images.to(device), labels.to(device)
            
            # batch_size=1 in DVS attack
            for b in range(images.shape[0]):
                image, label = images[b].unsqueeze(0), labels[b].unsqueeze(0)  # [1,T,C,H,W], [1]
                image_enc = encoder(image)                                     # [T,1,C,H,W]
                
                # 只攻击干净样本分类正确的
                with torch.no_grad():
                    out_clean = net(image_enc).mean(0)                         # [1,C]
                    pred_clean = out_clean.argmax(1)
                if pred_clean.item() != label.item():
                    continue

                if pred_clean.item() == target_label:
                    continue

                labels_tensor = torch.cat([labels_tensor, label.cpu()], dim = 0)
                images_tensor = torch.cat([images_tensor, image_enc.detach().clone().cpu()], dim = 1)

            if dataset_type == 'CIFAR10DVS' and labels_tensor.shape[0] >= 100:
                break
            if dataset_type == 'NMNIST' and labels_tensor.shape[0] >= 1000:
                break

        torch.save(labels_tensor, os.path.join(args.output_dir, 'Labels.pt'))
        torch.save(images_tensor, os.path.join(args.output_dir, 'Images.pt'))

    elif args.encoding != 'binary': # attacking static images or dynamic integer frames
        correct = 0
        adversarial_correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(data_loader_test):
            images, labels = images.to(device), labels.to(device)
            
            # original accuracy
            with torch.no_grad():
                encoded_images = encoder(images.detach())
                outputs = net(encoded_images).mean(0)
                _, predicted = outputs.max(1)
                total += float(labels.size(0))
                correct += float(predicted.eq(labels).sum().item())

            # perform attack
            adversarial_images = attack_generator(images, labels)

            # adversarial accuracy
            with torch.no_grad():
                encoded_images = encoder(adversarial_images)
                outputs = net(encoded_images).mean(0)
                _, predicted = outputs.max(1)
                adversarial_correct += float(predicted.eq(labels).sum().item())
                
            accuracy = 100.0 * correct / total
            adversarial_accuracy = 100.0 * adversarial_correct / total
            ASR = 100.0 * (accuracy - adversarial_accuracy) / accuracy
            
            logger.info(f'Batch:[{batch_idx+1}/{len(data_loader_test)}], Accuracy: {accuracy:.2f}%, Adversarial Accuracy: {adversarial_accuracy:.2f}%, Attack Success Rate: {ASR:.2f}%')
    elif args.attack == 'SDA':
        total = 0
        correct = 0
        L0_list = []
        
        # original accuracy
        for batch_idx, (images, labels) in enumerate(data_loader_test):
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                # print (images.shape)
                encoded_images = encoder(images.detach())
                # print (encoded_images.shape)
                outputs = net(encoded_images).mean(0)
                # print (outputs.shape)
                _, predicted = outputs.max(1)
                
                total += float(labels.size(0))
                correct += float(predicted.eq(labels).sum().item())
        logger.info(f'Accuracy: {100.0 * correct / total}')

        for batch_idx, (images, labels) in enumerate(data_loader_attack):
            images, labels = images.to(device), labels.to(device)
            
            # batch_size=1 in DVS attack
            for b in range(images.shape[0]):
                image, label = images[b].unsqueeze(0), labels[b].unsqueeze(0)
                image = encoder(image)
                with torch.no_grad():
                    output = net(image).mean(0)
                    _, predicted = output.max(1)
                    
                # only attack correctly classified inputs
                if predicted.eq(label).sum().item() == 0:
                    continue
                else:
                    adversarial_image = attack_generator(image, label)
                    L0 = image.not_equal(adversarial_image).sum().cpu().item()
                    L0_list.append(L0)
                    
            L0_array = np.array(L0_list)
            success = L0_array[L0_array > 0].size # if L0=0, attack failed
            L0_200 = 100.0 * L0_array[(L0_array > 0) & (L0_array < 200)].size / (success if success > 0 else 1)
            L0_800 = 100.0 * L0_array[(L0_array > 0) & (L0_array < 800)].size / (success if success > 0 else 1)
            L0_mean = np.mean(L0_array[L0_array > 0])
            L0_median = np.median(L0_array[L0_array > 0])
            logger.info(f'Batch:[{batch_idx+1}/{len(data_loader_attack)}], L0_200: {L0_200:.2f}%, L0_800: {L0_800:.2f}%, L0_mean: {L0_mean:.2f}, L0_median: {L0_median:.2f}')
            if len(L0_array) >= 100: # attack random 100 inputs
                break

        
                    
    logger.info('------SNN TESTING FINISHED------')
if __name__ == '__main__':
    main()
