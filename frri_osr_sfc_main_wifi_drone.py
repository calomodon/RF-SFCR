import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
import re
import pandas as pd
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import openood.utils.comm as comm
from openood.utils import config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators.metrics import compute_all_metrics
from scipy.special import softmax

os.environ['PYTHONHASHSEED'] = str(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_acc = 0
best_epoch_idx = 0
begin_time = time.time()
#[1,256,2]
class Config_SFCR:
    def __init__(self):
        self.num_classes = 16
        self.model_num_classes = self.num_classes + 1
        self.feature_dim = 256
        self.epochs = 100
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.loss_weight = 0.001
        self.classification_threshold = 0.9
        self.data_shape = (1, 2, 512)
        self.save_dir = "results/frri_osr_sfc_drone"
        self.device = device
        self.config_files = [
            './configs/datasets/drone/drone_benchmark.yml',
            './configs/datasets/drone/drone_ood_benchmark.yml',
            './configs/networks/resnet18_32x32.yml',
            './configs/pipelines/test/test_ood.yml',
            './configs/preprocessors/base_preprocessor.yml',
            './configs/postprocessors/msp.yml',
        ]


class SE_Block(nn.Module):
    def __init__(self, c, r=16, mid=0):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        if mid == 0:
            mid = c // r
        self.excitation = nn.Sequential(
            nn.Linear(c, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride=1, groups=1, fusion=['se'], height=1):
        super(BasicBlock, self).__init__()
        self.fusion = fusion
        self.groups = groups
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=(3, height), padding=(1, 0), stride=stride, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=(3, 1), padding=(1, 0), groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        
        self.relu = nn.LeakyReLU(inplace=True)
        if 'se' in fusion:
            self.se = SE_Block(out_planes)
            self.usese = True
        else:
            self.usese = False
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            ds_group = 1 if ('rc' in fusion) else groups
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=ds_group),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.usese:
            out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class GRNet(nn.Module):
    def __init__(self, data_shape=(1, 16384, 2), expand=[2*4, 2*4, 4*4, 8*4, 16*4], num_classes=100, fusion=[], loss_type="cel"):
        super(GRNet, self).__init__()
        self.in_planes = data_shape[0]
        self.usefpn = False
        self.groups = data_shape[0]
        self.fusion = fusion
        self.usese = False
        self.quad = False
        self.num_classes = num_classes
        num_blocks = 2
        if 'se' in self.fusion:
            self.usese = True
        if loss_type not in ["cel"]:
            raise ValueError("__init__() got unknown loss type")
        self.loss_type = loss_type
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.Conv2d(self.in_planes, self.in_planes*expand[0], kernel_size=(7, 2), stride=1, padding=0, bias=False, groups=self.groups),
            nn.BatchNorm2d(self.in_planes*expand[0]),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=2, padding=0)
        ))
        now_channels = self.in_planes * expand[0]
        strides_layers = [1, 1, 2, 2]
        for i, mult in enumerate(expand[1:]):
            out_channels = self.in_planes * mult
            strides_blocks = [strides_layers[i]] + [1] * (num_blocks - 1)
            for j in range(num_blocks):
                self.layers.append(BasicBlock(
                    now_channels,
                    out_channels,
                    groups=self.groups,
                    stride=strides_blocks[j],
                    fusion=fusion,
                ))
                now_channels = out_channels
        
        self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(now_channels, self.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
    
    def get_loss(self, pred, y=None, reduction='mean'):
        if self.loss_type == "cel":
            loss = F.cross_entropy(pred, y, reduction=reduction)
        return loss

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class SFCloss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2):
        super(SFCloss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.close_numclasses = self.num_classes - 1
        self.close_centers = nn.Parameter(torch.randn(self.close_numclasses, self.feat_dim))
        self.open_centers = nn.Parameter(torch.randn(1, self.feat_dim))
        self.out_weight = nn.Parameter(torch.zeros((1)))
    
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size, 2).
        """
        labels_class = labels[:, 0]  # (0,N-1)
        # labels_sim 在原始代码中定义但未使用，保留以保持一致性
        # labels_sim = labels[:, 1] * int(self.num_classes)  # (0 or N)
        batch_size = x.size(0)
        
        distmat_close = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.close_numclasses) + \
                  torch.pow(self.close_centers, 2).sum(dim=1, keepdim=True).expand(self.close_numclasses, batch_size).t()
        
        distmat_close.addmm_(x, self.close_centers.t(), beta=1, alpha=-2)
        out_weight = torch.sigmoid(self.out_weight)
        mid_centers = (1 - out_weight) * self.close_centers + out_weight * self.open_centers

        distmat_open = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, 1) + \
                  torch.pow(mid_centers, 2).sum(dim=1, keepdim=True).expand(self.close_numclasses, batch_size).t()
        distmat_open.addmm_(x, mid_centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.close_numclasses, dtype=torch.int64, device=x.device)
        labels_class = labels_class.unsqueeze(1).expand(batch_size, self.close_numclasses)
        mask = labels_class.eq(classes.expand(batch_size, self.close_numclasses)).float()
        
        dist_close = (distmat_close * mask.float()).clamp(min=1e-12, max=1e+12).sum(1)
        dist_open = (distmat_open * mask.float()).clamp(min=1e-12, max=1e+12).sum(1)
        use_syn_mask = (labels[:, 1] == 1)
        dist = dist_close * (~use_syn_mask) + use_syn_mask * (dist_open)
        loss = dist.sum() / batch_size
        return loss

class SFCR(GRNet):
    def __init__(self, data_shape=(1, 16384, 2), expand=[2*4, 2*4, 4*4, 8*4, 16*4], num_classes=100, fusion=[], loss_weight=0.001):
        super(SFCR, self).__init__(data_shape=data_shape, expand=expand, num_classes=num_classes, fusion=fusion)
        feature_size = data_shape[0] * expand[-1]
        self.feat_dim = 256
        self.fc = nn.Linear(feature_size, self.feat_dim)
        self.fc1 = nn.Linear(self.feat_dim, self.num_classes)
        self.num_classes = num_classes
        self.loss = SFCloss(num_classes=num_classes, feat_dim=self.feat_dim)
        self.loss_weight = loss_weight

    def get_loss(self, out, y=None, reduction='mean'):
        (pred, feature) = out
        labels_class = y[:, 0]
        labels_syn = y[:, 1]
        label_open = labels_class.clone()
        label_open[labels_syn == 1] = self.num_classes - 1
        loss_center = self.loss(feature, y)
        loss_cel = F.cross_entropy(pred, label_open)
        return loss_cel + self.loss_weight * loss_center

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        feature = torch.flatten(x, 1)
        latent = self.fc(feature)
        pred = self.fc1(latent)
        return pred, latent

class SFCRModel(nn.Module):
    def __init__(self, config_sfcr):
        super(SFCRModel, self).__init__()
        expand = [2*4, 2*4, 4*4, 8*4, 16*4]
        self.sfcr = SFCR(
            data_shape=config_sfcr.data_shape,
            expand=expand,
            num_classes=config_sfcr.model_num_classes,
            fusion=[],
            loss_weight=config_sfcr.loss_weight
        )
        
    def forward(self, x):
        pred, latent = self.sfcr(x)
        return pred, latent
    
    def get_loss(self, out, y):
        return self.sfcr.get_loss(out, y)

class TrainingManager:
    def __init__(self, model, config_sfcr):
        self.model = model.to(device)
        self.config_sfcr = config_sfcr
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config_sfcr.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config_sfcr.epochs
        )
    
    def train_phase1(self, train_loader, val_loader):
        global best_acc, best_epoch_idx
        best_acc = 0
        best_epoch_idx = 0

        for epoch in range(self.config_sfcr.epochs):
            train_metrics = self.train_epoch(train_loader, epoch + 1)
            val_metrics = eval_acc(self.model, val_loader, epoch + 1)
            save_model(self.model, val_metrics, self.config_sfcr.epochs, self.config_sfcr.save_dir)
            report(train_metrics, val_metrics)

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        print(f"Number of batches in epoch {epoch}: {len(loader)}")
        
        for batch in tqdm(loader, desc=f'Epoch {epoch}'):
            x = batch['data'].to(device)
            y = batch['label'].to(device)
            
            y_sfcr = torch.stack([y, torch.zeros_like(y)], dim=1).long()
            
            self.optimizer.zero_grad()
            logits, features = self.model(x)
            out = (logits, features)
            loss = self.model.get_loss(out, y_sfcr)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            _, pred = torch.max(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        metrics = {
            'epoch_idx': epoch,
            'loss': total_loss / len(loader),
            'acc': correct / total
        }
        return metrics

def eval_acc(model, data_loader, epoch_idx):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Eval: '):
            x = batch['data'].to(device)
            y = batch['label'].to(device)
            y_sfcr = torch.stack([y, torch.zeros_like(y)], dim=1).long()
            
            logits, features = model(x)
            out = (logits, features)
            loss = model.get_loss(out, y_sfcr)
            _, pred = torch.max(logits, dim=1)
            
            total_correct += (pred == y).sum().item()
            total_samples += y.size(0)
            total_loss += loss.item()
             
    metrics = {}
    metrics['epoch_idx'] = epoch_idx
    metrics['loss'] = save_metrics(total_loss / len(data_loader))
    metrics['acc'] = save_metrics(total_correct / total_samples)
    return metrics

def save_metrics(value):
    all_values = comm.gather(value)
    return sum(all_values)

def save_model(net, val_metrics, num_epochs, save_dir):
    global best_acc, best_epoch_idx
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        state_dict = net.module.state_dict()
    except AttributeError:
        state_dict = net.state_dict()

    if val_metrics['acc'] >= best_acc:
        old_fname = f'best_epoch{best_epoch_idx}_acc{best_acc:.4f}.ckpt'
        old_pth = os.path.join(save_dir, old_fname)
        Path(old_pth).unlink(missing_ok=True)

        best_epoch_idx = val_metrics['epoch_idx']
        best_acc = val_metrics['acc']

        torch.save(state_dict, os.path.join(save_dir, 'best.ckpt'))
        save_fname = f'best_epoch{best_epoch_idx}_acc{best_acc:.4f}.ckpt'
        torch.save(state_dict, os.path.join(save_dir, save_fname))

    if val_metrics['epoch_idx'] == num_epochs:
        save_fname = f'last_epoch{val_metrics["epoch_idx"]}_acc{val_metrics["acc"]:.4f}.ckpt'
        torch.save(state_dict, os.path.join(save_dir, save_fname))

def report(train_metrics, val_metrics):
    print('\n  Epoch {:03d} | Time {:5d}s | Train Loss {:.4f} | '
          'Test Loss {:.3f} | Test Acc {:.2f}'.format(
              train_metrics['epoch_idx'],
              int(time.time() - begin_time),
              train_metrics['loss'],
              val_metrics['loss'],
              100.0 * val_metrics['acc'],
            ),
          flush=True)

def load_model(model, checkpoint_path):
    """加载模型检查点"""
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

def extract_snr_from_path(line):
    match = re.search(r'/([^/]+)/signal_', line)
    if match:
        return match.group(1)
    return None

def filter_dataset_by_snr(dataset, target_snr):
    filtered_indices = []
    target_snr_str = str(target_snr)
    
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        imglist = base_dataset.imglist
        for subset_idx in range(len(dataset)):
            original_idx = dataset.indices[subset_idx]
            line = imglist[original_idx].strip('\n')
            snr = extract_snr_from_path(line)
            if snr == target_snr_str:
                filtered_indices.append(subset_idx)
    else:
        for idx in range(len(dataset)):
            line = dataset.imglist[idx].strip('\n')
            snr = extract_snr_from_path(line)
            if snr == target_snr_str:
                filtered_indices.append(idx)
    
    return filtered_indices

def create_filtered_dataloader(original_loader, target_snr):
    dataset = original_loader.dataset
    filtered_indices = filter_dataset_by_snr(dataset, target_snr)
    
    if len(filtered_indices) == 0:
        print(f"警告: SNR={target_snr} 的数据为空，跳过该 SNR")
        return None
    
    filtered_dataset = Subset(dataset, filtered_indices)
    batch_size = getattr(original_loader, 'batch_size', 128)
    num_workers = getattr(original_loader, 'num_workers', 0)
    pin_memory = getattr(original_loader, 'pin_memory', False)
    
    filtered_loader = DataLoader(
        filtered_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return filtered_loader

def test_ood(model, id_loader, ood_loader, config_sfcr):
    model.eval()
    class_num_kkc = config_sfcr.num_classes
    scores_N = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(id_loader, desc='ID Inference'):
            x = batch['data'].to(device)
            y = batch['label'].to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            out_cal = out[:, :class_num_kkc]
            scores_N.append(out_cal)
            labels.append(y)
    
    with torch.no_grad():
        for batch in tqdm(ood_loader, desc='OOD Inference'):
            x = batch['data'].to(device)
            y = batch['label'].to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            out_cal = out[:, :class_num_kkc]
            scores_N.append(out_cal)
            labels.append(y)
    
    scores_N = torch.cat(scores_N, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    
    closeset_indices = labels < class_num_kkc
    labels[labels >= class_num_kkc] = class_num_kkc
    
    pred_softmax_N, pred_softmax_threshold_N = [], []
    confidences = []
    
    for score in scores_N:
        ss = softmax(np.array(score.ravel()))
        pred_softmax_N.append(ss.argmax())
        pred_softmax_threshold_N.append(ss.argmax() if np.max(ss) >= config_sfcr.classification_threshold else class_num_kkc)
        sst = np.concatenate((ss, [ss.sum() - ss.max()])) / (ss.sum() * 2 - ss.max())
        confidences.append(1.0 - sst[-1])
    
    pred_softmax_N = np.array(pred_softmax_N)
    pred_softmax_threshold_N = np.array(pred_softmax_threshold_N)
    confidences = np.array(confidences)
    
    id_mask = closeset_indices
    ood_mask = ~closeset_indices
    
    id_pred = pred_softmax_N[id_mask]
    id_conf = confidences[id_mask]
    id_label = labels[id_mask]
    
    ood_pred = pred_softmax_threshold_N[ood_mask]
    ood_conf = confidences[ood_mask]
    ood_label = -1 * np.ones_like(ood_pred)
    
    pred = np.concatenate([id_pred, ood_pred])
    conf = np.concatenate([id_conf, ood_conf])
    label = np.concatenate([id_label, ood_label])
    
    metrics = compute_all_metrics(conf, label, pred)
    fpr, auroc, aupr_in, aupr_out, accuracy = metrics
    tnr = 1 - fpr
    
    results = {
        'TNR': tnr * 100,
        'AUROC': auroc * 100,
        'AUPR_IN': aupr_in * 100,
        'AUPR_OUT': aupr_out * 100,
        'ACC': accuracy * 100
    }
    
    print('\n' + '='*70)
    print('Open Set Recognition Test Results:')
    print('='*70)
    print(f'TNR:      {results["TNR"]:.2f}%')
    print(f'AUROC:    {results["AUROC"]:.2f}%')
    print(f'AUPR_IN:  {results["AUPR_IN"]:.2f}%')
    print(f'AUPR_OUT: {results["AUPR_OUT"]:.2f}%')
    print(f'ACC:      {results["ACC"]:.2f}%')
    print('='*70 + '\n')
    
    # 返回结果和评分数据
    return results, {'pred': pred, 'conf': conf, 'label': label}

def test_ood_by_snr(model, test_loader, test_ood_loader, config_sfcr, model_path, 
                    snr_range=None, save_results=True):
    # 注释掉按SNR循环测试的代码
    # if snr_range is None:
    #     snr_range = list(range(-6, 16, 2))
    #     snr_range = [str(snr) for snr in snr_range]
    # else:
    #     snr_range = [str(snr) for snr in snr_range]
    
    all_results = {}
    
    # 注释掉按SNR循环测试部分
    # print('\n' + '='*70)
    # print('开始按 SNR 循环测试开集识别表现')
    # print(f'SNR 范围: {snr_range}')
    # print('='*70 + '\n')
    
    # for snr in snr_range:
    #     print(f'\n{"="*70}')
    #     print(f'正在测试 SNR = {snr}')
    #     print(f'{"="*70}\n')
    #     
    #     filtered_test_loader = create_filtered_dataloader(test_loader, snr)
    #     if filtered_test_loader is None:
    #         print(f'跳过 SNR={snr}（无数据）\n')
    #         continue
    #     
    #     filtered_ood_loader = create_filtered_dataloader(test_ood_loader, snr)
    #     if filtered_ood_loader is None:
    #         print(f'跳过 SNR={snr}（OOD 数据为空）\n')
    #         continue
    #     
    #     current_model = SFCRModel(config_sfcr)
    #     current_model = load_model(current_model, model_path).cuda()
    #     current_model.eval()
    #     
    #     try:
    #         results, score_data = test_ood(current_model, filtered_test_loader, filtered_ood_loader, config_sfcr)
    #         all_results[snr] = results
    #         
    #         # 保存该SNR的评分数据到npz文件
    #         if save_results:
    #             npz_dir = os.path.join(config_sfcr.save_dir, 'scores')
    #             os.makedirs(npz_dir, exist_ok=True)
    #             npz_path = os.path.join(npz_dir, f'snr_{snr}.npz')
    #             np.savez(npz_path, 
    #                     pred=score_data['pred'].astype(np.int64),
    #                     conf=score_data['conf'].astype(np.float64),
    #                     label=score_data['label'].astype(np.int64))
    #             print(f'评分数据已保存到: {npz_path}')
    #         
    #         print(f'\nSNR={snr} 测试完成:')
    #         print(f'  TNR:      {results["TNR"]:>7.2f}%')
    #         print(f'  AUROC:    {results["AUROC"]:>7.2f}%')
    #         print(f'  AUPR_IN:  {results["AUPR_IN"]:>7.2f}%')
    #         print(f'  AUPR_OUT: {results["AUPR_OUT"]:>7.2f}%')
    #         print(f'  ACC:      {results["ACC"]:>7.2f}%')
    #     except Exception as e:
    #         print(f'SNR={snr} 测试出错: {str(e)}')
    #         import traceback
    #         traceback.print_exc()
    #         continue
    
    print('\n' + '='*70)
    print('测试全SNR（所有SNR数据）')
    print('='*70 + '\n')
    
    current_model = SFCRModel(config_sfcr)
    current_model = load_model(current_model, model_path).cuda()
    current_model.eval()
    
    try:
        results_all, score_data_all = test_ood(current_model, test_loader, test_ood_loader, config_sfcr)
        all_results['All'] = results_all
        
        # 保存全SNR的评分数据到npz文件
        if save_results:
            npz_dir = os.path.join(config_sfcr.save_dir, 'scores')
            os.makedirs(npz_dir, exist_ok=True)
            npz_path = os.path.join(npz_dir, 'sfcr_rml201610a.npz')
            np.savez(npz_path,
                    pred=score_data_all['pred'].astype(np.int64),
                    conf=score_data_all['conf'].astype(np.float64),
                    label=score_data_all['label'].astype(np.int64))
            print(f'全SNR评分数据已保存到: {npz_path}')
        
        print(f'\n全SNR测试完成:')
        print(f'  TNR:      {results_all["TNR"]:>7.2f}%')
        print(f'  AUROC:    {results_all["AUROC"]:>7.2f}%')
        print(f'  AUPR_IN:  {results_all["AUPR_IN"]:>7.2f}%')
        print(f'  AUPR_OUT: {results_all["AUPR_OUT"]:>7.2f}%')
        print(f'  ACC:      {results_all["ACC"]:>7.2f}%')
    except Exception as e:
        print(f'全SNR测试出错: {str(e)}')
        import traceback
        traceback.print_exc()
    
    # 注释掉结果汇总部分
    # print('\n' + '='*70)
    # print('所有 SNR 测试结果汇总')
    # print('='*70)
    # header = f'{"SNR":>6} | {"TNR":>8} | {"AUROC":>8} | {"AUPR_IN":>10} | {"AUPR_OUT":>11} | {"ACC":>8}'
    # print(header)
    # print('-'*70)
    # 
    # for snr in snr_range:
    #     if snr in all_results:
    #         r = all_results[snr]
    #         print(f'{snr:>6} | {r["TNR"]:>7.2f}% | {r["AUROC"]:>7.2f}% | {r["AUPR_IN"]:>9.2f}% | {r["AUPR_OUT"]:>10.2f}% | {r["ACC"]:>7.2f}%')
    # 
    # if 'All' in all_results:
    #     r = all_results['All']
    #     print(f'{"All":>6} | {r["TNR"]:>7.2f}% | {r["AUROC"]:>7.2f}% | {r["AUPR_IN"]:>9.2f}% | {r["AUPR_OUT"]:>10.2f}% | {r["ACC"]:>7.2f}%')
    # 
    # print('='*70 + '\n')
    # 
    # if save_results:
    #     excel_file = os.path.join(config_sfcr.save_dir, 'snr_test_results.xlsx')
    #     valid_snrs = sorted([int(snr) for snr in snr_range if snr in all_results])
    #     
    #     if valid_snrs or 'All' in all_results:
    #         data_dict = {
    #             '指标': ['TNR (%)', 'AUROC (%)', 'AUPR_IN (%)', 'AUPR_OUT (%)', 'ACC (%)']
    #         }
    #         
    #         for snr in valid_snrs:
    #             r = all_results[str(snr)]
    #             data_dict[snr] = [
    #                 round(r["TNR"], 2),
    #                 round(r["AUROC"], 2),
    #                 round(r["AUPR_IN"], 2),
    #                 round(r["AUPR_OUT"], 2),
    #                 round(r["ACC"], 2)
    #             ]
    #         
    #         if 'All' in all_results:
    #             r = all_results['All']
    #             data_dict['All'] = [
    #                 round(r["TNR"], 2),
    #                 round(r["AUROC"], 2),
    #                 round(r["AUPR_IN"], 2),
    #                 round(r["AUPR_OUT"], 2),
    #                 round(r["ACC"], 2)
    #             ]
    #         
    #         try:
    #             df = pd.DataFrame(data_dict)
    #             df.to_excel(excel_file, index=False, engine='openpyxl')
    #             print(f'结果已保存到Excel: {excel_file}')
    #         except ImportError:
    #             print('警告: 未安装openpyxl库，无法保存Excel文件。请运行: pip install openpyxl')
    #             try:
    #                 df.to_excel(excel_file, index=False, engine='xlsxwriter')
    #                 print(f'结果已保存到Excel: {excel_file} (使用xlsxwriter)')
    #             except ImportError:
    #                 print('警告: 未安装xlsxwriter库，跳过Excel保存')
    #         except Exception as e:
    #             print(f'保存Excel文件时出错: {str(e)}')
    #     
    #     txt_file = os.path.join(config_sfcr.save_dir, 'snr_test_results.txt')
    #     with open(txt_file, 'w', encoding='utf-8') as f:
    #         f.write('SNR 测试结果汇总\n')
    #         f.write('='*70 + '\n')
    #         f.write(header + '\n')
    #         f.write('-'*70 + '\n')
    #         
    #         for snr in snr_range:
    #             if snr in all_results:
    #                 r = all_results[snr]
    #                 f.write(f'{snr:>6} | {r["TNR"]:>7.2f}% | {r["AUROC"]:>7.2f}% | {r["AUPR_IN"]:>9.2f}% | {r["AUPR_OUT"]:>10.2f}% | {r["ACC"]:>7.2f}%\n')
    #         
    #         if 'All' in all_results:
    #             r = all_results['All']
    #             f.write(f'{"All":>6} | {r["TNR"]:>7.2f}% | {r["AUROC"]:>7.2f}% | {r["AUPR_IN"]:>9.2f}% | {r["AUPR_OUT"]:>10.2f}% | {r["ACC"]:>7.2f}%\n')
    #         
    #         f.write('='*70 + '\n')
    #     
    #     print(f'结果已保存到文本: {txt_file}')
    
    return all_results

if __name__ == '__main__':
    begin_time = time.time()
    config_sfcr = Config_SFCR()
    configopenood = config.Config(*config_sfcr.config_files)
    os.makedirs(config_sfcr.save_dir, exist_ok=True)
    
    loader_dict = get_dataloader(configopenood)
    test_loader = loader_dict['test']
    ood_loader_dict = get_ood_dataloader(configopenood)
    test_ood_loader = ood_loader_dict['val']


    #train
    train_loader = loader_dict['train']
    val_loader = loader_dict['val']
    test_loader = loader_dict['test']
    test_ood_loader = ood_loader_dict['val']
    model = SFCRModel(config_sfcr)
    trainer = TrainingManager(model, config_sfcr)
    trainer.train_phase1(train_loader, val_loader)

    best_model_path = os.path.join('results/frri_osr_sfc_drone/best.ckpt')

    all_results = test_ood_by_snr(
        model=SFCRModel,
        test_loader=test_loader,
        test_ood_loader=test_ood_loader,
        config_sfcr=config_sfcr,
        model_path=best_model_path,
        snr_range=None,
        save_results=True
    )
    

