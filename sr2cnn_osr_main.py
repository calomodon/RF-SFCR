import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
# 导入 OpenOOD 相关模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import openood.utils.comm as comm
from openood.utils import config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators.metrics import compute_all_metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False
# 设备和随机种子配置
os.environ['PYTHONHASHSEED'] = str(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局变量
best_acc = 0
best_epoch_idx = 0
begin_time = time.time()

class Config_SR2CNN:
    def __init__(self):
        # 训练参数
        self.num_classes = 26  # ID 类别数
        self.feature_dim = 256
        self.epochs = 100
        
        self.batch_size = 256
        self.learning_rate = 1e-3
        
        # SR2CNN 特定参数
        self.lam_center = 0.03  # center loss 权重
        self.lam_encoder = 0.10   # reconstruction loss 权重
        
        # 开集识别参数（参考 test.py）
        self.distance_type = 'MahaDiag'  # 'Maha', 'MahaDiag', 'SigmaEye'
        self.coef = 0.15  # 距离阈值系数（初始值）
        self.coef_unknown = 1.0  # 未知类别距离系数
        
        # 路径参数
        self.save_dir = "results/sr2cnn_osr_wifi"
        self.device = device
        
        # 配置文件
        self.config_files = [
            './configs/datasets/wifi/wifi_benchmark_2.yml',
            './configs/datasets/wifi/wifi_ood_benchmark_2.yml',
            './configs/networks/resnet18_32x32.yml',
            './configs/pipelines/test/test_ood.yml',
            './configs/preprocessors/base_preprocessor.yml',
            './configs/postprocessors/msp.yml',
        ]

class SR2CNN(nn.Module):
    def __init__(self, num_class, feature_dim):
        super(SR2CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, (3,3), stride=1, padding=(1,1))
        self.conv2 = nn.Conv2d(64, 128, (3,3), stride=1, padding=(1,1))
        self.conv3 = nn.Conv2d(128, 256, (3,3), stride=1, padding=(1,1))
        self.conv4 = nn.Conv2d(256, 512, (3,3), stride=1, padding=(1,1))

        self.maxPool = nn.MaxPool2d((1,2), stride=(1,2), return_indices=True)
        self.avgPool = nn.AvgPool2d((2,2), stride=2)

        self.fc0 = nn.Linear(16*512, 1024)
        self.fc1 = nn.Linear(1024, 512)

        # semantic layer
        self.fc2 = nn.Linear(512, feature_dim)
        # output layer
        self.fc3 = nn.Linear(feature_dim, num_class)  

        self.dropoutConv1 = nn.Dropout2d()  
        self.dropoutConv2 = nn.Dropout2d()
        self.dropoutConv3 = nn.Dropout2d()
        self.dropoutConv4 = nn.Dropout2d()  
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.dropout3 = nn.Dropout()

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(feature_dim)
        self.bnconv1 = nn.BatchNorm2d(64)
        self.bnconv2 = nn.BatchNorm2d(128)
        self.bnconv3 = nn.BatchNorm2d(256)
        self.bnconv4 = nn.BatchNorm2d(512)

        # decoder
        self.conv1_r = nn.ConvTranspose2d(64, 1, (3,3), stride=1, padding=(1,1))
        self.conv2_r = nn.ConvTranspose2d(128, 64, (3,3), stride=1, padding=(1,1))
        self.conv3_r = nn.ConvTranspose2d(256, 128, (3,3), stride=1, padding=(1,1))
        self.conv4_r = nn.ConvTranspose2d(512, 256, (3,3), stride=1, padding=(1,1))

        self.maxPool_r = nn.MaxUnpool2d((1,2), stride=(1,2))
        self.avgPool_r = nn.UpsamplingNearest2d(scale_factor=2)

        self.fc0_r = nn.Linear(1024, 16*512)
        self.fc1_r = nn.Linear(512, 1024)
        self.fc2_r = nn.Linear(feature_dim, 512)

        self.dropoutConv1_r = nn.Dropout2d()  
        self.dropoutConv2_r = nn.Dropout2d()
        self.dropoutConv3_r = nn.Dropout2d()
        self.dropoutConv4_r = nn.Dropout2d()  
        self.dropout1_r = nn.Dropout()
        self.dropout2_r = nn.Dropout()
        self.dropout3_r = nn.Dropout()

        self.bn1_r = nn.BatchNorm1d(16*512)
        self.bn2_r = nn.BatchNorm1d(1024)
        self.bn3_r = nn.BatchNorm1d(512)

        self.bnconv1_r = nn.BatchNorm2d(1)
        self.bnconv2_r = nn.BatchNorm2d(64)
        self.bnconv3_r = nn.BatchNorm2d(128)
        self.bnconv4_r = nn.BatchNorm2d(256)

    def forward(self, x):
        x = x.view(-1, 1, 2, 256)
        
        x, _ = self.maxPool(self.dropoutConv1(F.relu(self.bnconv1(self.conv1(x)))))
        x, _ = self.maxPool(self.dropoutConv2(F.relu(self.bnconv2(self.conv2(x)))))
        x, _ = self.maxPool(self.dropoutConv3(F.relu(self.bnconv3(self.conv3(x)))))
        x = self.avgPool(self.dropoutConv4(F.relu(self.bnconv4(self.conv4(x)))))

        x = x.view(-1, 16*512)
        x = self.dropout1(F.relu(self.bn1(self.fc0(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc1(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc2(x))))
        
        x = self.fc3(x)
        return x

    def decoder(self, x):
        x = x.view(-1, 1, 2, 256)
        
        x, indices1 = self.maxPool(self.dropoutConv1(F.relu(self.bnconv1(self.conv1(x)))))
        x, indices2 = self.maxPool(self.dropoutConv2(F.relu(self.bnconv2(self.conv2(x)))))
        x, indices3 = self.maxPool(self.dropoutConv3(F.relu(self.bnconv3(self.conv3(x)))))
        x = self.avgPool(self.dropoutConv4(F.relu(self.bnconv4(self.conv4(x)))))

        x = x.view(-1, 16*512)
        x = self.dropout1(F.relu(self.bn1(self.fc0(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc1(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc2(x))))
        
        x = F.relu(self.bn3_r(self.fc2_r(x)))
        x = F.relu(self.bn2_r(self.fc1_r(x)))
        x = F.relu(self.bn1_r(self.fc0_r(x)))

        x = x.view(-1, 512, 1, 16)

        x = F.relu(self.bnconv4_r(self.conv4_r(self.avgPool_r(x))))
        x = F.relu(self.bnconv3_r(self.conv3_r(self.maxPool_r(x, indices3))))
        x = F.relu(self.bnconv2_r(self.conv2_r(self.maxPool_r(x, indices2))))
        x = F.relu(self.bnconv1_r(self.conv1_r(self.maxPool_r(x, indices1))))

        x = x.view(-1, 2, 256)
        return x

    def getSemantic(self, x):        
        x = x.view(-1, 1, 2, 256)
        
        x, _ = self.maxPool(self.dropoutConv1(F.relu(self.bnconv1(self.conv1(x)))))
        x, _ = self.maxPool(self.dropoutConv2(F.relu(self.bnconv2(self.conv2(x)))))
        x, _ = self.maxPool(self.dropoutConv3(F.relu(self.bnconv3(self.conv3(x)))))
        x = self.avgPool(self.dropoutConv4(F.relu(self.bnconv4(self.conv4(x)))))

        x = x.view(-1, 16*512)
        x = self.dropout1(F.relu(self.bn1(self.fc0(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc1(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc2(x))))

        return x

class CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: 
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

class SR2CNNModel(nn.Module):
    def __init__(self, config_sr2cnn):
        super(SR2CNNModel, self).__init__()
        self.model = SR2CNN(num_class=config_sr2cnn.num_classes, feature_dim=config_sr2cnn.feature_dim)
        self.config_sr2cnn = config_sr2cnn
        
    def forward(self, x):
        return self.model(x)
    
    def getSemantic(self, x):
        return self.model.getSemantic(x)

class TrainingManager:
    def __init__(self, model, config_sr2cnn):
        self.model = model.to(device)
        self.config_sr2cnn = config_sr2cnn
        
        # Cross entropy loss
        self.criterion = nn.CrossEntropyLoss()
        # Center loss
        self.criterion_cent = CenterLoss(
            num_classes=config_sr2cnn.num_classes, 
            feat_dim=config_sr2cnn.feature_dim, 
            use_gpu=torch.cuda.is_available()
        )
        # Reconstruction loss
        self.criterion_encoder = nn.MSELoss()
        
        # Optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=config_sr2cnn.learning_rate)
        self.optimizer_cent = optim.Adam(self.criterion_cent.parameters(), lr=config_sr2cnn.learning_rate)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config_sr2cnn.epochs
        )
    
    def train_phase1(self, train_loader, val_loader):
        global best_acc, best_epoch_idx
        best_acc = 0
        best_epoch_idx = 0

        print('Start Training')
        print('Using lam_center {}, lam_encoder {}, feature dimension {}'.format(
            self.config_sr2cnn.lam_center, 
            self.config_sr2cnn.lam_encoder, 
            self.config_sr2cnn.feature_dim
        ))

        for epoch in range(self.config_sr2cnn.epochs):
            train_metrics = self.train_epoch(train_loader, epoch + 1)
            val_metrics = eval_acc(self.model, val_loader, epoch + 1)
            save_model(self.model, val_metrics, self.config_sr2cnn.epochs, self.config_sr2cnn.save_dir)
            report(train_metrics, val_metrics)

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        for batch in tqdm(loader, desc=f'Epoch {epoch}'):
            x = batch['data'].to(device)
            y = batch['label'].to(device)
            
            self.optimizer.zero_grad()
            self.optimizer_cent.zero_grad()
            
            outputs = self.model(x)
            loss_cross = self.criterion(outputs, y)
            loss_cent = self.criterion_cent(self.model.getSemantic(x), y)
            loss_encoder = self.criterion_encoder(self.model.model.decoder(x), x.view(-1, 2, 256))
            
            loss = loss_cross + self.config_sr2cnn.lam_center * loss_cent + self.config_sr2cnn.lam_encoder * loss_encoder
            loss.backward()
            self.optimizer.step()
            self.optimizer_cent.step()
            self.scheduler.step()

            total_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
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
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Eval: '):
            x = batch['data'].to(device)
            y = batch['label'].to(device)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            _, pred = torch.max(outputs, dim=1)
            
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

def calculate_distance(x, transform_matrix):
    """计算马氏距离（参考 test.py）"""
    # x 是一维数组 (feat_dim,)
    # transform_matrix 是 (feat_dim, feat_dim)
    # 返回标量距离
    if x.ndim == 1:
        return np.sqrt(np.dot(np.dot(x, transform_matrix), x))
    else:
        return np.sqrt(np.dot(np.dot(x, transform_matrix), x.transpose()))

def gen_semantic_vec(model, train_loader, num_classes, distance_type='MahaDiag'):
    """生成语义向量中心和距离变换矩阵（参考 test.py）"""
    model.eval()
    semantic_center_map = {}
    distance_map = {}
    
    # 收集每个类别的特征
    class_features = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for batch in tqdm(train_loader, desc='Extracting features'):
            x = batch['data'].to(device)
            y = batch['label'].to(device)
            
            features = model.getSemantic(x).cpu().numpy()
            labels = y.cpu().numpy()
            
            for feat, label in zip(features, labels):
                class_features[label].append(feat)
    
    # 计算每个类别的中心和协方差矩阵
    cov_inv_map = {}
    cov_inv_diag_map = {}
    sigma_identity_map = {}
    
    for certain_class in range(num_classes):
        if len(class_features[certain_class]) == 0:
            continue
            
        features_array = np.array(class_features[certain_class])
        semantic_center_map[certain_class] = np.mean(features_array, axis=0)
        
        covariance_mat = np.cov(features_array, rowvar=False, bias=True)
        cov_inv = np.linalg.pinv(covariance_mat)
        cov_inv_map[certain_class] = cov_inv
        
        cov_inv_diag_mat = np.diagflat(1 / (covariance_mat.diagonal()))
        cov_inv_diag_mat[cov_inv_diag_mat == np.inf] = 0.0
        cov_inv_diag_map[certain_class] = cov_inv_diag_mat
        
        sigma = np.mean(np.diagflat(covariance_mat.diagonal()))
        sigma_identity_map[certain_class] = 1 / sigma * np.eye(covariance_mat.shape[0])
    
    distance_map['Maha'] = cov_inv_map
    distance_map['MahaDiag'] = cov_inv_diag_map
    distance_map['SigmaEye'] = sigma_identity_map
    
    return semantic_center_map, distance_map

def classify_evol(transform_map, semantic_center_map, semantic_vector, coef, coef_unknown, num_classes):
    """
    原始SR2CNN的分类函数（参考test.py的classify_evol）
    包含动态学习未知类的机制
    """
    predicted_label = -1
    min_dist = float('inf')
    min_dist_recorded = float('inf')
    dists_known_I = []
    if_known = False
    
    # 计算到每个已知类别的距离
    eyeMat = None  # 将在循环内定义，与原始实现保持一致
    for certain_class in range(num_classes):
        if certain_class not in semantic_center_map:
            continue
            
        semantic_center = semantic_center_map[certain_class]
        dist = calculate_distance(semantic_vector - semantic_center, transform_map[certain_class])
        
        # 使用与原始实现相同的方式定义 eyeMat（在循环内定义，但值应该相同）
        eyeMat = np.eye(semantic_center_map[certain_class].shape[0])
        dist_I = calculate_distance(semantic_vector - semantic_center, eyeMat)
        
        dists_known_I.append(dist_I)
        
        if dist < 3 * np.sqrt(semantic_vector.shape[0]) * coef:
            if_known = True
        
        if dist < min_dist:
            min_dist = dist
            predicted_label = certain_class
    
    # 计算平均距离和最小距离（与原始实现保持一致）
    mean_dist = np.mean(dists_known_I) if len(dists_known_I) > 0 else 0
    min_dist = min(dists_known_I) if len(dists_known_I) > 0 else float('inf')
    
    # 如果不是已知类，检查是否属于已记录的未知类
    if not if_known:
        # 第一个未知实例出现（与原始实现保持一致：检查是否只有已知类）
        if len(semantic_center_map.keys()) == num_classes:
            predicted_label = -1
        else:
            # 检查是否属于已记录的未知类
            recorded_unknowns = set(semantic_center_map.keys()) - set(list(range(num_classes)))
            if_recorded = False
            
            # 确保 eyeMat 已定义（如果循环没有执行，使用 semantic_vector 的维度）
            if eyeMat is None:
                eyeMat = np.eye(semantic_vector.shape[0])
            
            for recorded_unknown_class in recorded_unknowns:
                semantic_center = semantic_center_map[recorded_unknown_class]
                dist = calculate_distance(semantic_vector - semantic_center, eyeMat)
                if dist <= coef_unknown * (min_dist + mean_dist) / 2:
                    if_recorded = True
                    break
            
            if if_recorded:
                # 找到最近的已记录未知类
                for recorded_unknown_class in recorded_unknowns:
                    semantic_center = semantic_center_map[recorded_unknown_class]
                    dist = calculate_distance(semantic_vector - semantic_center, eyeMat)
                    if dist < min_dist_recorded:
                        min_dist_recorded = dist
                        predicted_label = recorded_unknown_class
            else:
                # 新的未知类
                predicted_label = -1
    
    return predicted_label

def test_ood(model, train_loader, id_loader, ood_loader, config_sr2cnn):
    model.eval()
    num_classes = config_sr2cnn.num_classes
    
    # 生成语义向量中心和距离变换矩阵
    print("Generating semantic centers and distance matrices...")
    semantic_center_map, distance_map = gen_semantic_vec(
        model, train_loader, num_classes, config_sr2cnn.distance_type
    )
    
    transform_map = distance_map[config_sr2cnn.distance_type]
    
    # 收集测试数据的语义特征和标签
    id_features = []
    id_labels = []
    ood_features = []
    ood_labels = []
    
    print("Extracting ID features...")
    with torch.no_grad():
        for batch in tqdm(id_loader, desc='ID Inference'):
            x = batch['data'].to(device)
            y = batch['label'].to(device)
            
            features = model.getSemantic(x).cpu().numpy()
            id_features.append(features)
            id_labels.append(y.cpu().numpy())
    
    print("Extracting OOD features...")
    with torch.no_grad():
        for batch in tqdm(ood_loader, desc='OOD Inference'):
            x = batch['data'].to(device)
            y = batch['label'].to(device)
            
            features = model.getSemantic(x).cpu().numpy()
            ood_features.append(features)
            ood_labels.append(y.cpu().numpy())
    
    id_features = np.concatenate(id_features, axis=0)
    id_labels = np.concatenate(id_labels, axis=0)
    ood_features = np.concatenate(ood_features, axis=0)
    ood_labels = np.concatenate(ood_labels, axis=0)
    
    # 分类并计算置信度（使用原始SR2CNN的动态学习机制）
    print("Classifying samples (with dynamic unknown class learning)...")
    all_features = np.concatenate([id_features, ood_features], axis=0)
    all_labels = np.concatenate([id_labels, ood_labels], axis=0)
    
    # 动态学习未知类（参考原始test.py的实现）
    semanticMap = semantic_center_map.copy()
    new_class_instances_map = {}
    new_class_index = num_classes
    
    predictions = []
    confidences = []
    eyeMat = np.eye(all_features.shape[1])
    
    for feat, label in tqdm(zip(all_features, all_labels), desc='Classifying', total=len(all_features)):
        predicted_label = classify_evol(
            transform_map, semanticMap, feat, 
            config_sr2cnn.coef, config_sr2cnn.coef_unknown, num_classes
        )
        
        # 如果是新的未知类（predicted_label == -1），记录它
        if predicted_label == -1:
            # 初始化新的未知类中心
            semanticMap[new_class_index] = feat
            new_class_instances_map[new_class_index] = [feat]
            new_class_index += 1
        # 如果属于已记录的未知类，更新该类别的中心
        elif predicted_label >= num_classes:
            if predicted_label in new_class_instances_map:
                new_class_instances_map[predicted_label].append(feat)
                semanticMap[predicted_label] = np.mean(new_class_instances_map[predicted_label], axis=0)
        
        predictions.append(predicted_label)
        
        # 计算置信度：使用距离的倒数
        if predicted_label == -1 or predicted_label >= num_classes:
            # 未知类别：找到最小距离
            min_dist_I = float('inf')
            for certain_class in range(num_classes):
                if certain_class in semantic_center_map:
                    semantic_center = semantic_center_map[certain_class]
                    dist_I = calculate_distance(feat - semantic_center, eyeMat)
                    if dist_I < min_dist_I:
                        min_dist_I = dist_I
            conf = 1.0 / (1.0 + min_dist_I)  # 未知类别的置信度较低
        else:
            # 已知类别：使用马氏距离
            semantic_center = semantic_center_map[predicted_label]
            dist = calculate_distance(feat - semantic_center, transform_map[predicted_label])
            conf = 1.0 / (1.0 + dist)  # 已知类别的置信度
        
        confidences.append(conf)
    
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    
    # 准备 OpenOOD 标准指标计算的数据
    id_mask = all_labels < num_classes
    ood_mask = ~id_mask
    
    # ID 数据
    id_pred = predictions[id_mask]
    id_conf = confidences[id_mask]
    id_label = all_labels[id_mask]
    
    # OOD 数据
    ood_pred = predictions[ood_mask]
    ood_conf = confidences[ood_mask]
    ood_label = -1 * np.ones_like(ood_pred)  # OpenOOD 标准：OOD 标签为 -1
    
    # 合并数据
    pred = np.concatenate([id_pred, ood_pred])
    conf = np.concatenate([id_conf, ood_conf])
    label = np.concatenate([id_label, ood_label])
    
    # 计算 OpenOOD 标准指标
    metrics = compute_all_metrics(conf, label, pred)
    fpr, auroc, aupr_in, aupr_out, accuracy = metrics
    
    # 计算 TNR (True Negative Rate = 1 - FPR)
    tnr = 1 - fpr
    
    # 返回指标
    results = {
        'TNR': tnr * 100,
        'AUROC': auroc * 100,
        'AUPR_IN': aupr_in * 100,
        'AUPR_OUT': aupr_out * 100,
        'ACC': accuracy * 100
    }
    
    # 打印结果
    print('\n' + '='*70)
    print('Open Set Recognition Test Results (SR2CNN):')
    print('='*70)
    print(f'TNR:      {results["TNR"]:.2f}%')
    print(f'AUROC:    {results["AUROC"]:.2f}%')
    print(f'AUPR_IN:  {results["AUPR_IN"]:.2f}%')
    print(f'AUPR_OUT: {results["AUPR_OUT"]:.2f}%')
    print(f'ACC:      {results["ACC"]:.2f}%')
    
    # 返回结果和评分数据
    return results, {'pred': pred, 'conf': conf, 'label': label}

def test_ood_by_snr(model, train_loader, test_loader, test_ood_loader, config_sr2cnn, model_path,
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
    #     current_model = SR2CNNModel(config_sr2cnn)
    #     current_model = load_model(current_model, model_path).to(device)
    #     current_model.eval()
    #     
    #     try:
    #         results = test_ood(current_model, train_loader, filtered_test_loader, filtered_ood_loader, config_sr2cnn)
    #         all_results[snr] = results
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
    
    current_model = SR2CNNModel(config_sr2cnn)
    current_model = load_model(current_model, model_path).to(device)
    current_model.eval()
    
    try:
        results_all, score_data_all = test_ood(current_model, train_loader, test_loader, test_ood_loader, config_sr2cnn)
        all_results['All'] = results_all
        
        # 保存全SNR的评分数据到npz文件
        if save_results:
            npz_dir = os.path.join(config_sr2cnn.save_dir, 'scores')
            os.makedirs(npz_dir, exist_ok=True)
            npz_path = os.path.join(npz_dir, 'sr2cnn_rml201610a.npz')
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
    #     excel_file = os.path.join(config_sr2cnn.save_dir, 'snr_test_results.xlsx')
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
    #     txt_file = os.path.join(config_sr2cnn.save_dir, 'snr_test_results.txt')
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
    config_sr2cnn = Config_SR2CNN()
    configopenood = config.Config(*config_sr2cnn.config_files)
    os.makedirs(config_sr2cnn.save_dir, exist_ok=True)

    loader_dict = get_dataloader(configopenood)
    train_loader = loader_dict['train']
    val_loader = loader_dict['val']
    test_loader = loader_dict['test']
    
    ood_loader_dict = get_ood_dataloader(configopenood)
    test_ood_loader = ood_loader_dict['val']
    
    model = SR2CNNModel(config_sr2cnn)

    # train
    trainer = TrainingManager(model, config_sr2cnn)
    trainer.train_phase1(train_loader, val_loader)

    best_model_path = os.path.join('results/sr2cnn_osr_wifi/best.ckpt')
    
    all_results = test_ood_by_snr(
        model=SR2CNNModel,
        train_loader=train_loader,
        test_loader=test_loader,
        test_ood_loader=test_ood_loader,
        config_sr2cnn=config_sr2cnn,
        model_path=best_model_path,
        snr_range=None,
        save_results=True
    )
    

