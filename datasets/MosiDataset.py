import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random

class MOSIDataset(Dataset):
    """
    CMU-MOSI Aligned_50 Dataset Loader
    Features:
      1. Fixed/Dynamic Missing Modality (drop_rate)
      2. Fixed/Dynamic Noise Injection (noise_level) 🔥 NEW
      3. Strict Train/Val separation to prevent oscillation
    """

    def __init__(self, pkl_path, split, normalize=True, label_normalize=True, 
                 drop_rate=0.0, noise_level=0.0, seed=1111):
        """
        Args:
            pkl_path (str): path to aligned_50.pkl
            split (str): one of ['train', 'valid', 'test']
            drop_rate (float): probability of completely DROPPING a modality (Missingness)
            noise_level (float): intensity of noise to inject (0.0 to 1.0)
                                 - For Audio/Vision: Standard Deviation of Gaussian Noise
                                 - For Text: Probability of Word Dropout
            seed (int): random seed for static validation
        """
        # 1. 加载所有数据
        with open(pkl_path, 'rb') as f:
            all_data = pickle.load(f)
        
        self.data = all_data[split]
        # 🔥 关键：获取训练集数据用于计算统计量
        train_data_ref = all_data['train'] 

        self.split = split
        self.num_samples = self.data['raw_text'].shape[0]
        self.normalize = normalize
        self.label_normalize = label_normalize
        
        self.drop_rate = drop_rate
        self.noise_level = noise_level # 🔥 新增参数
        self.seed = seed

        # -----------------------------------------------------------
        # 🔥 关键逻辑：验证集 Mask 固定 (Static Validation Mask)
        # -----------------------------------------------------------
        self.fixed_missing_modes = None
        if split != 'train' and drop_rate > 0:
            rng = np.random.RandomState(seed)
            self.fixed_missing_modes = []
            for _ in range(self.num_samples):
                if rng.rand() >= drop_rate:
                    mode = 6 # no drop
                else:
                    mode = rng.randint(0, 6)
                self.fixed_missing_modes.append(mode)

        # -----------------------------------------------------------
        # 2. Normalize Audio + Vision (Strictly use Train Stats)
        # -----------------------------------------------------------
        if self.normalize:
            self.audio_mean = np.mean(train_data_ref['audio'], axis=(0, 1), keepdims=True)
            self.audio_std = np.std(train_data_ref['audio'], axis=(0, 1), keepdims=True) + 1e-5
            
            self.vision_mean = np.mean(train_data_ref['vision'], axis=(0, 1), keepdims=True)
            self.vision_std = np.std(train_data_ref['vision'], axis=(0, 1), keepdims=True) + 1e-5

            # Apply to current data
            self.data['audio'] = (self.data['audio'] - self.audio_mean) / self.audio_std
            self.data['vision'] = (self.data['vision'] - self.vision_mean) / self.vision_std
        
        # -----------------------------------------------------------
        # 3. Normalize Labels
        # -----------------------------------------------------------
        if self.label_normalize:
            self.label_mean = np.mean(train_data_ref['regression_labels'])
            self.label_std = np.std(train_data_ref['regression_labels']) + 1e-8

            self.data['regression_labels'] = (
                (self.data['regression_labels'] - self.label_mean) / self.label_std
            )

    # -----------------------------------------------------------
    # 🔥 Helper: Apply Noise (Standard Robustness Protocol)
    # -----------------------------------------------------------
    def apply_noise(self, text, audio, vision, idx):
        """
        注入噪声。
        - 训练集：动态随机噪声。
        - 验证集/测试集：基于 (seed + idx) 的固定噪声，防止震荡。
        """
        if self.noise_level <= 0.0:
            return text, audio, vision

        # 1. 确定随机数生成器
        if self.split == 'train':
            rng = np.random # 动态
        else:
            # 静态：每个样本拥有自己固定的随机状态
            rng = np.random.RandomState(self.seed + idx)

        # 2. Audio & Vision: Additive Gaussian Noise
        # feature ~ N(0, 1) after norm. So noise_level is directly sigma.
        # standard robust protocol: add noise to features
        noise_a = rng.normal(loc=0.0, scale=self.noise_level, size=audio.shape)
        noise_v = rng.normal(loc=0.0, scale=self.noise_level, size=vision.shape)
        
        audio = audio + torch.tensor(noise_a, dtype=torch.float32)
        vision = vision + torch.tensor(noise_v, dtype=torch.float32)

        # 3. Text: Random Word Dropout (Masking)
        # shape: [Channels, Seq_Len]
        # 我们随机把某些时间步(单词)的特征全部置为 0
        seq_len = text.shape[1]

        keep_mask = (rng.rand(1, seq_len) > self.noise_level).astype(np.float32)
        keep_tensor = torch.tensor(keep_mask, dtype=torch.float32)

        # 只对 input_ids (channel 0) 应用 mask
        # 复制 text 以免修改原始数据
        text = text.clone()
        text[0, :] = text[0, :] * keep_tensor # Channel 0: IDs * mask

        return text, audio, vision

    # -----------------------------------------------------------
    # Helper: Missing Modality Simulator
    # -----------------------------------------------------------
    def get_missing_mode(self, idx):
        if self.fixed_missing_modes is not None:
            return self.fixed_missing_modes[idx]
        
        if random.random() >= self.drop_rate:
            return 6
        else:
            return random.randint(0, 5)

    def apply_mask(self, text, audio, vision, mode):
        missing_text = 0
        missing_audio = 0
        missing_vision = 0
        
        if mode == 6:
            return text, audio, vision, mode, 0, 0, 0

        if mode in [1, 3, 5]: # drop text
            text = torch.zeros_like(text)
            missing_text = 1
        if mode in [2, 4, 5]: # drop audio
            audio = torch.zeros_like(audio)
            missing_audio = 1
        if mode in [0, 3, 4]: # drop vision
            vision = torch.zeros_like(vision)
            missing_vision = 1

        return text, audio, vision, mode, missing_text, missing_audio, missing_vision

    # -----------------------------------------------------------
    # Main __getitem__
    # -----------------------------------------------------------
    def __getitem__(self, idx):
        # 1. Load Raw Features
        text = torch.tensor(self.data['text_bert'][idx], dtype=torch.float32)
        audio = torch.tensor(self.data['audio'][idx], dtype=torch.float32)
        vision = torch.tensor(self.data['vision'][idx], dtype=torch.float32)

        # 2. 🔥 Step 1: Inject Noise (Fidelity Degradation)
        # 先加噪声，模拟信号质量下降（Quality Net 应该检测出这个）
        text, audio, vision = self.apply_noise(text, audio, vision, idx)

        # 3. 🔥 Step 2: Apply Missing Modality
        # 再处理完全缺失。如果缺失了，刚才加的噪声也会变成0，这是合理的（死传感器没有噪声）
        mode = self.get_missing_mode(idx)
        text, audio, vision, mode, mt, ma, mv = self.apply_mask(text, audio, vision, mode)

        sample = {
            'text': text,
            'audio': audio,
            'vision': vision,
            'classification_labels': torch.tensor(self.data['classification_labels'][idx], dtype=torch.float32),
            'regression_labels': torch.tensor(self.data['regression_labels'][idx], dtype=torch.float32),
            'missing_text': torch.tensor(mt, dtype=torch.bool),
            'missing_audio': torch.tensor(ma, dtype=torch.bool),
            'missing_vision': torch.tensor(mv, dtype=torch.bool),
            'missing_mode': torch.tensor(mode, dtype=torch.long),
        }
        return sample

    def __len__(self):
        return self.num_samples

if __name__ == "__main__":
    import os
    import sys

    # ==========================================
    # 🛠️ 测试配置区域
    # ==========================================
    # 如果你有真实数据，把这里改成你的路径，例如: "../data/aligned_50.pkl"
    # 如果没有，保持为 "dummy_data.pkl"，我会自动生成一个假数据给你测试
    pkl_path = "../../../data/ytzhu/CMU-MOSI/Processed/aligned_50.pkl" 
    
    # ==========================================
    # 0. 自动生成 Dummy Data (方便直接运行测试)
    # ==========================================
    if not os.path.exists(pkl_path):
        print(f"⚠️ 未找到 {pkl_path}，正在生成临时测试数据...")
        num_samples = 100
        dummy_data = {
            'train': {
                'raw_text': np.array(["sample"] * num_samples),
                'text_bert': np.random.randn(num_samples, 50, 768), # GloVe/BERT dim
                'audio': np.random.randn(num_samples, 50, 74),
                'vision': np.random.randn(num_samples, 50, 35),
                'classification_labels': np.random.randint(0, 2, (num_samples, 1)),
                'regression_labels': np.random.randn(num_samples, 1)
            },
            'valid': {
                'raw_text': np.array(["sample"] * num_samples),
                'text_bert': np.random.randn(num_samples, 50, 768),
                'audio': np.random.randn(num_samples, 50, 74),
                'vision': np.random.randn(num_samples, 50, 35),
                'classification_labels': np.random.randint(0, 2, (num_samples, 1)),
                'regression_labels': np.random.randn(num_samples, 1)
            }
        }
        with open(pkl_path, 'wb') as f:
            pickle.dump(dummy_data, f)
        print("✅ 临时数据已生成！")

    # ==========================================
    # 1. 测试验证集的“静态一致性” (关键！)
    # ==========================================
    print("\n" + "="*50)
    print("🧪 测试 1: 验证集 Mask 是否固定 (防止震荡)")
    print("="*50)
    
    # 初始化验证集，设置 drop_rate=0.5，种子固定
    val_ds = MOSIDataset(pkl_path, split='valid', drop_rate=0.5, seed=2024)
    
    print(f"验证集样本数: {len(val_ds)}")
    idx_to_test = 0
    
    modes = []
    print(f"读取验证集索引 {idx_to_test} 的数据 5 次...")
    for i in range(5):
        sample = val_ds[idx_to_test]
        mode = sample['missing_mode'].item()
        modes.append(mode)
        print(f"  第 {i+1} 次读取 -> Missing Mode: {mode}")
        
    if len(set(modes)) == 1:
        print("✅ pass: 验证集 Mask 稳定固定！(Loss 不会震荡了)")
    else:
        print("❌ fail: 验证集 Mask 在变动！(这会导致 Val Loss 震荡)")

    # ==========================================
    # 2. 测试训练集的“动态随机性”
    # ==========================================
    print("\n" + "="*50)
    print("🧪 测试 2: 训练集 Mask 是否随机 (增强鲁棒性)")
    print("="*50)
    
    # 初始化训练集
    train_ds = MOSIDataset(pkl_path, split='train', drop_rate=0.5)
    
    modes_train = []
    print(f"读取训练集索引 {idx_to_test} 的数据 10 次...")
    for i in range(10):
        sample = train_ds[idx_to_test]
        modes_train.append(sample['missing_mode'].item())
    
    print(f"  观测到的 Modes: {modes_train}")
    if len(set(modes_train)) > 1:
        print("✅ pass: 训练集 Mask 是动态变化的！")
    else:
        print("⚠️ warning: 训练集 Mask 居然没变？(可能是运气，或者是 bug)")

    # ==========================================
    # 3. 测试噪声注入 (Noise Injection)
    # ==========================================
    print("\n" + "="*50)
    print("🧪 测试 3: 噪声注入是否生效")
    print("="*50)

    # 干净的 Dataset
    clean_ds = MOSIDataset(pkl_path, split='train', normalize=False, noise_level=0.0)
    clean_audio = clean_ds[0]['audio']
    
    # 嘈杂的 Dataset
    noisy_ds = MOSIDataset(pkl_path, split='train', normalize=False, noise_level=0.5) # 大噪声
    noisy_audio = noisy_ds[0]['audio']
    
    # 计算差异
    diff = torch.abs(clean_audio - noisy_audio).mean().item()
    clean_std = clean_audio.std().item()
    noisy_std = noisy_audio.std().item()

    print(f"Clean Audio Std: {clean_std:.4f}")
    print(f"Noisy Audio Std: {noisy_std:.4f}")
    print(f"Mean Diff: {diff:.4f}")

    if noisy_std > clean_std:
        print("✅ pass: 噪声注入成功！特征方差变大了。")
    else:
        print("❌ fail: 噪声似乎没加进去？")
    
    # ==========================================
    # 4. 测试文本 Word Dropout
    # ==========================================
    print("\n" + "="*50)
    print("🧪 测试 4: 文本 Word Dropout")
    print("="*50)
    
    # 高 Dropout 率
    text_noisy_ds = MOSIDataset(pkl_path, split='train', noise_level=0.5) 
    sample_text = text_noisy_ds[0]['text'] # [Seq, Dim]
    
    # 检查是否有全 0 的行 (被 Mask 掉的单词)
    zero_rows = (sample_text.abs().sum(dim=1) == 0).sum().item()
    total_rows = sample_text.shape[0]
    
    print(f"Total Words: {total_rows}")
    print(f"Dropped Words (Zero Vectors): {zero_rows}")
    
    if zero_rows > 0:
         print("✅ pass: 文本 Word Dropout 生效！")
    else:
         print("⚠️ warning: 没有单词被丢弃 (可能是运气，也可能 noise_level 太低)")
         
    # 清理临时文件
    if pkl_path == "dummy_data.pkl" and os.path.exists(pkl_path):
        os.remove(pkl_path)
        print("\n🧹 已清理临时测试数据。")