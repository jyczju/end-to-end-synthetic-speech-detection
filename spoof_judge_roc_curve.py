#!/usr/bin/env python3
"""
数据集上的活体检测ROC曲线绘制脚本

该脚本用于绘制活体检测模型在VoxCeleb数据集上的ROC曲线，
其中未攻击的音频定义为正样本，攻击后的音频定义为负样本。

Author: Lingma
"""

import argparse
import os
import sys
import warnings
import json
import re
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import torch
import models

from spoof_judge import load_audio, judge_spoof

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['STHeiti']  # 使用黑体作为默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

warnings.filterwarnings("ignore", category=FutureWarning)
SAMPLE_RATE = 16000


class DatasetROCCalculator:
    """支持多数据集的ROC曲线计算器"""

    def __init__(self, model_path: str, model_type: str, dataset: str = "voxceleb", device: str = "cpu"):
        self.model_path = model_path
        self.model_type = model_type
        self.dataset = dataset.lower()
        self.device = device

        # 初始化模型
        self.model = self._init_model()
        
        # 根据数据集选择攻击参数
        self._init_attack_parameters()

    def _init_model(self):
        """初始化模型"""
        # Set device
        device = torch.device(
            'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        self.device = device
        
        # Initialize model
        if self.model_type == 'SSDNet1D':
            model = models.SSDNet1D()
        elif self.model_type == 'SSDNet2D':
            model = models.SSDNet2D()
        elif self.model_type == 'DilatedNet':
            model = models.DilatedNet()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load model weights
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        return model

    def _init_attack_parameters(self):
        """初始化不同数据集的攻击参数"""
        if self.dataset == "voxceleb":
            # VoxCeleb攻击参数（根据spoof_judge_batch.py中的设置）
            self.atk_amps = [0.5, 0.5, 0.3966, 0.1178, 0.44, 0.5, 0.5, 0.3378, 0.5, 0.1344,
                             0.4641, 0.119, 0.481, 0.3819, 0.2124, 0.1794, 0.3569, 0.2895,
                             0.3477, 0.4853]
            self.atk_fs = [1999.99, 10000, 7060.15, 6583.37, 9498.15, 3347.5, 3100.75,
                           4320.05, 5000, 1074.48, 1468.86, 6159.21, 2667.74, 3018.91,
                           618.74, 821.02, 3867.59, 1217.95, 614.54, 3976.73]
            self.dataset_name = "VoxCeleb"
        elif self.dataset == "aishell":
            # AISHELL攻击参数（根据spoof_judge_batch.py中的设置）
            self.atk_amps = [0.0581, 0.0648, 0.036, 0.2922, 0.1546, 0.0095, 0.0573, 0.0555, 0.0436, 0.3988, 0.5436, 0.1017,
                            0.41, 0.36, 0.1337, 0.5293, 0.404, 0.3726, 0.505, 0.5127]
            self.atk_fs = [3671.06, 4592.98, 943.95, 3542.28, 4954.2, 2133, 636.12, 1440.66, 332.77, 696.97, 1941.43,
                          4013.25, 2386.69, 1949.86, 1425.04, 2981.95, 2586.65, 1141.28, 2659.63, 4781.89]
            self.dataset_name = "AISHELL"
        else:
            raise ValueError(f"不支持的数据集: {self.dataset}，支持的数据集: voxceleb, aishell")

    def load_metadata(self, metadata_path: str) -> dict:
        """加载数据集元数据"""
        metadata = {}
        try:
            with open(metadata_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, filename, _, _, label = parts
                        metadata[filename] = label
        except FileNotFoundError:
            print(f"警告: 未找到元数据文件 {metadata_path}，将假设所有文件都是bonafide")
            metadata = {}
        return metadata

    def calculate_roc_points(self, audio_dir: str, metadata_path: str = None,
                             iterations: int = 10, test_times_per_file: int = 5) -> Tuple[
        List[float], List[float], List[float]]:
        """
        计算ROC曲线的各个点

        Args:
            audio_dir: 音频文件目录
            metadata_path: 元数据文件路径（可选）
            iterations: 每次测试的迭代次数
            test_times_per_file: 每个音频文件的重复测试次数

        Returns:
            tuple: (y_true, y_scores_clean, y_scores_attacked)
                - y_true: 真实标签（1表示正样本/bonafide，0表示负样本/spoof）
                - y_scores_clean: 未攻击音频的bonafide概率分数
                - y_scores_attacked: 攻击后音频的bonafide概率分数
        """
        # 加载元数据
        metadata = self.load_metadata(metadata_path) if metadata_path else {}

        y_true = []  # 真实标签
        y_scores_clean = []  # 未攻击的bonafide概率
        y_scores_attacked = []  # 攻击后的bonafide概率

        # 获取所有wav文件
        wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        print(f"找到 {len(wav_files)} 个音频文件")

        for file_idx, filename in enumerate(tqdm(wav_files, desc="Processing audio files")):
            file_path = os.path.join(audio_dir, filename)

            # 提取文件编号（用于匹配攻击参数）
            match = re.search(r'\d+', filename)
            attack_index = int(match.group()) - 1 if match else file_idx % len(self.atk_amps)

            # 确保索引在有效范围内
            attack_index = min(attack_index, len(self.atk_amps) - 1)

            # 构建标签：未攻击样本为正样本(1)，攻击样本为负样本(0)
            # 对于每个文件，我们生成test_times_per_file * iterations个未攻击样本和攻击样本
            total_samples_per_file = test_times_per_file * iterations
            y_true.extend([1] * total_samples_per_file)  # 未攻击样本标签为1（正样本）
            y_true.extend([0] * total_samples_per_file)  # 攻击样本标签为0（负样本）

            # 测试未攻击的情况（正样本）
            clean_bonafide_probs = []
            # 对每个文件进行多次测试以获得更多数据点
            for test_round in range(test_times_per_file):
                for _ in range(iterations):
                    try:
                        # Load audio without attack
                        audio_tensor = load_audio(file_path, show_plot=False)
                        if audio_tensor is None:
                            clean_bonafide_probs.append(0.5)  # 出错时给中性分数
                            continue
                        
                        # Judge spoof
                        _, _, _, spoof_prob = judge_spoof(self.model, audio_tensor, self.device)
                        bonafide_prob = 1 - spoof_prob  # bonafide probability
                        clean_bonafide_probs.append(bonafide_prob)
                    except Exception as e:
                        print(f"处理文件 {filename} 时出错: {e}")
                        clean_bonafide_probs.append(0.5)  # 出错时给中性分数

            y_scores_clean.extend(clean_bonafide_probs)

            # 测试攻击后的情况（负样本）
            attacked_bonafide_probs = []
            amp = self.atk_amps[attack_index]
            freq = self.atk_fs[attack_index]

            # 对每个文件进行多次测试以获得更多数据点
            for test_round in range(test_times_per_file):
                for _ in range(iterations):
                    try:
                        # Load audio with attack
                        audio_tensor = load_audio(file_path, atk_amp=amp, atk_f=freq, show_plot=False)
                        if audio_tensor is None:
                            attacked_bonafide_probs.append(0.5)  # 出错时给中性分数
                            continue
                        
                        # Judge spoof
                        _, _, _, spoof_prob = judge_spoof(self.model, audio_tensor, self.device)
                        bonafide_prob = 1 - spoof_prob  # bonafide probability
                        attacked_bonafide_probs.append(bonafide_prob)
                    except Exception as e:
                        print(f"处理攻击文件 {filename} 时出错: {e}")
                        attacked_bonafide_probs.append(0.5)

            y_scores_attacked.extend(attacked_bonafide_probs)

        return y_true, y_scores_clean, y_scores_attacked

    def plot_roc_curve(self, y_true: List, y_scores_clean: List,
                       y_scores_attacked: List[float], save_path: str = None):
        """
        绘制ROC曲线

        Args:
            y_true: 真实标签
            y_scores_clean: 未攻击音频的分数
            y_scores_attacked: 攻击后音频的分数
            save_path: 保存路径（可选）
        """
        # 合并数据
        y_scores_all = y_scores_clean + y_scores_attacked
        # 注意：这里不需要重新构建y_pred_all，因为我们已经有了正确的y_true
        # y_true已经按照[正样本..., 负样本...]的顺序构建

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores_all)
        roc_auc = auc(fpr, tpr)

        # 创建图形
        plt.figure(figsize=(4.5, 4))

        # 设置刻度坐标字号为12
        plt.tick_params(axis='both', which='major', labelsize=13)

        # 绘制ROC曲线
        plt.plot(fpr, tpr, color='darkorange', lw=2.5, marker='o', markersize=3,
                 label=f'ROC曲线(AUC = {roc_auc:.3f})', markevery=max(1, len(fpr) // 20))

        # 绘制对角线
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器(AUC = 0.5)')

        # 设置图形属性
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('误报率(攻击被误判为正常的比例)', fontsize=13)
        plt.ylabel('召回率(正常样本被正确识别的比例)', fontsize=13)
        # plt.title(f'活体检测模型在{self.dataset_name}数据集上的ROC曲线\n(未攻击=正样本, 攻击=负样本)', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=13)
        plt.grid(True, alpha=0.3)

        # 添加AUC文本框
        # plt.text(0.05, 0.75, f'AUC = {roc_auc:.4f}\n'
        #                   f'测试样本数: {len(y_true)}\n'
        #                   f'正样本(未攻击): {len(y_scores_clean)}\n'
        #                   f'负样本(攻击): {len(y_scores_attacked)}',
        #         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
        #         fontsize=10)

        # 移除重复的文本信息（已在上面添加）

        # 保存或显示图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

        return roc_auc, fpr, tpr, thresholds

    def print_statistics(self, y_true: List, y_scores_clean: List,
                         y_scores_attacked: List[float]):
        """打印统计信息"""
        print("\n=== ROC分析统计信息 ===")
        print(f"总样本数: {len(y_true)}")
        print(f"正样本数 (未攻击): {len(y_scores_clean)}")
        print(f"负样本数 (攻击): {len(y_scores_attacked)}")
        print(f"正样本平均bonafide概率: {np.mean(y_scores_clean):.4f} ± {np.std(y_scores_clean):.4f}")
        print(f"负样本平均bonafide概率: {np.mean(y_scores_attacked):.4f} ± {np.std(y_scores_attacked):.4f}")
        print(f"正样本最大bonafide概率: {np.max(y_scores_clean):.4f}")
        print(f"正样本最小bonafide概率: {np.min(y_scores_clean):.4f}")
        print(f"负样本最大bonafide概率: {np.max(y_scores_attacked):.4f}")
        print(f"负样本最小bonafide概率: {np.min(y_scores_attacked):.4f}")


def main():
    parser = argparse.ArgumentParser(description="绘制多数据集活体检测ROC曲线")
    parser.add_argument("--dataset",
                        dest="dataset",
                        type=str,
                        choices=["voxceleb", "aishell"],
                        help="数据集类型",
                        default="aishell")
    parser.add_argument("--audio_dir",
                        dest="audio_dir",
                        type=str,
                        required=False,
                        help="音频文件目录",
                        default=None)
    parser.add_argument("--model_path",
                        dest="model_path",
                        type=str,
                        help="模型权重路径",
                        default="./pretrained/Res_TSSDNet_time_frame_61_ASVspoof2019_LA_Loss_0.0017_dEER_0.74%_eEER_1.64%.pth")
    parser.add_argument("--model_type",
                        dest="model_type",
                        type=str,
                        choices=['SSDNet1D', 'SSDNet2D', 'DilatedNet'],
                        help="模型类型",
                        default="SSDNet1D")
    parser.add_argument("--device",
                        dest="device",
                        type=str,
                        help="计算设备 (cuda/cpu/mps)",
                        default="mps")
    parser.add_argument("--metadata",
                        dest="metadata",
                        type=str,
                        help="元数据文件路径（可选）",
                        default=None)
    parser.add_argument("--test_times",
                        dest="test_times",
                        type=int,
                        help="每个音频文件的测试次数",
                        default=20)
    parser.add_argument("--save_path",
                        dest="save_path",
                        type=str,
                        help="ROC曲线保存路径",
                        default="./figure/roc_curve.pdf")

    args = parser.parse_args()

    # 根据数据集设置默认音频目录
    if args.audio_dir is None:
        if args.dataset == "voxceleb":
            args.audio_dir = "/Users/jiangyancheng/Library/CloudStorage/OneDrive-个人/Ghost-SV/evaluation_audio/merged/VoxCeleb1/target_audio/"
        elif args.dataset == "aishell":
            args.audio_dir = "/Users/jiangyancheng/Library/CloudStorage/OneDrive-个人/Ghost-SV/evaluation_audio/merged/aishell/target_audio/"

    # 检查必要文件是否存在
    if not os.path.exists(args.audio_dir):
        print(f"错误: 音频目录不存在: {args.audio_dir}")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        sys.exit(1)

    # 创建输出目录
    if args.save_path:
        output_dir = os.path.dirname(args.save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    print(f"开始计算{args.dataset.upper()}数据集上的ROC曲线...")
    print(f"数据集: {args.dataset}")
    print(f"音频目录: {args.audio_dir}")
    print(f"模型路径: {args.model_path}")
    print(f"模型类型: {args.model_type}")
    print(f"设备: {args.device}")
    print(f"每个文件测试次数: {args.test_times}")

    # 初始化计算器
    calculator = DatasetROCCalculator(args.model_path, args.model_type, args.dataset, args.device)

    # 计算ROC点
    y_true, y_scores_clean, y_scores_attacked = calculator.calculate_roc_points(
        args.audio_dir, args.metadata, args.test_times)

    # 打印统计信息
    calculator.print_statistics(y_true, y_scores_clean, y_scores_attacked)

    # 绘制ROC曲线
    if len(y_true) > 0:
        auc_score, fpr, tpr, thresholds = calculator.plot_roc_curve(
            y_true, y_scores_clean, y_scores_attacked, args.save_path)
        print(f"\nROC AUC Score: {auc_score:.4f}")
    else:
        print("错误: 没有有效的测试数据")


if __name__ == "__main__":
    main()