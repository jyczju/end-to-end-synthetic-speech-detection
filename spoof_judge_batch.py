import argparse
import os
import torch
import re
from contextlib import redirect_stdout
import numpy as np

from tqdm import tqdm
import models

from spoof_judge import load_audio, judge_spoof


def main():
    parser = argparse.ArgumentParser(description='Judge if an audio is spoof or bonafide')
    parser.add_argument('--audio_path', type=str, default='/Users/jiangyancheng/Library/CloudStorage/OneDrive-个人/Ghost-SV/evaluation_audio/merged/aishell/target_audio/',
                        help='Path to the audio file to evaluate')
    parser.add_argument('--model_path', type=str,
                        default='./pretrained/Res_TSSDNet_time_frame_61_ASVspoof2019_LA_Loss_0.0017_dEER_0.74%_eEER_1.64%.pth',
                        help='Path to the trained model')
    parser.add_argument('--model_type', type=str, default='SSDNet1D',
                        choices=['SSDNet1D', 'SSDNet2D', 'DilatedNet'],
                        help='Type of model to use')
    parser.add_argument('--atk', type=bool, default=True)

    args = parser.parse_args()

    # Check if audio file exists
    if not os.path.exists(args.audio_path):
        print(f"Audio file not found: {args.audio_path}")
        return

    # Set device
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    if args.model_type == 'SSDNet1D':
        model = models.SSDNet1D()
    elif args.model_type == 'SSDNet2D':
        model = models.SSDNet2D()
    elif args.model_type == 'DilatedNet':
        model = models.DilatedNet()
    else:
        print(f"Unknown model type: {args.model_type}")
        return

    # Load model weights
    if os.path.exists(args.model_path):
        try:
            checkpoint = torch.load(args.model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Model loaded from {args.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print(f"Model file not found: {args.model_path}")
        print("Please provide a valid model path with --model_path")
        return

    atk_amps = []
    atk_fs = []
    if args.atk:
        print("Attack")
        if "aishell" in args.audio_path:
            atk_amps = [0.0581, 0.0648, 0.036, 0.2922, 0.1546, 0.0095, 0.0573, 0.0555, 0.0436, 0.3988, 0.5436, 0.1017,
                        0.41, 0.36, 0.1337, 0.5293, 0.404, 0.3726, 0.505, 0.5127]
            atk_fs = [3671.06, 4592.98, 943.95, 3542.28, 4954.2, 2133, 636.12, 1440.66, 332.77, 696.97, 1941.43,
                      4013.25, 2386.69, 1949.86, 1425.04, 2981.95, 2586.65, 1141.28, 2659.63, 4781.89]
        if "VoxCeleb" in args.audio_path:
            atk_amps = [0.5, 0.5, 0.3966, 0.1178, 0.44, 0.5, 0.5, 0.3378, 0.5, 0.1344, 0.4641, 0.119, 0.481, 0.3819,
                        0.2124, 0.1794, 0.3569, 0.2895, 0.3477, 0.4853]
            atk_fs = [1999.99, 10000, 7060.15, 6583.37, 9498.15, 3347.5, 3100.75, 4320.05, 5000, 1074.48, 1468.86,
                      6159.21, 2667.74, 3018.91, 618.74, 821.02, 3867.59, 1217.95, 614.54, 3976.73]

    spoof_probs = []
    spoof_flags = []
    # 遍历audio_path下的所有文件
    for file in os.listdir(args.audio_path):
        if not file.endswith(".wav"):
            continue
        # 提取文件名中的数字
        match = re.search(r'\d+', file)
        if match:
            number = int(match.group())
            print(f"File: {file}, Number: {number}")
        else:
            print(f"File: {file}, No number found")



        # 循环100次，记录平均Spoof probability
        for i in tqdm(range(75)):
            with redirect_stdout(open(os.devnull, 'w')):
                # Load audio
                if args.atk:
                    audio_tensor = load_audio(args.audio_path + file, atk_amp=atk_amps[number - 1],
                                              atk_f=atk_fs[number - 1], show_plot=False)
                else:
                    audio_tensor = load_audio(args.audio_path + file, show_plot=False)
                if audio_tensor is None:
                    print("Error loading audio file")
                    continue
                # Judge spoof
                prediction, confidence, is_spoof, spoof_prob = judge_spoof(model, audio_tensor, device)
                spoof_probs.append(spoof_prob)
                spoof_flags.append(1 if is_spoof else 0)
    print("Average Spoof probability: {:.4f}".format(np.mean(spoof_probs)))
    print("Average Spoof percent: {:.4f}".format(np.sum(spoof_flags) / len(spoof_flags)))


if __name__ == "__main__":
    main()