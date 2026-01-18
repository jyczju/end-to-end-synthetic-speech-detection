"""
python swap_attack_param.py --amp_max 0.5 --freq_max 4000 --audio_path ./mydata/voxceleb/pair6/00001.wav

"""

import torch
import soundfile as sf
import numpy as np
import models
import argparse
import os
import librosa
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 使用黑体作为默认字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

def load_audio(file_path, expected_sr=16000, atk_amp=None, atk_f=None, show_plot=False):
    """
    Load audio file and ensure it has correct format for the model
    """
    try:
        # Load audio file
        sample, sr = sf.read(file_path)
        
        # Check sampling rate
        if sr != expected_sr:
            print(f"Warning: Audio file {file_path} has sampling rate {sr}, expected {expected_sr}")
            # Resample
            sample = librosa.resample(sample, orig_sr=sr, target_sr=expected_sr)
            print('Resampled to {} Hz'.format(expected_sr))

            
        # Convert to mono if stereo
        if len(sample.shape) > 1:
            sample = np.mean(sample, axis=1)

        # 归一化
        sample = sample / np.max(np.abs(sample))

        # 如果是攻击，则在已有音频上叠加幅值为atk_amp、频率为atk_f的正弦波
        if atk_amp is not None and atk_f is not None:
            sample = sample + atk_amp * np.sin(2 * np.pi * atk_f * np.arange(sample.shape[0]) / expected_sr)

            if show_plot:
                # 绘制音频波形图和时间频谱图在同一张图上
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                librosa.display.waveshow(sample, sr=expected_sr, ax=ax1)
                ax1.set_title('Waveform (with attack)')
                librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(sample)), ref=np.max),
                                         sr=expected_sr, x_axis='time', y_axis='log', ax=ax2)
                ax2.set_title('Spectrogram (with attack)')
                plt.tight_layout()
                plt.show()
            
        # Convert to tensor and add batch dimension
        sample = torch.tensor(sample, dtype=torch.float32)
        sample = torch.unsqueeze(sample, 0)  # Add channel dimension
        sample = torch.unsqueeze(sample, 0)  # Add batch dimension
        
        return sample
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

def judge_spoof(model, audio_tensor, device):
    """
    Judge if an audio sample is spoof or bonafide
    Returns: (prediction, confidence)
    """
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        audio_tensor = audio_tensor.to(device)
        output = model(audio_tensor)
        # print("output:")
        # print(output)
        
        # Apply softmax to get probabilities
        probabilities = torch.softmax(output, dim=1)
        # print("probabilities[bonafide, spoofed]:",probabilities)
        
        # Get prediction (0 = bonafide, 1 = spoof)
        prediction = torch.argmax(probabilities, dim=1).item()
        
        # Get confidence score
        confidence = torch.max(probabilities).item()
        
        # Return spoof probability specifically
        spoof_prob = probabilities[0][1].item()  # Probability of being spoof
        
        return prediction, confidence, spoof_prob

def plot_spoof_heatmap(spoof_prob_grid, amplitudes, frequencies):
    """
    Plot a heatmap of spoof probabilities for different amplitudes and frequencies
    """

    # 创建热图
    plt.figure(figsize=(4, 3), constrained_layout=True)
    im = plt.imshow(spoof_prob_grid, cmap='RdYlBu', interpolation='bilinear',
                    extent=[amplitudes[0], amplitudes[-1], frequencies[0], frequencies[-1]], 
                    aspect='auto', origin='lower')
    
    plt.colorbar(im, label='概率差值')
    plt.xlabel('攻击幅度')
    plt.ylabel('攻击频率(Hz)')
    plt.show()


def evaluate_parameters(model, audio_path, amplitudes, frequencies, device, model_type):
    """
    Evaluate spoof probabilities for different amplitude and frequency combinations
    """
    # 计算无攻击时的概率
    audio_tensor = load_audio(audio_path)
    # Pad or trim audio to appropriate length
    if model_type == 'SSDNet1D' or model_type == 'DilatedNet':
        target_length = 96000
        current_length = audio_tensor.shape[-1]

        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
        elif current_length > target_length:
            # Trim audio
            audio_tensor = audio_tensor[:, :, :target_length]
    _, _, base_spoof_prob = judge_spoof(model, audio_tensor, device)
    print(f"Base spoof prob: {base_spoof_prob:.4f}")

    spoof_prob_grid = np.zeros((len(amplitudes), len(frequencies)))
    
    for i in range(len(amplitudes)):
        for j in range(len(frequencies)):
            amp, freq = amplitudes[i], frequencies[j]

            # Load audio with specific attack parameters
            audio_tensor = load_audio(audio_path, atk_amp=amp, atk_f=freq)
            if audio_tensor is None:
                spoof_prob_grid[i, j] = np.nan
                continue
            
            # Pad or trim audio to appropriate length
            if model_type == 'SSDNet1D' or model_type == 'DilatedNet':
                target_length = 96000
                current_length = audio_tensor.shape[-1]
                
                if current_length < target_length:
                    # Pad with zeros
                    padding = target_length - current_length
                    audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
                elif current_length > target_length:
                    # Trim audio
                    audio_tensor = audio_tensor[:, :, :target_length]
            
            # Judge spoof
            _, _, spoof_prob = judge_spoof(model, audio_tensor, device)
            spoof_prob_grid[i, j] = spoof_prob - base_spoof_prob + 0.6
            print(f"Evaluated amp={amp}, freq={freq}, Spoof prob gap : {spoof_prob - base_spoof_prob:.4f} ,Spoof probability: {spoof_prob:.4f}")

    # print("Spoof probability grid.shape:")
    # print(spoof_prob_grid.shape)
    print("max:" , np.max(spoof_prob_grid))
    print("min:" , np.min(spoof_prob_grid))
    print("mean:" , np.mean(spoof_prob_grid))
    
    return spoof_prob_grid

def main():
    parser = argparse.ArgumentParser(description='Judge if an audio is spoof or bonafide')
    parser.add_argument('--audio_path', type=str, required=True, 
                        help='Path to the audio file to evaluate')
    parser.add_argument('--model_path', type=str, default='./pretrained/Res_TSSDNet_time_frame_61_ASVspoof2019_LA_Loss_0.0017_dEER_0.74%_eEER_1.64%.pth',
                        help='Path to the trained model')
    parser.add_argument('--model_type', type=str, default='SSDNet1D',
                        choices=['SSDNet1D', 'SSDNet2D', 'DilatedNet'],
                        help='Type of model to use')
    # Parameter ranges for heatmap
    parser.add_argument('--amp_min', type=float, default=0.0,
                       help='Minimum amplitude for sweep')
    parser.add_argument('--amp_max', type=float, default=0.5,
                        help='Maximum amplitude for sweep')
    parser.add_argument('--amp_steps', type=int, default=26,
                        help='Number of amplitude steps')
    parser.add_argument('--freq_min', type=float, default=0.0,
                        help='Minimum frequency for sweep (Hz)')
    parser.add_argument('--freq_max', type=float, default=20000.0,
                        help='Maximum frequency for sweep (Hz)')
    parser.add_argument('--freq_steps', type=int, default=26,
                        help='Number of frequency steps')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_path):
        print(f"Audio file not found: {args.audio_path}")
        return
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
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

    # Define parameter ranges
    amplitudes = np.linspace(args.amp_min, args.amp_max, args.amp_steps)
    frequencies = np.linspace(args.freq_min, args.freq_max, args.freq_steps)
    
    print(f"Analyzing spoof probabilities for {len(amplitudes)} amplitudes and {len(frequencies)} frequencies")
    
    # Evaluate parameters
    spoof_prob_grid = evaluate_parameters(model, args.audio_path, amplitudes, frequencies, device, args.model_type)
    #
    # print("spoof_prob_grid:")
    # print(spoof_prob_grid)
    
    # Plot heatmap
    plot_spoof_heatmap(spoof_prob_grid, amplitudes, frequencies)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()