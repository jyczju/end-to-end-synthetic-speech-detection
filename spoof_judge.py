
import torch
import soundfile as sf
import numpy as np
import models
import argparse
import os
import librosa
import matplotlib.pyplot as plt


def load_audio(file_path, expected_sr=16000, atk_amp=None, atk_f=None, show_plot=True):
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

        # 随机裁剪（其实点在0～sample.shape[0]-6*sr之间随机）
        sample_len = sample.shape[0]
        max_len = int(6 * sr)
        if sample_len >= max_len:
            # 起点从0～x_len-max_len之间进行取值，取值范围是0～x_len-max_len
            stt = np.random.randint(sample_len - max_len)
            sample = sample[stt:stt + max_len]

        # 归一化
        sample = sample / np.max(np.abs(sample))

        # 如果是攻击，则在已有音频上叠加幅值为atk_amp、频率为atk_f的正弦波
        if atk_amp is not None and atk_f is not None:
            sample = sample + atk_amp * np.sin(2 * np.pi * atk_f * np.arange(sample.shape[0]) / expected_sr)

        if show_plot:
            # 绘制音频波形图和时间频谱图在同一张图上
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            librosa.display.waveshow(sample, sr=expected_sr, ax=ax1)
            ax1.set_title('Waveform')
            librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(sample)), ref=np.max),
                                     sr=expected_sr, x_axis='time', y_axis='log', ax=ax2)
            ax2.set_title('Spectrogram')
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
        print("probabilities[bonafide, spoofed]:",probabilities)
        
        # Get prediction (0 = bonafide, 1 = spoof)
        prediction = torch.argmax(probabilities, dim=1).item()
        
        # Get confidence score
        confidence = torch.max(probabilities).item()

        is_spoof = prediction == 1

        spoof_prob = probabilities[0][1].item()
        
        return prediction, confidence, is_spoof, spoof_prob

def main():
    parser = argparse.ArgumentParser(description='Judge if an audio is spoof or bonafide')
    parser.add_argument('--audio_path', type=str, default='mydata/aishell/attacker_audio/pair1.wav',
                        help='Path to the audio file to evaluate')
    parser.add_argument('--model_path', type=str, default='./pretrained/Res_TSSDNet_time_frame_61_ASVspoof2019_LA_Loss_0.0017_dEER_0.74%_eEER_1.64%.pth',
                        help='Path to the trained model')
    parser.add_argument('--model_type', type=str, default='SSDNet1D',
                        choices=['SSDNet1D', 'SSDNet2D', 'DilatedNet'],
                        help='Type of model to use')
    parser.add_argument('--atk_amp', type=float, default=None,
                        help='Amplitude of the attack waveform')
    parser.add_argument('--atk_f', type=float, default=None,
                        help='Frequency of the attack waveform')
    
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

    print("Begin to load audio")
    # Load audio
    audio_tensor = load_audio(args.audio_path, atk_amp=args.atk_amp, atk_f=args.atk_f, show_plot=True)
    if audio_tensor is None:
        return
    
    # Pad or trim audio to appropriate length (96000 samples for SSDNet1D)
    if args.model_type == 'SSDNet1D' or args.model_type == 'DilatedNet':
        target_length = 96000
        current_length = audio_tensor.shape[-1]
        
        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
            print(f"Audio padded with {padding} zeros")
        elif current_length > target_length:
            # Trim audio
            audio_tensor = audio_tensor[:, :, :target_length]
            print(f"Audio trimmed to {target_length} samples")


    
    print(f"Processing audio file: {args.audio_path}")
    
    # Judge spoof
    prediction, confidence, is_spoof, spoof_prob = judge_spoof(model, audio_tensor, device)
    
    # Output result
    label = "spoof" if prediction == 1 else "bonafide"
    print(f"\nResult:")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()