import numpy as np
import librosa
import os
import pandas as pd

def uniform_duration(sr, audio):
    target_length = 4 * sr

    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
    else:
        audio = audio[:target_length]
    return audio

def extract_features(audio_file):
    audio, sr = librosa.load(audio_file)

    n_fft = 1024
    if len(audio) < 1024:
        n_fft = 256
    '''print(n_fft)
    print(audio.shape[-1])'''
    
   
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=1, n_fft=n_fft)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)

    rmse = librosa.feature.rms(y=audio)  
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, roll_percent=0.85)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr,  n_fft=n_fft)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr,  n_fft=n_fft)
    
    fold = next(part for part in audio_file.split(os.sep) if "fold" in part)

    file_id = os.path.basename(audio_file).split('.')[0]  
    features = [
        file_id, fold,
        np.mean(mfccs), np.std(mfccs), 
        np.mean(chroma), np.std(chroma), 
        np.mean(spectral_contrast), np.std(spectral_contrast), 
        np.mean(tonnetz), np.std(tonnetz), 
        np.mean(zero_crossing_rate), np.std(zero_crossing_rate), 
        np.mean(rmse), np.std(rmse),
        np.mean(spectral_rolloff), np.std(spectral_rolloff),
        np.mean(centroid), np.std(centroid),
        np.mean(bandwidth), np.std(bandwidth),
        np.mean(contrast), np.std(contrast),
        np.mean(mel_spectrogram), np.std(mel_spectrogram)
    ]

    return features

def feature_extraction(audio_files, features_list, columns):
    i = 0
    for file_path in audio_files:
        features = extract_features(file_path)
        features_list.append(features)  
        i+=1
        if (i%100)==0:
            print(f"{i} iterações realizadas")

   
    if not features_list or len(features_list[0]) != len(columns):
        raise ValueError(f"Features list não corresponde ao número de colunas. "
                         f"Esperado {len(columns)}, mas encontrou {len(features_list[0])}.")

    
    df = pd.DataFrame(features_list, columns=columns)
    df.to_csv("feature_extraction.csv", index=False)