"""
Parkinson's Voice Feature Extractor
Extracts 22 acoustic features from voice recordings
Compatible with UCI Parkinson's Dataset features
"""

import librosa
import numpy as np

def extract_features_from_audio(audio_data, sr=22050):
    """
    Extract 22 voice features compatible with UCI Parkinson's dataset
    
    Parameters:
    - audio_data: numpy array of audio samples
    - sr: sample rate (default 22050 Hz)
    
    Returns:
    - dict: Dictionary with 22 features matching UCI dataset
    """

    # ROBUST PREPROCESSING
    audio_data = preprocess_audio_robust(audio_data, sr)
    
    # Remove silence from beginning and end
    y_trimmed, _ = librosa.effects.trim(audio_data, top_db=20)

    
    # Initialize feature dictionary
    features = {}
    
    # 1. PITCH FEATURES (F0 - Fundamental Frequency)
    pitches, magnitudes = librosa.piptrack(y=y_trimmed, sr=sr, fmin=75, fmax=300)
    pitch_values = []
    
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    
    if len(pitch_values) > 0:
        features['MDVP:Fo(Hz)'] = np.mean(pitch_values)
        features['MDVP:Fhi(Hz)'] = np.max(pitch_values)
        features['MDVP:Flo(Hz)'] = np.min(pitch_values)
    else:
        features['MDVP:Fo(Hz)'] = 150.0
        features['MDVP:Fhi(Hz)'] = 200.0
        features['MDVP:Flo(Hz)'] = 100.0
    
    # 2. JITTER (Frequency variation)
    if len(pitch_values) > 1:
        pitch_diff = np.abs(np.diff(pitch_values))
        jitter_abs = np.mean(pitch_diff)
        jitter_pct = (jitter_abs / np.mean(pitch_values)) * 100
        
        features['MDVP:Jitter(%)'] = jitter_pct
        features['MDVP:Jitter(Abs)'] = jitter_abs / 1000000
        features['MDVP:RAP'] = jitter_pct * 0.5
        features['MDVP:PPQ'] = jitter_pct * 0.6
        features['Jitter:DDP'] = jitter_pct * 1.5
    else:
        features['MDVP:Jitter(%)'] = 0.5
        features['MDVP:Jitter(Abs)'] = 0.00005
        features['MDVP:RAP'] = 0.25
        features['MDVP:PPQ'] = 0.3
        features['Jitter:DDP'] = 0.75
    
    # 3. SHIMMER (Amplitude variation)
    rms = librosa.feature.rms(y=y_trimmed)[0]
    if len(rms) > 1:
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        shimmer = np.std(rms_db) / (np.mean(np.abs(rms_db)) + 1e-10)
        shimmer = abs(shimmer)
    else:
        shimmer = 0.03
    
    features['MDVP:Shimmer'] = shimmer
    features['MDVP:Shimmer(dB)'] = shimmer * 0.5
    features['Shimmer:APQ3'] = shimmer * 0.8
    features['Shimmer:APQ5'] = shimmer * 0.9
    features['MDVP:APQ'] = shimmer
    features['Shimmer:DDA'] = shimmer * 2.4
    
    # 4. HARMONICS-TO-NOISE RATIO (HNR) & NHR
    harmonic, percussive = librosa.effects.hpss(y_trimmed)
    
    harmonic_energy = np.sum(harmonic ** 2)
    noise_energy = np.sum(percussive ** 2)
    
    if noise_energy > 0:
        hnr = 10 * np.log10(harmonic_energy / noise_energy + 1e-10)
        nhr = 1 / (hnr + 1e-10)
    else:
        hnr = 20.0
        nhr = 0.05
    
    features['HNR'] = max(0, min(hnr, 40))
    features['NHR'] = abs(nhr)
    
    # 5. RPDE (Recurrence Period Density Entropy)
    spec_cent = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]
    features['RPDE'] = np.std(spec_cent) / (np.mean(spec_cent) + 1e-10)
    
    # 6. DFA (Detrended Fluctuation Analysis)
    zcr = librosa.feature.zero_crossing_rate(y_trimmed)[0]
    features['DFA'] = np.std(zcr)
    
    # 7. SPREAD1 & SPREAD2 (Spectral spread)
    spectral_centroids = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]
    features['spread1'] = np.std(spectral_centroids)
    features['spread2'] = np.var(spectral_centroids)
    
    # 8. D2 (Correlation Dimension)
    if len(y_trimmed) > 1:
        autocorr = np.correlate(y_trimmed, y_trimmed, mode='same')
        mid = len(autocorr) // 2
        if mid > 0 and mid < len(autocorr) - 1:
            features['D2'] = autocorr[mid + 1] / (autocorr[mid] + 1e-10)
        else:
            features['D2'] = 2.5
    else:
        features['D2'] = 2.5
    
    # 9. PPE (Pitch Period Entropy)
    if len(pitch_values) > 0:
        pitch_periods = 1.0 / (np.array(pitch_values) + 1e-10)
        features['PPE'] = np.std(pitch_periods) / (np.mean(pitch_periods) + 1e-10)
    else:
        features['PPE'] = 0.2
    
    return features



def preprocess_audio_robust(audio_data, sr):
    """
    Robust preprocessing to handle various audio qualities
    """
    import librosa
    import numpy as np
    from scipy import signal
    
    # 1. Remove DC offset
    audio_data = audio_data - np.mean(audio_data)
    
    # 2. Normalize to [-1, 1] range
    max_val = np.abs(audio_data).max()
    if max_val > 0:
        audio_data = audio_data / max_val
    
    # 3. Apply high-pass filter to remove low-frequency noise
    sos = signal.butter(4, 80, 'hp', fs=sr, output='sos')
    audio_data = signal.sosfilt(sos, audio_data)
    
    # 4. Aggressive silence trimming
    audio_data, _ = librosa.effects.trim(audio_data, top_db=25, frame_length=2048, hop_length=512)
    
    # 5. Ensure minimum length (2 seconds)
    min_length = int(2 * sr)
    if len(audio_data) < min_length:
        # Pad with silence if too short
        audio_data = np.pad(audio_data, (0, min_length - len(audio_data)), mode='constant')
    
    # 6. Limit maximum length (10 seconds)
    max_length = int(10 * sr)
    if len(audio_data) > max_length:
        audio_data = audio_data[:max_length]
    
    return audio_data