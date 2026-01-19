import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import librosa

class MusicDataset(Dataset):
    """
    Dataset class for Music VAE (Legacy/Synthetic Wrapper).
    """
    def __init__(self, data, labels=None):
        if isinstance(data, tuple):
            # Hybrid mode: (audio, lyrics)
            self.data = data # Tuple of arrays
            self.is_tuple = True
            self.length = len(data[0])
        else:
            self.data = torch.FloatTensor(data)
            self.is_tuple = False
            self.length = len(data)
            
        self.labels = labels if labels is not None else None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.is_tuple:
            audio = torch.FloatTensor(self.data[0][idx])
            lyrics = torch.FloatTensor(self.data[1][idx])
            sample = (audio, lyrics)
        else:
            sample = self.data[idx]
            
        if self.labels is not None:
            return sample, self.labels[idx]
        return sample

class RealAudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=22050, n_mels=64, segment_length=64):
        """
        Args:
            root_dir (str): Path to folder containing audio files. 
                            Can be flat or 'class_name/file.mp3'
            sample_rate (int): SR for librosa loading
            n_mels (int): Number of Mel bands (height of image)
            segment_length (int): Number of time frames (width of image)
        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.segment_length = segment_length
        self.files = []
        self.labels = []
        self.class_map = {}
        
        # Walk through directory
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        if not classes:
            # Flat directory, treat as single class 'unknown' or just raw
            print(f"No subdirectories found in {root_dir}. Treating as unlabelled data.")
            self.files = glob.glob(os.path.join(root_dir, "*.*"))
            # Filter for audio extensions
            self.files = [f for f in self.files if f.lower().endswith(('.mp3', '.wav', '.flac', '.ogg'))]
            self.labels = [-1] * len(self.files)
        else:
            # Class directories
            for i, class_name in enumerate(classes):
                self.class_map[i] = class_name
                class_path = os.path.join(root_dir, class_name)
                class_files = glob.glob(os.path.join(class_path, "*.*"))
                class_files = [f for f in class_files if f.lower().endswith(('.mp3', '.wav', '.flac', '.ogg'))]
                
                self.files.extend(class_files)
                self.labels.extend([i] * len(class_files))
                
        print(f"Found {len(self.files)} audio files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        try:
            # Load audio
            # Load only a few seconds to speed up? No, need random crop. 
            # Better to pre-process, but for now we do on-the-fly.
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=30.0) # Cap at 30s
            
            # Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalization (roughly -80 to 0 dB -> 0 to 1)
            # A common strategy is (X - min) / (max - min)
            # Or simplified: X / 80 + 1 (if X is -80..0)
            norm_spec = (log_mel_spec + 80.0) / 80.0
            
            # Crop to segment
            if norm_spec.shape[1] < self.segment_length:
                # Pad if too short
                pad_width = self.segment_length - norm_spec.shape[1]
                norm_spec = np.pad(norm_spec, ((0,0), (0, pad_width)), mode='constant')
            else:
                # Random crop
                start = np.random.randint(0, norm_spec.shape[1] - self.segment_length)
                norm_spec = norm_spec[:, start:start+self.segment_length]
            
            # Ensure shape is [1, 64, 64]
            sample = torch.FloatTensor(norm_spec).unsqueeze(0)
            
            return sample, label
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zeros in case of error to not crash training
            return torch.zeros((1, self.n_mels, self.segment_length)), label

class SyntheticDataGenerator:
    """
    Generates synthetic music feature data for testing.
    Simulates clusters to verify VAE disentanglement/clustering capabilities.
    """
    def __init__(self, n_samples=1000, n_features=128, n_classes=5, mode='linear'):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.mode = mode

    def generate(self):
        if self.mode == 'linear':
            return self._generate_linear()
        elif self.mode in ['conv', 'resnet']: # ResNet uses same data shape as conv
            return self._generate_conv()
        elif self.mode == 'hybrid':
            return self._generate_hybrid()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _generate_linear(self):
        # Vectorized generation
        samples_per_class = self.n_samples // self.n_classes
        centers = np.random.uniform(-5, 5, size=(self.n_classes, self.n_features))
        
        # Create indices for all samples
        # shape: (n_samples,)
        y = np.repeat(np.arange(self.n_classes), samples_per_class)
        
        # Get centers for each sample
        sample_centers = centers[y] 
        
        # Add noise
        noise = np.random.normal(0, 1.0, size=(len(y), self.n_features))
        X = sample_centers + noise
        
        return self._shuffle(X, y)

    def _generate_conv(self):
        X = np.zeros((self.n_samples, 1, 64, 64), dtype=np.float32)
        y = np.repeat(np.arange(self.n_classes), self.n_samples // self.n_classes) 
        
        for i in range(self.n_samples):
            label = y[i]
            if label % 3 == 0: 
                row = np.random.randint(0, 60)
                X[i, 0, row:row+5, :] = 1.0
            elif label % 3 == 1: 
                col = np.random.randint(0, 60)
                X[i, 0, :, col:col+5] = 1.0
            else: 
                r, c = np.random.randint(0, 54, 2)
                X[i, 0, r:r+10, c:c+10] = 1.0
                
        X += np.random.normal(0, 0.1, X.shape)
        
        return self._shuffle(X, y)

    def _generate_hybrid(self):
        audio_dim = self.n_features
        lyrics_dim = 100
        samples_per_class = self.n_samples // self.n_classes
        y = np.repeat(np.arange(self.n_classes), samples_per_class)
        
        centers_audio = np.random.uniform(-5, 5, size=(self.n_classes, audio_dim))
        X_audio = centers_audio[y] + np.random.normal(0, 1.0, size=(len(y), audio_dim))
        
        centers_lyrics = np.random.uniform(-3, 3, size=(self.n_classes, lyrics_dim))
        X_lyrics = centers_lyrics[y] + np.random.normal(0, 1.0, size=(len(y), lyrics_dim))
        
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        
        return (X_audio[indices], X_lyrics[indices]), y[indices]

    def _shuffle(self, X, y):
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        return X[indices], y[indices]

def get_dataloader(batch_size=32, synthetic=True, mode='linear', num_workers=2, data_dir=None):
    if synthetic:
        generator = SyntheticDataGenerator(mode=mode)
        X, y = generator.generate()
        dataset = MusicDataset(X, y)
    else:
        if data_dir is None:
            raise ValueError("data_dir must be provided for real data")
        if mode not in ['conv', 'resnet']:
            raise ValueError("Real audio only supports 'conv' or 'resnet' mode currently")
            
        dataset = RealAudioDataset(root_dir=data_dir)
        X = None
        y = np.array(dataset.labels)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True), X, y
