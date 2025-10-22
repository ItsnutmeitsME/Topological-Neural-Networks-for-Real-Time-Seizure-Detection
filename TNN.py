!pip install gudhi ripser persim pyedflib
!pip install scikit-learn==1.2.2 imbalanced-learn==0.10.1

import os
import numpy as np
import pandas as pd
import pyedflib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, hilbert, coherence
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
import pickle
import gc
import random
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

try:
    import gudhi
    from ripser import ripser, Rips
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False

torch.backends.cudnn.benchmark = True


class TopologicalEEGProcessor:
    
    def __init__(self, base_path, target_fs=128):
        self.base_path = base_path
        self.target_fs = target_fs
        self.seizure_files = self._load_seizure_files()
        self.all_files = self._load_all_files()
        
    def _load_seizure_files(self):
        seizure_files = set()
        records_file = os.path.join(self.base_path, 'RECORDS-WITH-SEIZURES')
        if os.path.exists(records_file):
            with open(records_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        seizure_files.add(line)
        return seizure_files
    
    def _load_all_files(self):
        all_files = set()
        records_file = os.path.join(self.base_path, 'RECORDS')
        if os.path.exists(records_file):
            with open(records_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        all_files.add(line)
        return all_files
        
    def get_patient_files(self, patient_id, max_seizure_files=2, max_normal_files=3):
        patient_prefix = f'chb{patient_id:02d}/'
        seizure_files = [f for f in self.seizure_files if f.startswith(patient_prefix)]
        all_patient_files = [f for f in self.all_files if f.startswith(patient_prefix)]
        non_seizure_files = [f for f in all_patient_files if f not in self.seizure_files]
        
        selected_seizure = seizure_files[:max_seizure_files]
        selected_normal = non_seizure_files[:max_normal_files]
        
        return selected_seizure, selected_normal
    
    def parse_summary_file(self, patient_dir):
        summary_files = [f for f in os.listdir(patient_dir) if f.endswith('-summary.txt')]
        if not summary_files:
            return {}
        
        summary_file = os.path.join(patient_dir, summary_files[0])
        seizure_info = {}
        
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
            
            sections = content.split('File Name:')[1:]
            for section in sections:
                lines = section.strip().split('\n')
                if not lines:
                    continue
                    
                file_name = lines[0].strip()
                seizures = []
                
                for line in lines:
                    if 'seizure' in line.lower() and ('start' in line.lower() or 'onset' in line.lower()):
                        parts = line.split()
                        times = [float(p) for p in parts if p.replace('.', '').replace('-', '').isdigit()]
                        if len(times) >= 2:
                            seizures.append((int(times[0]), int(times[1])))
                
                if seizures:
                    seizure_info[file_name] = {'seizures': seizures}
                    
        except Exception as e:
            pass
            
        return seizure_info
    
    def load_edf_optimized(self, filepath, max_duration=240, target_channels=8):
        try:
            f = pyedflib.EdfReader(filepath)
            signal_labels = f.getSignalLabels()
            original_fs = f.getSampleFrequency(0)
            
            eeg_priorities = ['C3', 'C4', 'F3', 'F4', 'P3', 'P4', 'O1', 'O2', 
                            'FP1', 'FP2', 'T3', 'T4', 'T5', 'T6', 'F7', 'F8']
            exclude_keywords = ['ECG', 'EMG', 'EOG', 'RESP', 'PLETH']
            
            eeg_indices = []
            for priority_channel in eeg_priorities:
                for i, label in enumerate(signal_labels):
                    if (priority_channel in label.upper() and 
                        not any(ex in label.upper() for ex in exclude_keywords) and
                        i not in eeg_indices):
                        eeg_indices.append(i)
                        if len(eeg_indices) >= target_channels:
                            break
                if len(eeg_indices) >= target_channels:
                    break
            
            if len(eeg_indices) < 4:
                f.close()
                return None, None, None
            
            max_samples = min(f.getNSamples()[0], int(max_duration * original_fs))
            signals = np.zeros((len(eeg_indices), max_samples))
            
            for i, idx in enumerate(eeg_indices):
                signal_data = f.readSignal(idx)[:max_samples]
                signals[i, :len(signal_data)] = signal_data
            
            f.close()
            
            if original_fs != self.target_fs:
                from scipy.signal import resample
                new_length = int(max_samples * self.target_fs / original_fs)
                signals = resample(signals, new_length, axis=1)
            
            for i in range(signals.shape[0]):
                signals[i] = self._bandpass_filter(signals[i], self.target_fs)
            
            selected_labels = [signal_labels[i] for i in eeg_indices]
            return signals, self.target_fs, selected_labels
            
        except Exception as e:
            return None, None, None
    
    def _bandpass_filter(self, signal, fs, lowcut=0.5, highcut=30):
        try:
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = min(highcut / nyquist, 0.99)
            b, a = butter(4, [low, high], btype='band')
            return filtfilt(b, a, signal)
        except:
            return signal


class TopologicalFeatureExtractor:
    
    def __init__(self, window_size=256, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap
        self.step = int(window_size * (1 - overlap))
        
    def extract_windows_with_topology(self, data, sample_freq, seizure_times=None, max_windows=80):
        windows = []
        labels = []
        topological_features = []
        
        for i in range(0, data.shape[1] - self.window_size + 1, self.step):
            window = data[:, i:i + self.window_size]
            
            window_start_sec = i / sample_freq
            window_end_sec = (i + self.window_size) / sample_freq
            
            is_seizure = False
            if seizure_times:
                for start_sec, end_sec in seizure_times:
                    overlap_start = max(window_start_sec, start_sec)
                    overlap_end = min(window_end_sec, end_sec)
                    if overlap_end > overlap_start:
                        overlap_ratio = (overlap_end - overlap_start) / (window_end_sec - window_start_sec)
                        if overlap_ratio > 0.4:
                            is_seizure = True
                            break
            
            topo_features = self._extract_topological_features(window, sample_freq)
            
            windows.append(window)
            labels.append(1 if is_seizure else 0)
            topological_features.append(topo_features)
            
            if len(windows) >= max_windows:
                break
        
        seizure_indices = [i for i, label in enumerate(labels) if label == 1]
        normal_indices = [i for i, label in enumerate(labels) if label == 0]
        
        if len(seizure_indices) > 0 and len(normal_indices) > 0:
            max_seizures = min(len(seizure_indices), max_windows // 8)
            max_normals = min(len(normal_indices), max_seizures * 7)
            
            selected_seizure_idx = np.random.choice(seizure_indices, max_seizures, replace=False)
            selected_normal_idx = np.random.choice(normal_indices, max_normals, replace=False)
            
            selected_indices = np.concatenate([selected_seizure_idx, selected_normal_idx])
            np.random.shuffle(selected_indices)
            
            final_windows = [windows[i] for i in selected_indices]
            final_labels = [labels[i] for i in selected_indices]
            final_features = [topological_features[i] for i in selected_indices]
            
            return np.array(final_windows), np.array(final_labels), np.array(final_features)
        
        return np.array(windows), np.array(labels), np.array(topological_features)
    
    def _extract_topological_features(self, window, sample_freq):
        features = []
        n_channels = window.shape[0]
        
        distance_matrices = self._compute_distance_matrices(window)
        
        for dist_name, dist_matrix in distance_matrices.items():
            diagrams = self._compute_persistent_homology(dist_matrix)
            topo_features = self._extract_persistence_features(diagrams)
            features.extend(topo_features)
        
        graph_features = self._extract_graph_topology_features(window)
        features.extend(graph_features)
        
        spectral_features = self._extract_spectral_topology_features(window, sample_freq)
        features.extend(spectral_features)
        
        return np.array(features)
    
    def _compute_distance_matrices(self, window):
        matrices = {}
        n_channels = window.shape[0]
        
        corr_matrix = np.corrcoef(window)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        matrices['correlation'] = 1 - np.abs(corr_matrix)
        np.fill_diagonal(matrices['correlation'], 0)
        
        if window.shape[1] > 64:
            step = max(1, window.shape[1] // 64)
            window_ds = window[:, ::step]
        else:
            window_ds = window
            
        try:
            euclidean_dist = squareform(pdist(window_ds))
            max_dist = np.max(euclidean_dist)
            matrices['euclidean'] = euclidean_dist / (max_dist + 1e-8)
        except:
            matrices['euclidean'] = matrices['correlation']
        
        try:
            analytic_signals = hilbert(window, axis=1)
            phases = np.angle(analytic_signals)
            phase_matrix = np.zeros((n_channels, n_channels))
            
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    phase_diff = phases[i] - phases[j]
                    pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                    phase_matrix[i, j] = phase_matrix[j, i] = 1 - pli
                    
            matrices['phase'] = phase_matrix
        except:
            matrices['phase'] = matrices['correlation']
        
        try:
            coherence_matrix = np.zeros((n_channels, n_channels))
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    f, coh = coherence(window[i], window[j], fs=128, nperseg=min(64, window.shape[1]//4))
                    avg_coh = np.mean(coh)
                    coherence_matrix[i, j] = coherence_matrix[j, i] = 1 - avg_coh
            matrices['coherence'] = coherence_matrix
        except:
            matrices['coherence'] = matrices['correlation']
            
        return matrices
    
    def _compute_persistent_homology(self, distance_matrix, max_dim=1):
        try:
            if TDA_AVAILABLE and distance_matrix.shape[0] <= 16:
                rips = Rips(maxdim=max_dim, thresh=1.0)
                diagrams = rips.fit_transform(distance_matrix, distance_matrix=True)
                return diagrams
            else:
                return self._simplified_topology(distance_matrix)
        except:
            return self._simplified_topology(distance_matrix)
    
    def _simplified_topology(self, distance_matrix):
        threshold_range = np.linspace(0.1, 0.9, 10)
        features_0d = []
        features_1d = []
        
        for thresh in threshold_range:
            adj_matrix = (distance_matrix <= thresh).astype(int)
            np.fill_diagonal(adj_matrix, 0)
            
            from scipy.sparse.csgraph import connected_components
            n_components, labels = connected_components(adj_matrix)
            features_0d.append([thresh, float('inf'), n_components])
            
            n_triangles = 0
            n = adj_matrix.shape[0]
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if adj_matrix[i,j] and adj_matrix[j,k] and adj_matrix[k,i]:
                            n_triangles += 1
            
            if n_triangles > 0:
                features_1d.append([thresh, thresh + 0.1, n_triangles])
        
        return [np.array(features_0d), np.array(features_1d)]
    
    def _extract_persistence_features(self, diagrams):
        features = []
        
        for dim, diagram in enumerate(diagrams):
            if len(diagram) == 0:
                features.extend([0] * 12)
                continue
            
            births = diagram[:, 0]
            deaths = diagram[:, 1]
            
            finite_mask = np.isfinite(deaths)
            lifetimes = deaths - births
            finite_lifetimes = lifetimes[finite_mask]
            
            feat_dim = [
                len(diagram),
                np.max(finite_lifetimes) if len(finite_lifetimes) > 0 else 0,
                np.mean(finite_lifetimes) if len(finite_lifetimes) > 0 else 0,
                np.std(finite_lifetimes) if len(finite_lifetimes) > 0 else 0,
                np.sum(finite_lifetimes) if len(finite_lifetimes) > 0 else 0,
                np.sum(~finite_mask),
                np.percentile(finite_lifetimes, 75) if len(finite_lifetimes) > 0 else 0,
                np.percentile(finite_lifetimes, 25) if len(finite_lifetimes) > 0 else 0,
                entropy(finite_lifetimes + 1e-8) if len(finite_lifetimes) > 1 else 0,
                np.mean(births) if len(births) > 0 else 0,
                np.std(births) if len(births) > 0 else 0,
                len(finite_lifetimes) / len(diagram) if len(diagram) > 0 else 0
            ]
            features.extend(feat_dim)
        
        return features
    
    def _extract_graph_topology_features(self, window):
        n_channels = window.shape[0]
        
        corr_matrix = np.corrcoef(window)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        features = []
        
        for thresh in [0.3, 0.5, 0.7, 0.9]:
            adj_matrix = (np.abs(corr_matrix) > thresh).astype(int)
            np.fill_diagonal(adj_matrix, 0)
            
            degree_sequence = np.sum(adj_matrix, axis=1)
            
            features.extend([
                np.sum(adj_matrix) / 2,
                np.mean(degree_sequence),
                np.std(degree_sequence),
                np.max(degree_sequence),
                np.sum(degree_sequence == 0),
            ])
        
        return features
    
    def _extract_spectral_topology_features(self, window, fs):
        features = []
        
        for ch in range(min(4, window.shape[0])):
            signal = window[ch]
            
            freqs = np.fft.fftfreq(len(signal), 1/fs)[:len(signal)//2]
            psd = np.abs(np.fft.fft(signal)[:len(signal)//2])**2
            
            delta = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])
            theta = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
            alpha = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
            beta = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
            
            total_power = delta + theta + alpha + beta + 1e-8
            
            features.extend([
                delta/total_power,
                theta/total_power, 
                alpha/total_power,
                beta/total_power,
                alpha/(delta + beta + 1e-8),
                (theta + alpha)/(delta + beta + 1e-8)
            ])
        
        while len(features) < 24:
            features.append(0.0)
            
        return features[:24]


class TopologicalMessagePassing(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, message_type='standard'):
        super(TopologicalMessagePassing, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.message_type = message_type
        
        self.message_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.aggregation = nn.Linear(hidden_dim, hidden_dim)
        
        self.update_mlp = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        if message_type == 'attention':
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
    
    def forward(self, x, edge_index=None, neighborhood_matrix=None):
        batch_size, seq_len, feature_dim = x.shape
        
        messages = []
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    message_input = torch.cat([x[:, i, :], x[:, j, :]], dim=-1)
                    message = self.message_mlp(message_input)
                    messages.append(message.unsqueeze(1))
        
        if messages:
            all_messages = torch.cat(messages, dim=1)
            
            if self.message_type == 'attention':
                aggregated_messages, _ = self.attention(all_messages, all_messages, all_messages)
                aggregated_messages = torch.mean(aggregated_messages, dim=1)
            else:
                aggregated_messages = torch.mean(all_messages, dim=1)
            
            final_messages = self.aggregation(aggregated_messages)
            
            updated_features = []
            for i in range(seq_len):
                update_input = torch.cat([x[:, i, :], final_messages], dim=-1)
                updated_feature = self.update_mlp(update_input)
                updated_features.append(updated_feature.unsqueeze(1))
            
            return torch.cat(updated_features, dim=1)
        else:
            return x


class TopologicalNeuralNetwork(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.3):
        super(TopologicalNeuralNetwork, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        
        self.topo_layers = nn.ModuleList([
            TopologicalMessagePassing(hidden_dim, hidden_dim, 'standard'),
            TopologicalMessagePassing(hidden_dim, hidden_dim, 'attention'),
            TopologicalMessagePassing(hidden_dim, hidden_dim, 'standard')
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.layer_norm1(x)
        x = F.relu(x)
        
        for i, (topo_layer, layer_norm) in enumerate(zip(self.topo_layers, self.layer_norms)):
            residual = x
            x = topo_layer(x)
            x = layer_norm(x + residual)
            x = F.relu(x)
            x = self.dropout(x)
        
        lstm_out, _ = self.lstm(x)
        
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        pooled = torch.mean(attn_out, dim=1)
        
        output = self.classifier(pooled)
        
        return output


class TopologicalDataset(Dataset):
    
    def __init__(self, features, labels, sequence_length=8):
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        
        self.sequences = []
        self.sequence_labels = []
        
        for i in range(len(features) - sequence_length + 1):
            seq_features = features[i:i + sequence_length]
            seq_label_votes = labels[i:i + sequence_length]
            seq_label = 1 if np.any(seq_label_votes == 1) else 0
            
            self.sequences.append(seq_features)
            self.sequence_labels.append(seq_label)
        
        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.sequence_labels = torch.LongTensor(np.array(self.sequence_labels))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.sequence_labels[idx]


class TopologicalSeizureDetectionPipeline:
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.processor = TopologicalEEGProcessor(base_path)
        self.feature_extractor = TopologicalFeatureExtractor()
        self.scaler = RobustScaler()
        self.model = None
        
    def load_multi_patient_data(self, patient_ids, max_seizure_files=1, max_normal_files=3):
        all_features = []
        all_labels = []
        patient_summaries = []
        
        for patient_id in patient_ids:
            seizure_files, normal_files = self.processor.get_patient_files(
                patient_id, max_seizure_files, max_normal_files
            )
            
            if not seizure_files and not normal_files:
                continue
            
            patient_dir = os.path.join(self.base_path, f'chb{patient_id:02d}')
            if not os.path.exists(patient_dir):
                continue
                
            seizure_info = self.processor.parse_summary_file(patient_dir)
            
            patient_windows = []
            patient_labels = []
            patient_topo_features = []
            
            for seizure_file in seizure_files:
                file_path = os.path.join(self.base_path, seizure_file)
                if not os.path.exists(file_path):
                    continue
                
                eeg_data, sample_freq, _ = self.processor.load_edf_optimized(file_path)
                if eeg_data is None:
                    continue
                
                file_name = os.path.basename(seizure_file)
                seizure_times = seizure_info.get(file_name, {}).get('seizures', [])
                
                windows, labels, topo_features = self.feature_extractor.extract_windows_with_topology(
                    eeg_data, sample_freq, seizure_times, max_windows=60
                )
                
                if len(windows) > 0:
                    patient_windows.extend(windows)
                    patient_labels.extend(labels)
                    patient_topo_features.extend(topo_features)
                
                del eeg_data
                gc.collect()
            
            for normal_file in normal_files:
                file_path = os.path.join(self.base_path, normal_file)
                if not os.path.exists(file_path):
                    continue
                
                eeg_data, sample_freq, _ = self.processor.load_edf_optimized(file_path)
                if eeg_data is None:
                    continue
                
                windows, labels, topo_features = self.feature_extractor.extract_windows_with_topology(
                    eeg_data, sample_freq, None, max_windows=50
                )
                
                if len(windows) > 0:
                    patient_windows.extend(windows)
                    patient_labels.extend(labels)
                    patient_topo_features.extend(topo_features)
                
                del eeg_data
                gc.collect()
            
            if patient_windows:
                all_features.extend(patient_topo_features)
                all_labels.extend(patient_labels)
                
                patient_summaries.append({
                    'patient_id': patient_id,
                    'total_windows': len(patient_windows),
                    'seizure_ratio': np.mean(patient_labels),
                    'files_processed': len(seizure_files) + len(normal_files)
                })
        
        if not all_features:
            return None, None, None
        
        return np.array(all_features), np.array(all_labels), patient_summaries
    
    def train_topological_model(self, features, labels, sequence_length=6, 
                               batch_size=16, epochs=80, learning_rate=0.001):
        
        valid_mask = np.isfinite(features).all(axis=1) & np.isfinite(labels)
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        if len(unique_labels) < 2:
            return None, 0
        
        features_scaled = self.scaler.fit_transform(features)
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.3, random_state=42,
            stratify=labels
        )
        
        train_dataset = TopologicalDataset(X_train, y_train, sequence_length)
        test_dataset = TopologicalDataset(X_test, y_test, sequence_length)
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            return None, 0
        
        train_labels = [train_dataset.sequence_labels[i].item() for i in range(len(train_dataset))]
        class_counts = Counter(train_labels)
        
        class_weights = {0: 1.0, 1: class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0}
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        input_dim = features_scaled.shape[1]
        self.model = TopologicalNeuralNetwork(input_dim, hidden_dim=64, num_layers=3)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        pos_weight = torch.tensor([class_counts[0] / class_counts[1]], device=device)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()], device=device))
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.7)
        
        best_f1 = 0
        best_model_state = None
        patience_counter = 0
        max_patience = 15
        
        train_losses = []
        val_metrics = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                optimizer.zero_grad()
                
                try:
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += batch_labels.size(0)
                    train_correct += (predicted == batch_labels).sum().item()
                    
                except Exception as e:
                    continue
            
            self.model.eval()
            val_predictions = []
            val_labels_list = []
            val_probabilities = []
            
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                    
                    try:
                        outputs = self.model(batch_features)
                        probabilities = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        val_predictions.extend(predicted.cpu().numpy())
                        val_labels_list.extend(batch_labels.cpu().numpy())
                        val_probabilities.extend(probabilities.cpu().numpy())
                        
                    except Exception as e:
                        continue
            
            avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            
            if len(val_predictions) > 0 and len(np.unique(val_labels_list)) > 1:
                val_acc = 100 * accuracy_score(val_labels_list, val_predictions)
                val_precision = precision_score(val_labels_list, val_predictions, pos_label=1, zero_division=0)
                val_recall = recall_score(val_labels_list, val_predictions, pos_label=1, zero_division=0)
                val_f1 = f1_score(val_labels_list, val_predictions, pos_label=1, zero_division=0)
                
                try:
                    val_auc = roc_auc_score(val_labels_list, np.array(val_probabilities)[:, 1])
                except:
                    val_auc = 0.5
                
                scheduler.step(1 - val_f1)
                
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                train_losses.append(avg_loss)
                val_metrics.append({
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_f1': val_f1,
                    'val_auc': val_auc
                })
                
                if patience_counter >= max_patience:
                    break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.model.eval()
        final_predictions = []
        final_labels = []
        final_probabilities = []
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                try:
                    outputs = self.model(batch_features)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    final_predictions.extend(predicted.cpu().numpy())
                    final_labels.extend(batch_labels.cpu().numpy())
                    final_probabilities.extend(probabilities.cpu().numpy())
                except:
                    continue
        
        if len(final_predictions) == 0:
            return self.model, 0
        
        accuracy = accuracy_score(final_labels, final_predictions)
        
        if len(np.unique(final_labels)) > 1:
            precision = precision_score(final_labels, final_predictions, pos_label=1, zero_division=0)
            recall = recall_score(final_labels, final_predictions, pos_label=1, zero_division=0)
            f1 = f1_score(final_labels, final_predictions, pos_label=1, zero_division=0)
            
            try:
                final_probs_array = np.array(final_probabilities)
                auc = roc_auc_score(final_labels, final_probs_array[:, 1])
            except:
                auc = 0.5
        else:
            precision = recall = f1 = auc = 0
        
        cm = confusion_matrix(final_labels, final_predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-Seizure", "Seizure"],
                    yticklabels=["Non-Seizure", "Seizure"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Topological Neural Network - Seizure Detection")
        plt.show()
        
        if val_metrics:
            self._plot_training_progress(train_losses, val_metrics)
        
        return self.model, f1
    
    def _plot_training_progress(self, train_losses, val_metrics):
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            epochs = [m['epoch'] for m in val_metrics]
            
            ax1.plot(train_losses, color='blue', alpha=0.7)
            ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(epochs, [m['val_acc'] for m in val_metrics], label='Accuracy', marker='o', color='green')
            ax2.plot(epochs, [m['val_f1']*100 for m in val_metrics], label='F1×100', marker='s', color='red')
            ax2.set_title('Validation Performance', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Score (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            ax3.plot(epochs, [m['val_precision'] for m in val_metrics], label='Precision', marker='^', color='purple')
            ax3.plot(epochs, [m['val_recall'] for m in val_metrics], label='Recall', marker='v', color='orange')
            ax3.set_title('Precision vs Recall', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Score')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            ax4.plot(epochs, [m['val_auc'] for m in val_metrics], label='AUC', marker='d', color='brown')
            ax4.set_title('AUC Score', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('AUC')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle('Topological Neural Network Training Progress', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            pass


def main_topological_seizure_detection():
    base_path = '/kaggle/input/seizure-epilepcy-chb-mit-eeg-dataset-pediatric/chb-mit-scalp-eeg-database-1.0.0'
    
    pipeline = TopologicalSeizureDetectionPipeline(base_path)
    
    target_patients = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    selected_patients = target_patients[:10]
    
    features, labels, patient_summaries = pipeline.load_multi_patient_data(
        selected_patients, 
        max_seizure_files=1, 
        max_normal_files=3
    )
    
    if features is None:
        return
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_windows = len(features)
    seizure_ratio = counts[1] / total_windows if len(counts) > 1 else 0
    
    try:
        model, final_f1 = pipeline.train_topological_model(
            features,
            labels,
            sequence_length=6,
            batch_size=12,
            epochs=100,
            learning_rate=0.0015
        )
        
        del features, labels
        gc.collect()
        
        return model, final_f1
        
    except Exception as e:
        pass


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    model, performance = main_topological_seizure_detection()