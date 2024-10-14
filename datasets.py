import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser
from torch.utils.data import Dataset
from joblib import Memory
from typing import List, Tuple, Dict
from tqdm import tqdm
import pickle

class ProteinDynamicsDatasets(Dataset):
    def __init__(self, csv_file: str, cache_dir: str = './cache', nearest_nodes: int = 64, 
                 bins: int = 1024, cutoff: float = 25.0, dihedral_bins: int = 36):
        self.data = pd.read_csv(csv_file)
        self.cache_dir = cache_dir
        self.nearest_nodes = nearest_nodes
        self.bins = bins
        self.cutoff = cutoff
        self.dihedral_bins = dihedral_bins
        
        # 设置缓存
        os.makedirs(cache_dir, exist_ok=True)
        # self.memory = Memory(cache_dir, verbose=0)
        # self.process_pdb = self.memory.cache(self._process_pdb)
        
        # 检查缓存状态
        self.cache_info = self._check_cache()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return self._get_or_process_pdb(row['pdb_name'], row['pdb_path'])

    def _check_cache(self) -> Dict[str, bool]:
        cache_info = {}
        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="check cache"):
            pdb_name = row['pdb_name']
            cache_file = os.path.join(self.cache_dir, f"{pdb_name}.pkl")
            cache_info[pdb_name] = os.path.exists(cache_file)
        return cache_info
    
    def _get_or_process_pdb(self, pdb_name: str, pdb_path: str) -> Dict:
        cache_file = os.path.join(self.cache_dir, f"{pdb_name}.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            processed_data = self._process_pdb(pdb_name, pdb_path)
            with open(cache_file, 'wb') as f:
                pickle.dump(processed_data, f)
            return processed_data

    def _process_pdb(self, pdb_name: str, pdb_path: str) -> Dict:
        parser = PDBParser()
        structure = parser.get_structure(pdb_name, pdb_path)
        
        # 提取氨基酸序列
        sequence = ''.join([residue.resname for residue in structure.get_residues() if residue.id[0] == ' '])
        
        # 提取坐标
        coords = np.array([atom.coord for residue in structure.get_residues() if residue.id[0] == ' ' for atom in residue.get_atoms() if atom.name in ['N', 'CA', 'C']])
        coords = coords.reshape(-1, 3, 3)  # reshape to (residue, atom, xyz)
        
        # 计算二面角
        phi, psi = self._calculate_dihedrals(coords)
        
        # 计算其他特征
        edge_idx, edge_atr, _, dist_ca, dist_cb, phi_bins, psi_bins = self._get_dist(coords)
        
        return {
            'pdb_name': pdb_name,
            'sequence': sequence,
            'phi': phi,
            'psi': psi,
            'edge_idx': edge_idx,
            'edge_atr': edge_atr,
            'dist_ca': dist_ca,
            'dist_cb': dist_cb,
            'phi_bins': phi_bins,
            'psi_bins': psi_bins
        }

    def _calculate_dihedrals(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        def dihedral(p0, p1, p2, p3):
            b0 = -1.0*(p1 - p0)
            b1 = p2 - p1
            b2 = p3 - p2
            b1 /= np.linalg.norm(b1)
            v = b0 - np.dot(b0, b1)*b1
            w = b2 - np.dot(b2, b1)*b1
            x = np.dot(v, w)
            y = np.dot(np.cross(b1, v), w)
            return np.arctan2(y, x)

        phi = []
        psi = []
        
        for i in range(1, len(coords)-1):
            phi.append(dihedral(coords[i-1,2], coords[i,0], coords[i,1], coords[i,2]))
            psi.append(dihedral(coords[i,0], coords[i,1], coords[i,2], coords[i+1,0]))
        
        return np.array(phi), np.array(psi)

    def _get_dist(self, coords: np.ndarray) -> Tuple:
        L = coords.shape[0]
        
        # Calculate pairwise distances
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        
        # Get C-alpha and C-beta distances
        dist_ca = dist[:, :, 1]
        dist_cb = dist[:, :, 1]  # Using C-alpha as C-beta for simplicity
        
        # Get edge indices and attributes
        edge_idx = np.argsort(dist_ca, axis=-1)[:, :self.nearest_nodes]
        edge_atr = np.take_along_axis(dist_ca, edge_idx, axis=1)
        
        # Bin the distances
        interval = self.cutoff / (self.bins - 2)
        edge_bin = np.floor(edge_atr / interval).astype(int)
        edge_bin = np.clip(edge_bin, 0, self.bins - 2) + 1
        
        # Bin the dihedrals
        phi, psi = self._calculate_dihedrals(coords)
        interval = 2 * np.pi / (self.dihedral_bins - 2)
        phi_bins = (np.floor(phi / interval) + 2 + (self.dihedral_bins - 2) // 2).astype(int)
        psi_bins = (np.floor(psi / interval) + 2 + (self.dihedral_bins - 2) // 2).astype(int)
        phi_bins = np.clip(phi_bins, 2, self.dihedral_bins - 1)
        psi_bins = np.clip(psi_bins, 2, self.dihedral_bins - 1)
        
        return edge_idx, edge_atr, None, dist_ca, dist_cb, phi_bins, psi_bins

    def process_all(self):
        """手动处理所有未缓存的PDB文件"""
        for _, row in tqdm(self.data.iterrows(), total=len(self.data)):
            pdb_name = row['pdb_name']
            if not self.cache_info[pdb_name]:
                self._get_or_process_pdb(pdb_name, row['pdb_path'])
                self.cache_info[pdb_name] = True

# 使用示例
if __name__ == "__main__":
    dataset = ProteinDynamicsDatasets('/lustre/grp/cmclab/wangd/v8/cath60_analyse.csv')
    print(f"Dataset size: {len(dataset)}")
    
    # 检查缓存状态
    cached_count = sum(dataset.cache_info.values())
    print(f"Cached items: {cached_count}/{len(dataset)}")

    # 如果有未缓存的项，可以手动处理它们
    if cached_count < len(dataset):
        print("Processing uncached items...")
        dataset.process_all()

    # 获取一个样本（现在会直接从缓存中读取，如果已经处理过）
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f"{key} shape:", value.shape)
        else:
            print(f"{key}:", value)