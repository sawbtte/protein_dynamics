import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
import pickle
import warnings
from Bio.PDB.Residue import Residue
from Bio.PDB import Structure, calc_dihedral
from .constants import AA, restype_to_heavyatom_names, max_num_heavyatoms
from .icoord import *
from .parsers import *
import warnings
warnings.filterwarnings("ignore")

class ProteinDynamicsDatasets(Dataset):
    def __init__(self, csv_file: str, cache_dir: str = './cache', nearest_nodes: int = 64, 
                 bins: int = 1024, cutoff: float = 25.0, dihedral_bins: int = 36,
                 num_models: int = 5, model_ids: List[int] = [1, 2, 3, 4, 5]):
        self.data = pd.read_csv(csv_file)
        self.cache_dir = cache_dir
        self.nearest_nodes = nearest_nodes
        self.bins = bins
        self.cutoff = cutoff
        self.dihedral_bins = dihedral_bins
        self.num_models = num_models
        self.model_ids = model_ids
        
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_info = self._check_cache()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, max_seq_len=200):
        row = self.data.iloc[idx]
        processed_data = self._get_or_process_pdb(row['pdb_name'], row['pdb_path'])
        # 处理processed_data,使所有序列长度一致
        
        # 处理每个模型的数据
        for i, model_data in enumerate(processed_data['models']):
            # new_model_data = self._process_model_data(model_data, max_seq_len)
            processed_data['models'][i] = self._process_model_data(model_data, max_seq_len)
        
        return processed_data
    
    def _process_model_data(self, model_data, max_seq_len):
        seq_len = len(model_data['sequence'])
        if seq_len > max_seq_len:
            # 随机选择起始点
            start = torch.randint(0, seq_len - max_seq_len + 1, (1,)).item()
            end = start + max_seq_len
            
            # 截取序列
            model_data['sequence'] = model_data['sequence'][start:end]
            model_data['phi'] = model_data['phi'][start:end]
            model_data['psi'] = model_data['psi'][start:end]
            model_data['chi'] = model_data['chi'][start:end]
            
            # 截取距离矩阵
            model_data['dist_ca'] = model_data['dist_ca'][start:end, start:end]
            model_data['dist_cb'] = model_data['dist_cb'][start:end, start:end]
            model_data['coords'] = model_data['coords'][start:end]
            
            # 更新edge_idx和edge_atr
            # mask = (model_data['edge_idx'][0] >= start) & (model_data['edge_idx'][0] < end) & \
            #        (model_data['edge_idx'][1] >= start) & (model_data['edge_idx'][1] < end)
            # model_data['edge_idx'] = model_data['edge_idx'][:, mask] - start
            # model_data['edge_atr'] = model_data['edge_atr'][mask]
        
        elif seq_len < max_seq_len:
            pad_len = max_seq_len - seq_len
            
            # 填充序列
            model_data['sequence'] = model_data['sequence'] + [AA.UNK] * pad_len
            
            # 填充phi和psi
            model_data['phi'] = torch.cat([model_data['phi'], torch.zeros(pad_len)])
            model_data['psi'] = torch.cat([model_data['psi'], torch.zeros(pad_len)])
            
            # 填充chi角
            chi_pad = torch.zeros(pad_len, 4)
            model_data['chi'] = torch.cat([model_data['chi'], chi_pad], dim=0)
            
            # 填充距离矩阵
            dist_pad = torch.full((pad_len, seq_len + pad_len), self.cutoff)
            model_data['dist_ca'] = torch.cat([
                torch.cat([model_data['dist_ca'], torch.full((seq_len, pad_len), self.cutoff)], dim=1),
                dist_pad
            ], dim=0)
            model_data['dist_cb'] = torch.cat([
                torch.cat([model_data['dist_cb'], torch.full((seq_len, pad_len), self.cutoff)], dim=1),
                dist_pad
            ], dim=0)
            
            # 更新edge_idx和edge_atr
            # new_edges = [(i, j) for i in range(seq_len, max_seq_len) for j in range(max_seq_len) if i != j]
            # new_edge_idx = torch.tensor(new_edges, dtype=torch.long).t()
            # new_edge_atr = torch.full((len(new_edges), model_data['edge_atr'].shape[1]), self.cutoff)

            pos_heavyatom_pad = torch.zeros((pad_len, model_data['coords'].shape[1], 3))
            model_data['coords'] = torch.cat([model_data['coords'], pos_heavyatom_pad], dim=0)
            
            # model_data['edge_idx'] = torch.cat([model_data['edge_idx'], new_edge_idx], dim=1)
            # model_data['edge_atr'] = torch.cat([model_data['edge_atr'], new_edge_atr], dim=0)
        
        return model_data

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
        
        models_data = []
        
        if self.model_ids:
            models_to_process = [structure[i] for i in self.model_ids if i in structure]
        elif self.num_models:
            models_to_process = list(structure.get_models())[:self.num_models]
        else:
            models_to_process = list(structure.get_models())
        
        for model in models_to_process:
            data, seq_map = parse_biopython_structure(structure[model.id], unknown_threshold=1.0)
            
            # # 计算其他特征
            edge_idx, edge_atr, _, dist_ca, dist_cb, phi_bins, psi_bins = self._get_dist(data['pos_heavyatom'])
            models_data.append({
                'model_id': model.id,
                'sequence': data['aa'],
                'phi': data['phi'],
                'psi': data['psi'],
                'chi': data['chi'],
                # 'edge_idx': edge_idx,
                # 'edge_atr': edge_atr,
                'dist_ca': dist_ca,
                'dist_cb': dist_cb,
                # 'phi_bins': phi_bins,
                # 'psi_bins': psi_bins,
                'coords': data['pos_heavyatom'],
            })
        return {
            'pdb_name': pdb_name,
            'models': models_data
        }

    def _extract_coords_and_angles(self, residues: List[Residue]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords = []
        phi = []
        psi = []
        
        for residue in residues:
            restype = AA(residue.get_resname())
            pos_heavyatom = np.zeros((max_num_heavyatoms, 3))
            
            for idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
                if atom_name and atom_name in residue:
                    pos_heavyatom[idx] = residue[atom_name].get_coord()
            
            coords.append(pos_heavyatom)
            
            if residue.internal_coord is None:
                residue.atom_to_internal_coordinates()
            ic = residue.internal_coord
            
            if ic is not None:
                phi_angle = ic.get_angle('phi')
                psi_angle = ic.get_angle('psi')
                phi.append(np.deg2rad(phi_angle) if phi_angle is not None else 0)
                psi.append(np.deg2rad(psi_angle) if psi_angle is not None else 0)
            else:
                phi.append(0)
                psi.append(0)
        
        return np.array(coords), np.array(phi), np.array(psi)

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
        phi_bins = (np.floor(phi / interval) + 2 + (self.dihedral_bins - 2) // 2).to(torch.int)
        psi_bins = (np.floor(psi / interval) + 2 + (self.dihedral_bins - 2) // 2).to(torch.int)
        phi_bins = np.clip(phi_bins, 2, self.dihedral_bins - 1)
        psi_bins = np.clip(psi_bins, 2, self.dihedral_bins - 1)
        
        return edge_idx, edge_atr, None, dist_ca, dist_cb, phi_bins, psi_bins

    def _calculate_dihedrals(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 这个方法保持不变,因为我们已经在_extract_coords_and_angles中计算了二面角
        # 这里只是为了保持与原有_get_dist方法的兼容性
        return coords[:, 0], coords[:, 1]  # 返回phi和psi

    def process_all(self):
        """手动处理所有未缓存的PDB文件"""
        for _, row in tqdm(self.data.iterrows(), total=len(self.data)):
            pdb_name = row['pdb_name']
            if not self.cache_info[pdb_name]:
                self._get_or_process_pdb(pdb_name, row['pdb_path'])
                self.cache_info[pdb_name] = True

# 使用示例保持不变
if __name__ == "__main__":
    dataset = ProteinDynamicsDatasets('/home/bingxing2/gpuuser834/protein_dynamics/data/pdb_dataset.csv')
    print(f"Dataset size: {len(dataset)}")
    
    cached_count = sum(dataset.cache_info.values())
    print(f"Cached items: {cached_count}/{len(dataset)}")

    if cached_count < len(dataset):
        print("Processing uncached items...")
        dataset.process_all()

    sample = dataset[0]
    print("Sample keys:", sample.keys())
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f"{key} shape:", value.shape)
        else:
            print(f"{key}:", value)