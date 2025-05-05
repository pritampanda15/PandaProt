"""
Module for detecting pi-pi and pi-cation interactions in protein structures.
"""

from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from Bio.PDB.Structure import Structure
from Bio.PDB.vectors import Vector
from Bio.PDB.Residue import Residue
from . import utils


def find_pi_stacking(residues_by_chain: Dict[str, List[Residue]],
                    distance_cutoff: float = 7.0,
                    angle_cutoff: float = 30.0,
                    include_intrachain: bool = False) -> List[Dict]:
    """
    Find pi-pi stacking interactions between aromatic residues.
    
    Args:
        residues_by_chain: Dictionary of residues organized by chain
        distance_cutoff: Maximum distance between ring centers (Å)
        angle_cutoff: Maximum angle between ring normals (degrees)
        include_intrachain: Whether to include interactions within the same chain
        
    Returns:
        List of dictionaries containing pi-stacking interaction information
    """
    pi_stacking_interactions = []
    
    # Define aromatic residues and their ring atoms
    aromatic_residues = {
        'PHE': [('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ')],  # Phenyl ring
        'TYR': [('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ')],  # Phenyl ring
        'TRP': [('CG', 'CD1', 'NE1', 'CE2', 'CD2'),         # Pyrrole ring
                ('CE2', 'CD2', 'CE3', 'CZ3', 'CH2', 'CZ2')]  # Benzene ring
        # HIS rings could be added here
    }
    
    # Process each pair of chains
    chain_pairs = []
    for chain1 in residues_by_chain:
        for chain2 in residues_by_chain:
            # Skip if same chain and not including intrachain interactions
            if chain1 == chain2 and not include_intrachain:
                continue
                
            chain_pairs.append((chain1, chain2))
    
    # Find pi-stacking interactions
    for chain1, chain2 in chain_pairs:
        for res1 in residues_by_chain[chain1]:
            if res1.resname not in aromatic_residues:
                continue
                
            for res2 in residues_by_chain[chain2]:
                if res2.resname not in aromatic_residues:
                    continue
                
                # Check each pair of rings
                for ring1 in aromatic_residues[res1.resname]:
                    # Try to get ring atoms
                    try:
                        ring1_atoms = [res1[atom_name] for atom_name in ring1 
                                      if atom_name in res1]
                        
                        # Skip if missing atoms
                        if len(ring1_atoms) < len(ring1):
                            continue
                            
                        # Calculate ring 1 center and normal
                        ring1_center = utils.calculate_centroid([atom.coord for atom in ring1_atoms])
                        ring1_normal = utils.calculate_ring_normal(ring1_atoms)
                    except KeyError:
                        continue
                    
                    for ring2 in aromatic_residues[res2.resname]:
                        # Try to get ring atoms
                        try:
                            ring2_atoms = [res2[atom_name] for atom_name in ring2 
                                          if atom_name in res2]
                            
                            # Skip if missing atoms
                            if len(ring2_atoms) < len(ring2):
                                continue
                                
                            # Calculate ring 2 center and normal
                            ring2_center = utils.calculate_centroid([atom.coord for atom in ring2_atoms])
                            ring2_normal = utils.calculate_ring_normal(ring2_atoms)
                        except KeyError:
                            continue
                        
                        # Calculate distance between ring centers
                        dist = np.linalg.norm(np.array(ring1_center) - np.array(ring2_center))
                        
                        if dist <= distance_cutoff:
                            # Calculate angle between ring normals
                            angle = utils.calculate_angle_between_vectors(ring1_normal, ring2_normal)
                            angle_deg = np.degrees(angle)
                            
                            # Check for parallel stacking (small angle)
                            if angle_deg <= angle_cutoff or angle_deg >= (180 - angle_cutoff):
                                stacking_type = 'parallel' if angle_deg <= angle_cutoff else 'antiparallel'
                                
                                pi_stacking_interactions.append({
                                    'type': 'pi_stacking',
                                    'stacking_type': stacking_type,
                                    'chain1': chain1,
                                    'residue1': f"{res1.resname} {res1.id[1]}",
                                    'ring1': '-'.join(ring1),
                                    'chain2': chain2,
                                    'residue2': f"{res2.resname} {res2.id[1]}",
                                    'ring2': '-'.join(ring2),
                                    'distance': dist,
                                    'angle': min(angle_deg, 180 - angle_deg)
                                })
                            
                            # Check for T-shaped stacking (angle close to 90 degrees)
                            elif 90 - angle_cutoff <= angle_deg <= 90 + angle_cutoff:
                                pi_stacking_interactions.append({
                                    'type': 'pi_stacking',
                                    'stacking_type': 'T-shaped',
                                    'chain1': chain1,
                                    'residue1': f"{res1.resname} {res1.id[1]}",
                                    'ring1': '-'.join(ring1),
                                    'chain2': chain2,
                                    'residue2': f"{res2.resname} {res2.id[1]}",
                                    'ring2': '-'.join(ring2),
                                    'distance': dist,
                                    'angle': angle_deg
                                })
    
    return pi_stacking_interactions


def find_pi_cation(residues_by_chain: Dict[str, List[Residue]],
                  distance_cutoff: float = 6.0,
                  include_intrachain: bool = False) -> List[Dict]:
    """
    Find pi-cation interactions between aromatic and charged residues.
    
    Args:
        residues_by_chain: Dictionary of residues organized by chain
        distance_cutoff: Maximum distance for pi-cation detection (Å)
        include_intrachain: Whether to include interactions within the same chain
        
    Returns:
        List of dictionaries containing pi-cation interaction information
    """
    pi_cation_interactions = []
    
    # Define aromatic residues and their ring atoms
    aromatic_residues = {
        'PHE': [('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ')],  # Phenyl ring
        'TYR': [('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ')],  # Phenyl ring
        'TRP': [('CG', 'CD1', 'NE1', 'CE2', 'CD2'),         # Pyrrole ring
                ('CE2', 'CD2', 'CE3', 'CZ3', 'CH2', 'CZ2')]  # Benzene ring
    }
    
    # Define cationic residues and their cationic atoms
    cationic_residues = {
        'ARG': ['NH1', 'NH2', 'NE'],
        'LYS': ['NZ'],
        'HIS': ['ND1', 'NE2']  # His can be charged depending on pH
    }
    
    # Process each pair of chains
    chain_pairs = []
    for chain1 in residues_by_chain:
        for chain2 in residues_by_chain:
            # Skip if same chain and not including intrachain interactions
            if chain1 == chain2 and not include_intrachain:
                continue
                
            chain_pairs.append((chain1, chain2))
    
    # Find pi-cation interactions
    for chain1, chain2 in chain_pairs:
        # Check aromatic residues in chain1 and cationic residues in chain2
        for res1 in residues_by_chain[chain1]:
            if res1.resname in aromatic_residues:
                for res2 in residues_by_chain[chain2]:
                    if res2.resname in cationic_residues:
                        _process_pi_cation_pair(
                            pi_cation_interactions,
                            res1, chain1, res2, chain2,
                            aromatic_residues, cationic_residues,
                            distance_cutoff
                        )
        
        # Check cationic residues in chain1 and aromatic residues in chain2
        for res1 in residues_by_chain[chain1]:
            if res1.resname in cationic_residues:
                for res2 in residues_by_chain[chain2]:
                    if res2.resname in aromatic_residues:
                        _process_pi_cation_pair(
                            pi_cation_interactions,
                            res2, chain2, res1, chain1,
                            aromatic_residues, cationic_residues,
                            distance_cutoff
                        )
    
    return pi_cation_interactions


def _process_pi_cation_pair(interactions: List[Dict],
                           aromatic_res: Residue, aromatic_chain: str,
                           cationic_res: Residue, cationic_chain: str,
                           aromatic_residues: Dict, cationic_residues: Dict,
                           distance_cutoff: float):
    """
    Process a pair of potential pi-cation interacting residues.
    
    Args:
        interactions: List to store detected interactions
        aromatic_res: Aromatic residue
        aromatic_chain: Chain ID of aromatic residue
        cationic_res: Cationic residue
        cationic_chain: Chain ID of cationic residue
        aromatic_residues: Dictionary of aromatic residues and their ring atoms
        cationic_residues: Dictionary of cationic residues and their charged atoms
        distance_cutoff: Maximum distance for interaction detection
    """
    # Get cationic atoms
    try:
        cationic_atoms = [cationic_res[atom_name] for atom_name in cationic_residues[cationic_res.resname] 
                         if atom_name in cationic_res]
        
        if not cationic_atoms:
            return
    except KeyError:
        return
    
    # Check each ring in the aromatic residue
    for ring in aromatic_residues[aromatic_res.resname]:
        # Try to get ring atoms
        try:
            ring_atoms = [aromatic_res[atom_name] for atom_name in ring 
                         if atom_name in aromatic_res]
            
            # Skip if missing atoms
            if len(ring_atoms) < len(ring):
                continue
                
            # Calculate ring center
            ring_center = utils.calculate_centroid([atom.coord for atom in ring_atoms])
        except KeyError:
            continue
        
        # Check distance to each cationic atom
        for cation_atom in cationic_atoms:
            dist = np.linalg.norm(np.array(ring_center) - np.array(cation_atom.coord))
            
            if dist <= distance_cutoff:
                interactions.append({
                    'type': 'pi_cation',
                    'aromatic_chain': aromatic_chain,
                    'aromatic_residue': f"{aromatic_res.resname} {aromatic_res.id[1]}",
                    'ring': '-'.join(ring),
                    'cationic_chain': cationic_chain,
                    'cationic_residue': f"{cationic_res.resname} {cationic_res.id[1]}",
                    'cationic_atom': cation_atom.name,
                    'distance': dist
                })