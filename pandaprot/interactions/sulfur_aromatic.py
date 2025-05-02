# pandaprot/interactions/sulfur_aromatic.py
"""
Module for detecting sulfur-aromatic interactions in protein structures.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from Bio.PDB import Residue

from . import utils


def find_sulfur_aromatic_interactions(residues_by_chain: Dict[str, List[Residue]],
                                     distance_cutoff: float = 5.5,
                                     include_intrachain: bool = False) -> List[Dict]:
    """
    Find sulfur-aromatic interactions between sulfur-containing residues and aromatic rings.
    
    Args:
        residues_by_chain: Dictionary of residues organized by chain
        distance_cutoff: Maximum distance for interaction detection (Å)
        include_intrachain: Whether to include interactions within the same chain
        
    Returns:
        List of dictionaries containing sulfur-aromatic interaction information
    """
    sulfur_aromatic_interactions = []
    
    # Define sulfur-containing residues and their sulfur atoms
    sulfur_residues = {
        'CYS': ['SG'],
        'MET': ['SD']
    }
    
    # Define aromatic residues and their ring atoms
    aromatic_residues = {
        'PHE': [('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ')],  # Phenyl ring
        'TYR': [('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ')],  # Phenyl ring
        'TRP': [('CG', 'CD1', 'NE1', 'CE2', 'CD2'),         # Pyrrole ring
                ('CE2', 'CD2', 'CE3', 'CZ3', 'CH2', 'CZ2')] # Benzene ring
    }
    
    # Process each pair of chains
    chain_pairs = []
    for chain1 in residues_by_chain:
        for chain2 in residues_by_chain:
            # Skip if same chain and not including intrachain interactions
            if chain1 == chain2 and not include_intrachain:
                continue
                
            chain_pairs.append((chain1, chain2))
    
    # Find sulfur-aromatic interactions
    for chain1, chain2 in chain_pairs:
        # Sulfur residues in chain1, aromatic residues in chain2
        for res1 in residues_by_chain[chain1]:
            if res1.resname in sulfur_residues:
                # Get sulfur atoms
                try:
                    sulfur_atoms = [res1[atom_name] for atom_name in sulfur_residues[res1.resname] 
                                   if atom_name in res1]
                    if not sulfur_atoms:
                        continue
                except KeyError:
                    continue
                
                for res2 in residues_by_chain[chain2]:
                    if res2.resname in aromatic_residues:
                        # Check each aromatic ring
                        for ring in aromatic_residues[res2.resname]:
                            # Get ring atoms
                            try:
                                ring_atoms = [res2[atom_name] for atom_name in ring 
                                             if atom_name in res2]
                                if len(ring_atoms) != len(ring):
                                    continue
                            except KeyError:
                                continue
                            
                            # Calculate ring center
                            ring_center = utils.calculate_centroid([atom.coord for atom in ring_atoms])
                            
                            # Check distance to each sulfur atom
                            for sulfur_atom in sulfur_atoms:
                                dist = np.linalg.norm(np.array(ring_center) - np.array(sulfur_atom.coord))
                                
                                if dist <= distance_cutoff:
                                    sulfur_aromatic_interactions.append({
                                        'type': 'sulfur_aromatic',
                                        'sulfur_chain': chain1,
                                        'sulfur_residue': f"{res1.resname} {res1.id[1]}",
                                        'sulfur_atom': sulfur_atom.name,
                                        'aromatic_chain': chain2,
                                        'aromatic_residue': f"{res2.resname} {res2.id[1]}",
                                        'aromatic_ring': '-'.join(ring),
                                        'distance': dist
                                    })
        
        # Aromatic residues in chain1, sulfur residues in chain2
        for res1 in residues_by_chain[chain1]:
            if res1.resname in aromatic_residues:
                for res2 in residues_by_chain[chain2]:
                    if res2.resname in sulfur_residues:
                        # Get sulfur atoms
                        try:
                            sulfur_atoms = [res2[atom_name] for atom_name in sulfur_residues[res2.resname] 
                                          if atom_name in res2]
                            if not sulfur_atoms:
                                continue
                        except KeyError:
                            continue
                        
                        # Check each aromatic ring
                        for ring in aromatic_residues[res1.resname]:
                            # Get ring atoms
                            try:
                                ring_atoms = [res1[atom_name] for atom_name in ring 
                                             if atom_name in res1]
                                if len(ring_atoms) != len(ring):
                                    continue
                            except KeyError:
                                continue
                            
                            # Calculate ring center
                            ring_center = utils.calculate_centroid([atom.coord for atom in ring_atoms])
                            
                            # Check distance to each sulfur atom
                            for sulfur_atom in sulfur_atoms:
                                dist = np.linalg.norm(np.array(ring_center) - np.array(sulfur_atom.coord))
                                
                                if dist <= distance_cutoff:
                                    sulfur_aromatic_interactions.append({
                                        'type': 'sulfur_aromatic',
                                        'sulfur_chain': chain2,
                                        'sulfur_residue': f"{res2.resname} {res2.id[1]}",
                                        'sulfur_atom': sulfur_atom.name,
                                        'aromatic_chain': chain1,
                                        'aromatic_residue': f"{res1.resname} {res1.id[1]}",
                                        'aromatic_ring': '-'.join(ring),
                                        'distance': dist
                                    })
    
    return sulfur_aromatic_interactions