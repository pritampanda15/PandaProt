# pandaprot/interactions/amide_aromatic.py
"""
Module for detecting amide-aromatic interactions in protein structures.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from Bio.PDB import Residue

from . import utils


def find_amide_aromatic_interactions(residues_by_chain: Dict[str, List[Residue]],
                                    distance_cutoff: float = 4.5,
                                    include_intrachain: bool = False) -> List[Dict]:
    """
    Find amide-aromatic interactions between amide groups and aromatic rings.
    
    Args:
        residues_by_chain: Dictionary of residues organized by chain
        distance_cutoff: Maximum distance for interaction detection (Å)
        include_intrachain: Whether to include interactions within the same chain
        
    Returns:
        List of dictionaries containing amide-aromatic interaction information
    """
    amide_aromatic_interactions = []
    
    # Define amide-containing residues and their amide groups
    amide_residues = {
        'ASN': [('ND2', 'CG', 'OD1')],   # side chain amide
        'GLN': [('NE2', 'CD', 'OE1')],   # side chain amide
        # All residues have backbone amides except PRO
        'ALA': [('N', 'C', 'O')],
        'CYS': [('N', 'C', 'O')],
        'ASP': [('N', 'C', 'O')],
        'GLU': [('N', 'C', 'O')],
        'PHE': [('N', 'C', 'O')],
        'GLY': [('N', 'C', 'O')],
        'HIS': [('N', 'C', 'O')],
        'ILE': [('N', 'C', 'O')],
        'LYS': [('N', 'C', 'O')],
        'LEU': [('N', 'C', 'O')],
        'MET': [('N', 'C', 'O')],
        'ASN': [('N', 'C', 'O')],
        'GLN': [('N', 'C', 'O')],
        'ARG': [('N', 'C', 'O')],
        'SER': [('N', 'C', 'O')],
        'THR': [('N', 'C', 'O')],
        'VAL': [('N', 'C', 'O')],
        'TRP': [('N', 'C', 'O')],
        'TYR': [('N', 'C', 'O')]
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
    
    # Find amide-aromatic interactions
    for chain1, chain2 in chain_pairs:
        # Amide residues in chain1, aromatic residues in chain2
        for res1 in residues_by_chain[chain1]:
            if res1.resname in amide_residues:
                # Get amide groups
                amide_groups = []
                for amide_atoms in amide_residues[res1.resname]:
                    try:
                        atoms = [res1[atom_name] for atom_name in amide_atoms if atom_name in res1]
                        if len(atoms) == len(amide_atoms):
                            amide_groups.append((atoms, '-'.join(amide_atoms)))
                    except KeyError:
                        continue
                
                if not amide_groups:
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
                            
                            # Check distance from each amide group
                            for amide_atoms, amide_name in amide_groups:
                                # Calculate amide center
                                amide_center = utils.calculate_centroid([atom.coord for atom in amide_atoms])
                                
                                # Calculate distance
                                dist = np.linalg.norm(np.array(ring_center) - np.array(amide_center))
                                
                                if dist <= distance_cutoff:
                                    amide_aromatic_interactions.append({
                                        'type': 'amide_aromatic',
                                        'amide_chain': chain1,
                                        'amide_residue': f"{res1.resname} {res1.id[1]}",
                                        'amide_group': amide_name,
                                        'aromatic_chain': chain2,
                                        'aromatic_residue': f"{res2.resname} {res2.id[1]}",
                                        'aromatic_ring': '-'.join(ring),
                                        'distance': dist
                                    })
        
        # Aromatic residues in chain1, amide residues in chain2
        for res1 in residues_by_chain[chain1]:
            if res1.resname in aromatic_residues:
                for res2 in residues_by_chain[chain2]:
                    if res2.resname in amide_residues:
                        # Get amide groups
                        amide_groups = []
                        for amide_atoms in amide_residues[res2.resname]:
                            try:
                                atoms = [res2[atom_name] for atom_name in amide_atoms if atom_name in res2]
                                if len(atoms) == len(amide_atoms):
                                    amide_groups.append((atoms, '-'.join(amide_atoms)))
                            except KeyError:
                                continue
                        
                        if not amide_groups:
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
                            
                            # Check distance from each amide group
                            for amide_atoms, amide_name in amide_groups:
                                # Calculate amide center
                                amide_center = utils.calculate_centroid([atom.coord for atom in amide_atoms])
                                
                                # Calculate distance
                                dist = np.linalg.norm(np.array(ring_center) - np.array(amide_center))
                                
                                if dist <= distance_cutoff:
                                    amide_aromatic_interactions.append({
                                        'type': 'amide_aromatic',
                                        'amide_chain': chain2,
                                        'amide_residue': f"{res2.resname} {res2.id[1]}",
                                        'amide_group': amide_name,
                                        'aromatic_chain': chain1,
                                        'aromatic_residue': f"{res1.resname} {res1.id[1]}",
                                        'aromatic_ring': '-'.join(ring),
                                        'distance': dist
                                    })
    
    return amide_aromatic_interactions