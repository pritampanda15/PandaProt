"""
Module for detecting hydrophobic interactions in protein structures.
"""

from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom 

from . import utils


def find_hydrophobic_interactions(residues_by_chain: Dict[str, List[Residue]],
                                 distance_cutoff: float = 5.0,
                                 include_intrachain: bool = False) -> List[Dict]:
    """
    Find hydrophobic interactions between residues in different chains.
    
    Args:
        residues_by_chain: Dictionary of residues organized by chain
        distance_cutoff: Maximum distance for hydrophobic interaction detection (Å)
        include_intrachain: Whether to include interactions within the same chain
        
    Returns:
        List of dictionaries containing hydrophobic interaction information
    """
    hydrophobic_interactions = []
    
    # Define hydrophobic residues
    hydrophobic_residues = [
        'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'TYR'
    ]
    
    # Process each pair of chains
    chain_pairs = []
    for chain1 in residues_by_chain:
        for chain2 in residues_by_chain:
            # Skip if same chain and not including intrachain interactions
            if chain1 == chain2 and not include_intrachain:
                continue
                
            chain_pairs.append((chain1, chain2))
    
    # Find hydrophobic interactions
    for chain1, chain2 in chain_pairs:
        for res1 in residues_by_chain[chain1]:
            if res1.resname not in hydrophobic_residues:
                continue
                
            for res2 in residues_by_chain[chain2]:
                if res2.resname not in hydrophobic_residues:
                    continue
                
                # Get carbon atoms (for hydrophobic interactions)
                c_atoms1 = [atom for atom in res1 if atom.element == 'C']
                c_atoms2 = [atom for atom in res2 if atom.element == 'C']
                
                # Calculate distances
                min_dist = float('inf')
                min_atom1 = None
                min_atom2 = None
                
                for atom1 in c_atoms1:
                    for atom2 in c_atoms2:
                        dist = utils.calculate_distance(atom1, atom2)
                        
                        if dist < min_dist:
                            min_dist = dist
                            min_atom1 = atom1
                            min_atom2 = atom2
                
                if min_dist <= distance_cutoff:
                    hydrophobic_interactions.append({
                        'type': 'hydrophobic',
                        'chain1': chain1,
                        'residue1': f"{res1.resname} {res1.id[1]}",
                        'atom1': min_atom1.name,
                        'chain2': chain2,
                        'residue2': f"{res2.resname} {res2.id[1]}",
                        'atom2': min_atom2.name,
                        'distance': min_dist
                    })
    
    return hydrophobic_interactions