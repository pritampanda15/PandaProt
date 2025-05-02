"""
Module for detecting ionic interactions in protein structures.
"""

from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from Bio.PDB import Residue

from . import utils


def find_ionic_interactions(residues_by_chain: Dict[str, List[Residue]],
                           distance_cutoff: float = 6.0,
                           include_intrachain: bool = False) -> List[Dict]:
    """
    Find ionic interactions between charged residues in different chains.
    
    Args:
        residues_by_chain: Dictionary of residues organized by chain
        distance_cutoff: Maximum distance for ionic interaction detection (Å)
        include_intrachain: Whether to include interactions within the same chain
        
    Returns:
        List of dictionaries containing ionic interaction information
    """
    ionic_interactions = []
    
    # Define charged residues and their charged atoms
    pos_charged_residues = {
        'ARG': ['NH1', 'NH2', 'NE'],
        'LYS': ['NZ'],
        'HIS': ['ND1', 'NE2']  # His can be charged depending on pH
    }
    
    neg_charged_residues = {
        'ASP': ['OD1', 'OD2'],
        'GLU': ['OE1', 'OE2']
    }
    
    # Process each pair of chains
    chain_pairs = []
    for chain1 in residues_by_chain:
        for chain2 in residues_by_chain:
            # Skip if same chain and not including intrachain interactions
            if chain1 == chain2 and not include_intrachain:
                continue
                
            chain_pairs.append((chain1, chain2))
    
    # Find ionic interactions
    for chain1, chain2 in chain_pairs:
        for res1 in residues_by_chain[chain1]:
            for res2 in residues_by_chain[chain2]:
                # Check for positive-negative pairs
                if res1.resname in pos_charged_residues and res2.resname in neg_charged_residues:
                    pos_res, neg_res = res1, res2
                    pos_chain, neg_chain = chain1, chain2
                elif res1.resname in neg_charged_residues and res2.resname in pos_charged_residues:
                    pos_res, neg_res = res2, res1
                    pos_chain, neg_chain = chain2, chain1
                else:
                    continue
                
                # Get charged atoms
                try:
                    pos_atoms = [pos_res[atom_name] for atom_name in pos_charged_residues[pos_res.resname] 
                                if atom_name in pos_res]
                    neg_atoms = [neg_res[atom_name] for atom_name in neg_charged_residues[neg_res.resname] 
                                if atom_name in neg_res]
                except KeyError:
                    continue
                
                # Calculate distances
                for pos_atom in pos_atoms:
                    for neg_atom in neg_atoms:
                        dist = utils.calculate_distance(pos_atom, neg_atom)
                        
                        if dist <= distance_cutoff:
                            ionic_interactions.append({
                                'type': 'ionic',
                                'positive_chain': pos_chain,
                                'positive_residue': f"{pos_res.resname} {pos_res.id[1]}",
                                'positive_atom': pos_atom.name,
                                'negative_chain': neg_chain,
                                'negative_residue': f"{neg_res.resname} {neg_res.id[1]}",
                                'negative_atom': neg_atom.name,
                                'distance': dist
                            })
    
    return ionic_interactions