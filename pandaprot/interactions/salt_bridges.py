"""
Module for detecting salt bridges in protein structures.
"""

from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from Bio.PDB.Residue import Residue

from . import utils


def find_salt_bridges(residues_by_chain: Dict[str, List[Residue]],
                     distance_cutoff: float = 4.0,
                     include_intrachain: bool = False) -> List[Dict]:
    """
    Find salt bridges between oppositely charged residues.
    
    Args:
        residues_by_chain: Dictionary of residues organized by chain
        distance_cutoff: Maximum distance for salt bridge detection (Å)
        include_intrachain: Whether to include interactions within the same chain
        
    Returns:
        List of dictionaries containing salt bridge information
    """
    salt_bridges = []
    
    # Define acidic and basic residues and their charged atoms
    acidic_residues = {
        'ASP': ['OD1', 'OD2'],
        'GLU': ['OE1', 'OE2']
    }
    
    basic_residues = {
        'ARG': ['NH1', 'NH2'],
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
    
    # Find salt bridges
    for chain1, chain2 in chain_pairs:
        for res1 in residues_by_chain[chain1]:
            for res2 in residues_by_chain[chain2]:
                # Check for acidic-basic pairs
                if res1.resname in acidic_residues and res2.resname in basic_residues:
                    acidic_res, basic_res = res1, res2
                    acidic_chain, basic_chain = chain1, chain2
                elif res1.resname in basic_residues and res2.resname in acidic_residues:
                    acidic_res, basic_res = res2, res1
                    acidic_chain, basic_chain = chain2, chain1
                else:
                    continue
                
                # Get charged atoms
                try:
                    acidic_atoms = [acidic_res[atom_name] for atom_name in acidic_residues[acidic_res.resname] 
                                   if atom_name in acidic_res]
                    basic_atoms = [basic_res[atom_name] for atom_name in basic_residues[basic_res.resname] 
                                  if atom_name in basic_res]
                except KeyError:
                    continue
                
                # Calculate distances
                for acidic_atom in acidic_atoms:
                    for basic_atom in basic_atoms:
                        dist = utils.calculate_distance(acidic_atom, basic_atom)
                        
                        if dist <= distance_cutoff:
                            salt_bridges.append({
                                'type': 'salt_bridge',
                                'acidic_chain': acidic_chain,
                                'acidic_residue': f"{acidic_res.resname} {acidic_res.id[1]}",
                                'acidic_atom': acidic_atom.name,
                                'basic_chain': basic_chain,
                                'basic_residue': f"{basic_res.resname} {basic_res.id[1]}",
                                'basic_atom': basic_atom.name,
                                'distance': dist
                            })
    
    return salt_bridges