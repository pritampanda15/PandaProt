"""
Module for detecting hydrogen bonds in protein structures.
"""

from typing import Dict, List, Tuple, Set, Optional
import math
import numpy as np
from Bio.PDB import Atom, Residue

from . import utils


def find_hydrogen_bonds(atoms_by_chain: Dict[str, List[Atom]], 
                       distance_cutoff: float = 3.5,
                       angle_cutoff: float = 120,
                       include_intrachain: bool = False) -> List[Dict]:
    """
    Find hydrogen bonds between atoms in different chains.
    
    Args:
        atoms_by_chain: Dictionary of atoms organized by chain
        distance_cutoff: Maximum distance for hydrogen bond detection (Å)
        angle_cutoff: Minimum angle for hydrogen bond detection (degrees)
        include_intrachain: Whether to include interactions within the same chain
        
    Returns:
        List of dictionaries containing hydrogen bond information
    """
    h_bonds = []
    
    # Define donors and acceptors
    # Donor atoms typically have covalently bound hydrogens
    donors = {
        'N': ['backbone', 'sidechain'],  # Backbone and sidechain nitrogens
        'O': ['sidechain'],            # Hydroxyl oxygens in sidechains
        'S': ['sidechain']             # Thiol groups
    }
    
    # Acceptor atoms typically have lone pairs
    acceptors = {
        'O': ['backbone', 'sidechain'],  # Carbonyl and hydroxyl oxygens
        'N': ['sidechain'],            # Some sidechain nitrogens
        'S': ['sidechain']             # Some sulfur atoms
    }
    
    # Process each pair of chains
    chain_pairs = []
    for chain1 in atoms_by_chain:
        for chain2 in atoms_by_chain:
            # Skip if same chain and not including intrachain interactions
            if chain1 == chain2 and not include_intrachain:
                continue
                
            chain_pairs.append((chain1, chain2))
    
    # Find hydrogen bonds
    for chain1, chain2 in chain_pairs:
        for atom1 in atoms_by_chain[chain1]:
            # Check if atom1 is a potential donor
            if atom1.element in donors:
                # For each potential acceptor
                for atom2 in atoms_by_chain[chain2]:
                    if atom2.element in acceptors:
                        # Calculate distance
                        dist = utils.calculate_distance(atom1, atom2)
                        
                        if dist <= distance_cutoff:
                            # For true hydrogen bonds, we'd need to find hydrogen atoms
                            # and calculate angles, but since H atoms are often missing
                            # in PDB files, we'll use a distance-based approximation
                            
                            atom1_res = atom1.get_parent()
                            atom2_res = atom2.get_parent()
                            
                            h_bonds.append({
                                'type': 'hydrogen_bond',
                                'donor_chain': chain1,
                                'donor_residue': f"{atom1_res.resname} {atom1_res.id[1]}",
                                'donor_atom': atom1.name,
                                'acceptor_chain': chain2,
                                'acceptor_residue': f"{atom2_res.resname} {atom2_res.id[1]}",
                                'acceptor_atom': atom2.name,
                                'distance': dist
                            })
    
    return h_bonds