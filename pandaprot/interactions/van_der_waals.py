# pandaprot/interactions/van_der_waals.py
"""
Module for detecting van der Waals interactions in protein structures.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from Bio.PDB import Residue, Atom

from . import utils


def find_van_der_waals_interactions(residues_by_chain: Dict[str, List[Residue]],
                                   distance_factor: float = 1.4,  # Factor over sum of vdW radii
                                   include_intrachain: bool = False) -> List[Dict]:
    """
    Find van der Waals interactions between atoms.
    
    Args:
        residues_by_chain: Dictionary of residues organized by chain
        distance_factor: Maximum distance as a factor of the sum of vdW radii
        include_intrachain: Whether to include interactions within the same chain
        
    Returns:
        List of dictionaries containing van der Waals interaction information
    """
    vdw_interactions = []
    
    # Define van der Waals radii for common elements (in Å)
    vdw_radii = {
        'H': 1.20,
        'C': 1.70,
        'N': 1.55,
        'O': 1.52,
        'P': 1.80,
        'S': 1.80,
        'F': 1.47,
        'CL': 1.75,
        'BR': 1.85,
        'I': 1.98
    }
    
    # Process each pair of chains
    chain_pairs = []
    for chain1 in residues_by_chain:
        for chain2 in residues_by_chain:
            # Skip if same chain and not including intrachain interactions
            if chain1 == chain2 and not include_intrachain:
                continue
                
            chain_pairs.append((chain1, chain2))
    
    # Find van der Waals interactions
    for chain1, chain2 in chain_pairs:
        for res1 in residues_by_chain[chain1]:
            for res2 in residues_by_chain[chain2]:
                # Skip if residues are too far apart (approximate check based on CA atoms)
                try:
                    ca1 = res1['CA']
                    ca2 = res2['CA']
                    if ca1 - ca2 > 10.0:  # Skip if CA-CA distance > 10Å
                        continue
                except KeyError:
                    pass  # If no CA (e.g., non-standard residue), continue checking
                
                # Check all pairs of atoms
                for atom1 in res1:
                    if atom1.element not in vdw_radii:
                        continue
                        
                    for atom2 in res2:
                        if atom2.element not in vdw_radii:
                            continue
                        
                        # Calculate distance
                        dist = utils.calculate_distance(atom1, atom2)
                        
                        # Calculate sum of van der Waals radii
                        vdw_sum = vdw_radii[atom1.element] + vdw_radii[atom2.element]
                        
                        # Check if within van der Waals contact distance
                        if dist <= vdw_sum * distance_factor:
                            vdw_interactions.append({
                                'type': 'van_der_waals',
                                'chain1': chain1,
                                'residue1': f"{res1.resname} {res1.id[1]}",
                                'atom1': atom1.name,
                                'chain2': chain2,
                                'residue2': f"{res2.resname} {res2.id[1]}",
                                'atom2': atom2.name,
                                'distance': dist,
                                'vdw_sum': vdw_sum,
                                'distance_ratio': dist / vdw_sum
                            })
    
    return vdw_interactions