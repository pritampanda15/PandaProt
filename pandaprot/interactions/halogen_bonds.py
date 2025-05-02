# pandaprot/interactions/halogen_bonds.py
"""
Module for detecting halogen bonds in protein structures.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from Bio.PDB import Residue, Atom

from . import utils


def find_halogen_bonds(residues_by_chain: Dict[str, List[Residue]],
                      distance_cutoff: float = 4.0,
                      angle_cutoff: float = 30.0,
                      include_intrachain: bool = False) -> List[Dict]:
    """
    Find halogen bonds in the protein structure.
    
    Args:
        residues_by_chain: Dictionary of residues organized by chain
        distance_cutoff: Maximum distance for halogen bond detection (Å)
        angle_cutoff: Maximum angle for halogen bond detection (degrees)
        include_intrachain: Whether to include interactions within the same chain
        
    Returns:
        List of dictionaries containing halogen bond information
    """
    halogen_bonds = []
    
    # Define halogen atoms (can be in non-standard residues)
    halogen_elements = ['F', 'CL', 'BR', 'I']
    
    # Define acceptor atoms
    acceptor_elements = ['O', 'N', 'S']
    
    # Process each pair of chains
    chain_pairs = []
    for chain1 in residues_by_chain:
        for chain2 in residues_by_chain:
            # Skip if same chain and not including intrachain interactions
            if chain1 == chain2 and not include_intrachain:
                continue
                
            chain_pairs.append((chain1, chain2))
    
    # Find halogen bonds
    for chain1, chain2 in chain_pairs:
        for res1 in residues_by_chain[chain1]:
            # Get halogen atoms
            halogen_atoms = []
            for atom in res1:
                if atom.element in halogen_elements:
                    # Find the atom to which the halogen is bonded
                    # This is approximate, based on distance
                    bonded_atom = None
                    min_dist = float('inf')
                    
                    for potential_bond_atom in res1:
                        if potential_bond_atom.element == 'C':
                            dist = utils.calculate_distance(atom, potential_bond_atom)
                            if dist < min_dist and dist < 2.0:  # typical C-X bond length
                                min_dist = dist
                                bonded_atom = potential_bond_atom
                    
                    if bonded_atom:
                        halogen_atoms.append((atom, bonded_atom))
            
            if not halogen_atoms:
                continue
                
            for res2 in residues_by_chain[chain2]:
                # Get acceptor atoms
                acceptor_atoms = []
                for atom in res2:
                    if atom.element in acceptor_elements:
                        acceptor_atoms.append(atom)
                
                if not acceptor_atoms:
                    continue
                
                # Check for halogen bonds
                for halogen_atom, bonded_atom in halogen_atoms:
                    for acceptor_atom in acceptor_atoms:
                        dist = utils.calculate_distance(halogen_atom, acceptor_atom)
                        
                        if dist <= distance_cutoff:
                            # Calculate C-X...Y angle (should be close to 180° for halogen bond)
                            # Vector from bonded atom to halogen
                            v1 = np.array(halogen_atom.coord) - np.array(bonded_atom.coord)
                            v1 = v1 / np.linalg.norm(v1)
                            
                            # Vector from halogen to acceptor
                            v2 = np.array(acceptor_atom.coord) - np.array(halogen_atom.coord)
                            v2 = v2 / np.linalg.norm(v2)
                            
                            # Calculate angle
                            angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
                            
                            # Halogen bonds have a linear arrangement (C-X...Y angle close to 180°)
                            if angle >= (180 - angle_cutoff):
                                halogen_bonds.append({
                                    'type': 'halogen_bond',
                                    'halogen_chain': chain1,
                                    'halogen_residue': f"{res1.resname} {res1.id[1]}",
                                    'halogen_atom': halogen_atom.name,
                                    'acceptor_chain': chain2,
                                    'acceptor_residue': f"{res2.resname} {res2.id[1]}",
                                    'acceptor_atom': acceptor_atom.name,
                                    'distance': dist,
                                    'angle': angle
                                })
    
    return halogen_bonds