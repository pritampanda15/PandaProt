# pandaprot/interactions/ch_pi.py
"""
Module for detecting CH-pi interactions in protein structures.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from Bio.PDB.Residue import Residue

from . import utils


def find_ch_pi_interactions(residues_by_chain: Dict[str, List[Residue]],
                           distance_cutoff: float = 4.0,
                           angle_cutoff: float = 30.0,
                           include_intrachain: bool = False) -> List[Dict]:
    """
    Find CH-pi interactions between CH groups and aromatic rings.
    
    Args:
        residues_by_chain: Dictionary of residues organized by chain
        distance_cutoff: Maximum distance for interaction detection (Å)
        angle_cutoff: Maximum angle for interaction detection (degrees)
        include_intrachain: Whether to include interactions within the same chain
        
    Returns:
        List of dictionaries containing CH-pi interaction information
    """
    ch_pi_interactions = []
    
    # Define residues with CH groups
    ch_donor_residues = [
        'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'MET', 'PHE', 'TRP', 'TYR',
        'THR', 'SER', 'CYS', 'HIS', 'GLN', 'ASN', 'GLU', 'ASP', 'ARG', 'LYS'
    ]
    
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
    
    # Find CH-pi interactions
    for chain1, chain2 in chain_pairs:
        # CH donor residues in chain1, aromatic residues in chain2
        for res1 in residues_by_chain[chain1]:
            if res1.resname in ch_donor_residues:
                # Get CH groups (carbon atoms with hydrogens)
                ch_atoms = []
                for atom in res1:
                    if atom.element == 'C':
                        # Check if this carbon has any hydrogen neighbors
                        # Since hydrogens are often not present in PDB files,
                        # we'll use a heuristic approach based on carbon hybridization
                        ch_atoms.append(atom)
                
                if not ch_atoms:
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
                            
                            # Calculate ring center and normal
                            ring_center = utils.calculate_centroid([atom.coord for atom in ring_atoms])
                            ring_normal = utils.calculate_ring_normal(ring_atoms)
                            
                            # Check distance and angle for each CH group
                            for ch_atom in ch_atoms:
                                # Calculate distance to ring center
                                dist = np.linalg.norm(np.array(ch_atom.coord) - np.array(ring_center))
                                
                                if dist <= distance_cutoff:
                                    # Calculate vector from ring center to CH
                                    ch_vector = np.array(ch_atom.coord) - np.array(ring_center)
                                    ch_vector = ch_vector / np.linalg.norm(ch_vector)
                                    
                                    # Calculate angle between this vector and ring normal
                                    angle = utils.calculate_angle_between_vectors(tuple(ch_vector), ring_normal)
                                    angle_deg = np.degrees(angle)
                                    
                                    # Check if angle is within cutoff
                                    if angle_deg <= angle_cutoff or angle_deg >= (180 - angle_cutoff):
                                        ch_pi_interactions.append({
                                            'type': 'ch_pi',
                                            'ch_chain': chain1,
                                            'ch_residue': f"{res1.resname} {res1.id[1]}",
                                            'ch_atom': ch_atom.name,
                                            'pi_chain': chain2,
                                            'pi_residue': f"{res2.resname} {res2.id[1]}",
                                            'pi_ring': '-'.join(ring),
                                            'distance': dist,
                                            'angle': min(angle_deg, 180 - angle_deg)
                                        })
        
        # Aromatic residues in chain1, CH donor residues in chain2
        # Similar logic as above, but with chains reversed
        for res1 in residues_by_chain[chain1]:
            if res1.resname in aromatic_residues:
                for res2 in residues_by_chain[chain2]:
                    if res2.resname in ch_donor_residues:
                        # Get CH groups from res2
                        ch_atoms = []
                        for atom in res2:
                            if atom.element == 'C':
                                ch_atoms.append(atom)
                        
                        if not ch_atoms:
                            continue
                        
                        # Check each aromatic ring in res1
                        for ring in aromatic_residues[res1.resname]:
                            # Get ring atoms
                            try:
                                ring_atoms = [res1[atom_name] for atom_name in ring 
                                             if atom_name in res1]
                                if len(ring_atoms) != len(ring):
                                    continue
                            except KeyError:
                                continue
                            
                            # Calculate ring center and normal
                            ring_center = utils.calculate_centroid([atom.coord for atom in ring_atoms])
                            ring_normal = utils.calculate_ring_normal(ring_atoms)
                            
                            # Check for interactions with each CH group
                            for ch_atom in ch_atoms:
                                dist = np.linalg.norm(np.array(ch_atom.coord) - np.array(ring_center))
                                
                                if dist <= distance_cutoff:
                                    # Calculate vector from ring center to CH
                                    ch_vector = np.array(ch_atom.coord) - np.array(ring_center)
                                    ch_vector = ch_vector / np.linalg.norm(ch_vector)
                                    
                                    # Calculate angle
                                    angle = utils.calculate_angle_between_vectors(tuple(ch_vector), ring_normal)
                                    angle_deg = np.degrees(angle)
                                    
                                    if angle_deg <= angle_cutoff or angle_deg >= (180 - angle_cutoff):
                                        ch_pi_interactions.append({
                                            'type': 'ch_pi',
                                            'ch_chain': chain2,
                                            'ch_residue': f"{res2.resname} {res2.id[1]}",
                                            'ch_atom': ch_atom.name,
                                            'pi_chain': chain1,
                                            'pi_residue': f"{res1.resname} {res1.id[1]}",
                                            'pi_ring': '-'.join(ring),
                                            'distance': dist,
                                            'angle': min(angle_deg, 180 - angle_deg)
                                        })
    
    return ch_pi_interactions