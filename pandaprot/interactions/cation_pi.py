# pandaprot/interactions/cation_pi.py
"""
Module for detecting cation-pi interactions in protein structures.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from Bio.PDB import Residue

from . import utils


def find_cation_pi_interactions(residues_by_chain: Dict[str, List[Residue]],
                               distance_cutoff: float = 6.0,
                               include_intrachain: bool = False) -> List[Dict]:
    """
    Find cation-pi interactions between cationic residues and aromatic residues.
    
    Args:
        residues_by_chain: Dictionary of residues organized by chain
        distance_cutoff: Maximum distance for interaction detection (Å)
        include_intrachain: Whether to include interactions within the same chain
        
    Returns:
        List of dictionaries containing cation-pi interaction information
    """
    cation_pi_interactions = []
    
    # Define cationic residues and their cationic groups
    cationic_residues = {
        'ARG': ['NH1', 'NH2', 'NE'],
        'LYS': ['NZ'],
        'HIS': ['ND1', 'NE2']  # His can be charged depending on pH
    }
    
    # Define aromatic residues and their ring centers
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
    
    # Find cation-pi interactions
    for chain1, chain2 in chain_pairs:
        # Cationic residues in chain1, aromatic residues in chain2
        for res1 in residues_by_chain[chain1]:
            if res1.resname in cationic_residues:
                # Get cationic atoms
                try:
                    cation_atoms = [res1[atom_name] for atom_name in cationic_residues[res1.resname] 
                                   if atom_name in res1]
                    if not cation_atoms:
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
                            
                            # Check distance to each cationic atom
                            for cation_atom in cation_atoms:
                                dist = np.linalg.norm(np.array(ring_center) - np.array(cation_atom.coord))
                                
                                if dist <= distance_cutoff:
                                    cation_pi_interactions.append({
                                        'type': 'cation_pi',
                                        'cation_chain': chain1,
                                        'cation_residue': f"{res1.resname} {res1.id[1]}",
                                        'cation_atom': cation_atom.name,
                                        'pi_chain': chain2,
                                        'pi_residue': f"{res2.resname} {res2.id[1]}",
                                        'pi_ring': '-'.join(ring),
                                        'distance': dist
                                    })
        
        # Aromatic residues in chain1, cationic residues in chain2
        for res1 in residues_by_chain[chain1]:
            if res1.resname in aromatic_residues:
                for res2 in residues_by_chain[chain2]:
                    if res2.resname in cationic_residues:
                        # Get cationic atoms
                        try:
                            cation_atoms = [res2[atom_name] for atom_name in cationic_residues[res2.resname] 
                                          if atom_name in res2]
                            if not cation_atoms:
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
                            
                            # Check distance to each cationic atom
                            for cation_atom in cation_atoms:
                                dist = np.linalg.norm(np.array(ring_center) - np.array(cation_atom.coord))
                                
                                if dist <= distance_cutoff:
                                    cation_pi_interactions.append({
                                        'type': 'cation_pi',
                                        'cation_chain': chain2,
                                        'cation_residue': f"{res2.resname} {res2.id[1]}",
                                        'cation_atom': cation_atom.name,
                                        'pi_chain': chain1,
                                        'pi_residue': f"{res1.resname} {res1.id[1]}",
                                        'pi_ring': '-'.join(ring),
                                        'distance': dist
                                    })
    
    return cation_pi_interactions

















