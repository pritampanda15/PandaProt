# pandaprot/interactions/water_mediated.py
"""
Module for detecting water-mediated interactions in protein structures.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom 

from . import utils


def find_water_mediated_interactions(structure: Structure,
                                    chains: Optional[List[str]] = None,
                                    h_bond_distance_cutoff: float = 3.5,
                                    include_intrachain: bool = False) -> List[Dict]:
    """
    Find water-mediated hydrogen bonds between residues.
    
    Args:
        structure: PDB structure
        chains: List of chains to analyze
        h_bond_distance_cutoff: Maximum distance for hydrogen bond detection (Å)
        include_intrachain: Whether to include interactions within the same chain
        
    Returns:
        List of dictionaries containing water-mediated interaction information
    """
    water_mediated_interactions = []
    
    # Define potential hydrogen bond donors and acceptors
    donors = ['N', 'O', 'S']
    acceptors = ['O', 'N', 'S']
    
    # Get water molecules (HOH/WAT residues)
    water_molecules = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.resname in ['HOH', 'WAT']:
                    try:
                        # Get oxygen atom of water
                        o_atom = residue['O']
                        water_molecules.append(o_atom)
                    except KeyError:
                        continue
    
    # Get potential hydrogen bond atoms from protein residues
    protein_atoms = {}
    for model in structure:
        for chain in model:
            # Skip if chains are specified and this chain is not in the list
            if chains and chain.id not in chains:
                continue
                
            chain_atoms = []
            for residue in chain:
                # Skip water and hetero-atoms
                if residue.resname in ['HOH', 'WAT'] or residue.id[0] != ' ':
                    continue
                    
                for atom in residue:
                    if atom.element in donors or atom.element in acceptors:
                        chain_atoms.append(atom)
            
            protein_atoms[chain.id] = chain_atoms
    
    # Check for water-mediated interactions
    for water in water_molecules:
        # Find all protein atoms within hydrogen bond distance of water
        interacting_atoms = []
        
        for chain_id, atoms in protein_atoms.items():
            for atom in atoms:
                dist = utils.calculate_distance(water, atom)
                if dist <= h_bond_distance_cutoff:
                    interacting_atoms.append((chain_id, atom, dist))
        
        # If water interacts with at least two protein atoms
        if len(interacting_atoms) >= 2:
            # Check all pairs of interacting atoms
            for i in range(len(interacting_atoms)):
                for j in range(i+1, len(interacting_atoms)):
                    chain1, atom1, dist1 = interacting_atoms[i]
                    chain2, atom2, dist2 = interacting_atoms[j]
                    
                    # Skip if same chain and not including intrachain interactions
                    if chain1 == chain2 and not include_intrachain:
                        continue
                    
                    # Record water-mediated interaction
                    res1 = atom1.get_parent()
                    res2 = atom2.get_parent()
                    
                    water_mediated_interactions.append({
                        'type': 'water_mediated',
                        'chain1': chain1,
                        'residue1': f"{res1.resname} {res1.id[1]}",
                        'atom1': atom1.name,
                        'chain2': chain2,
                        'residue2': f"{res2.resname} {res2.id[1]}",
                        'atom2': atom2.name,
                        'water_residue': water.get_parent().id[1],
                        'distance1': dist1,
                        'distance2': dist2
                    })
    
    return water_mediated_interactions