# pandaprot/interactions/metal_coordination.py
"""
Module for detecting metal-coordinated bonds in protein structures.
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from Bio.PDB import Residue, Structure

from . import utils


def find_metal_coordination(structure: Structure,
                           chains: Optional[List[str]] = None,
                           distance_cutoff: float = 3.0,
                           include_intrachain: bool = False) -> List[Dict]:
    """
    Find metal-coordinated bonds in the protein structure.
    
    Args:
        structure: PDB structure
        chains: List of chains to analyze
        distance_cutoff: Maximum distance for coordination detection (Å)
        include_intrachain: Whether to include interactions within the same chain
        
    Returns:
        List of dictionaries containing metal coordination information
    """
    metal_coordination_bonds = []
    
    # Define metal ions
    metal_ions = {
        'ZN': 'Zinc',
        'CA': 'Calcium',
        'MG': 'Magnesium',
        'FE': 'Iron',
        'MN': 'Manganese',
        'CU': 'Copper',
        'CO': 'Cobalt',
        'NI': 'Nickel',
        'CD': 'Cadmium',
        'HG': 'Mercury',
        'PT': 'Platinum',
        'NA': 'Sodium',
        'K': 'Potassium'
    }
    
    # Define coordinating atoms
    coordinating_atoms = {
        'O': ['OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OW'],
        'N': ['ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ'],
        'S': ['SD', 'SG']
    }
    
    # Find metal ions in the structure
    metal_ion_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element in metal_ions:
                        metal_ion_atoms.append(atom)
    
    # Get potential coordinating atoms from protein residues
    protein_atoms = {}
    for model in structure:
        for chain in model:
            # Skip if chains are specified and this chain is not in the list
            if chains and chain.id not in chains:
                continue
                
            chain_atoms = []
            for residue in chain:
                # Skip water and hetero-atoms
                if residue.id[0] != ' ':
                    continue
                    
                for atom in residue:
                    if atom.element in coordinating_atoms and atom.name in coordinating_atoms[atom.element]:
                        chain_atoms.append(atom)
            
            protein_atoms[chain.id] = chain_atoms
    
    # Check for metal coordination
    for metal in metal_ion_atoms:
        metal_element = metal.element
        metal_name = metal_ions[metal_element]
        metal_coord = metal.coord
        
        # Store coordinating atoms for this metal
        coordinating_res_atoms = []
        
        # Find coordinating atoms
        for chain_id, atoms in protein_atoms.items():
            for atom in atoms:
                dist = np.linalg.norm(metal_coord - atom.coord)
                
                if dist <= distance_cutoff:
                    coordinating_res_atoms.append((chain_id, atom, dist))
        
        # Group by chain for organizing metal bridges
        atoms_by_chain = {}
        for chain_id, atom, dist in coordinating_res_atoms:
            if chain_id not in atoms_by_chain:
                atoms_by_chain[chain_id] = []
            atoms_by_chain[chain_id].append((atom, dist))
        
        # If metal coordinates atoms from multiple chains, record as bridge
        if len(atoms_by_chain) >= 2 or (len(atoms_by_chain) == 1 and include_intrachain):
            # Record each coordinating atom
            for chain_id, chain_atoms in atoms_by_chain.items():
                for atom, dist in chain_atoms:
                    res = atom.get_parent()
                    metal_coordination_bonds.append({
                        'type': 'metal_coordination',
                        'metal': metal_name,
                        'metal_element': metal_element,
                        'chain': chain_id,
                        'residue': f"{res.resname} {res.id[1]}",
                        'atom': atom.name,
                        'distance': dist
                    })
            
            # If multiple chains, also record bridges
            if len(atoms_by_chain) >= 2:
                chain_ids = list(atoms_by_chain.keys())
                for i in range(len(chain_ids)):
                    for j in range(i+1, len(chain_ids)):
                        chain1 = chain_ids[i]
                        chain2 = chain_ids[j]
                        
                        # Get representative atoms from each chain
                        atom1, dist1 = atoms_by_chain[chain1][0]
                        atom2, dist2 = atoms_by_chain[chain2][0]
                        
                        res1 = atom1.get_parent()
                        res2 = atom2.get_parent()
                        
                        metal_coordination_bonds.append({
                            'type': 'metal_bridge',
                            'metal': metal_name,
                            'metal_element': metal_element,
                            'chain1': chain1,
                            'residue1': f"{res1.resname} {res1.id[1]}",
                            'atom1': atom1.name,
                            'distance1': dist1,
                            'chain2': chain2,
                            'residue2': f"{res2.resname} {res2.id[1]}",
                            'atom2': atom2.name,
                            'distance2': dist2
                        })
    
    return metal_coordination_bonds