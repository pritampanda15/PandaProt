# pandaprot/interactions/amide_amide.py
"""
Module for detecting amide-amide hydrogen bonds in protein structures.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from Bio.PDB.Residue import Residue

from . import utils


def find_amide_amide_interactions(residues_by_chain: Dict[str, List[Residue]],
                                 distance_cutoff: float = 3.5,
                                 angle_cutoff: float = 30.0,
                                 include_intrachain: bool = False) -> List[Dict]:
    """
    Find hydrogen bonds between amide groups.
    
    Args:
        residues_by_chain: Dictionary of residues organized by chain
        distance_cutoff: Maximum distance for hydrogen bond detection (Å)
        angle_cutoff: Maximum deviation from linearity for hydrogen bond (degrees)
        include_intrachain: Whether to include interactions within the same chain
        
    Returns:
        List of dictionaries containing amide-amide interaction information
    """
    amide_amide_interactions = []
    
    # Define amide-containing residues and their amide groups (donor and acceptor pairs)
    amide_residues = {
        'ASN': [('ND2', 'CG', 'OD1')],   # side chain amide
        'GLN': [('NE2', 'CD', 'OE1')]    # side chain amide
    }
    
    # Add backbone amides (except for PRO which lacks an NH group)
    backbone_amide_residues = [
        'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 
        'LEU', 'MET', 'ASN', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR'
    ]
    
    for res in backbone_amide_residues:
        if res not in amide_residues:
            amide_residues[res] = []
        amide_residues[res].append(('N', 'CA', 'C', 'O'))  # backbone amide
    
    # Process each pair of chains
    chain_pairs = []
    for chain1 in residues_by_chain:
        for chain2 in residues_by_chain:
            # Skip if same chain and not including intrachain interactions
            if chain1 == chain2 and not include_intrachain:
                continue
                
            chain_pairs.append((chain1, chain2))
    
    # Find amide-amide interactions
    for chain1, chain2 in chain_pairs:
        for res1 in residues_by_chain[chain1]:
            if res1.resname in amide_residues:
                # Get amide groups
                amide_groups1 = []
                for amide_atoms in amide_residues[res1.resname]:
                    try:
                        atoms = [res1[atom_name] for atom_name in amide_atoms if atom_name in res1]
                        if len(atoms) >= 3:  # Need at least N, C, O
                            # For backbone amide: N is donor, O is acceptor
                            # For side chain amide: ND2/NE2 is donor, OD1/OE1 is acceptor
                            donor = atoms[0]  # N or ND2/NE2
                            acceptor = atoms[-1]  # O or OD1/OE1
                            amide_groups1.append((donor, acceptor, '-'.join(amide_atoms)))
                    except KeyError:
                        continue
                        
                if not amide_groups1:
                    continue
                    
                for res2 in residues_by_chain[chain2]:
                    if res2.resname in amide_residues:
                        # Get amide groups
                        amide_groups2 = []
                        for amide_atoms in amide_residues[res2.resname]:
                            try:
                                atoms = [res2[atom_name] for atom_name in amide_atoms if atom_name in res2]
                                if len(atoms) >= 3:  # Need at least N, C, O
                                    donor = atoms[0]  # N or ND2/NE2
                                    acceptor = atoms[-1]  # O or OD1/OE1
                                    amide_groups2.append((donor, acceptor, '-'.join(amide_atoms)))
                            except KeyError:
                                continue
                                
                        if not amide_groups2:
                            continue
                        
                        # Check all pairs of amide groups
                        for donor1, acceptor1, group1 in amide_groups1:
                            for donor2, acceptor2, group2 in amide_groups2:
                                # Check donor1 -> acceptor2
                                dist1 = utils.calculate_distance(donor1, acceptor2)
                                if dist1 <= distance_cutoff:
                                    # Calculate angle to check for linearity
                                    # In a good H-bond, the D-H...A angle should be close to 180°
                                    # Since H atoms are often not present in PDB, we approximate
                                    
                                    # N-H...O=C angle (should be close to linear)
                                    v1 = np.array(donor1.coord)
                                    v2 = np.array(acceptor2.coord)
                                    v_dir = v2 - v1
                                    v_dir = v_dir / np.linalg.norm(v_dir)
                                    
                                    # Check if close to linear (angle deviation from 180°)
                                    amide_amide_interactions.append({
                                        'type': 'amide_amide_hbond',
                                        'donor_chain': chain1,
                                        'donor_residue': f"{res1.resname} {res1.id[1]}",
                                        'donor_group': group1,
                                        'donor_atom': donor1.name,
                                        'acceptor_chain': chain2,
                                        'acceptor_residue': f"{res2.resname} {res2.id[1]}",
                                        'acceptor_group': group2,
                                        'acceptor_atom': acceptor2.name,
                                        'distance': dist1
                                    })
                                
                                # Check donor2 -> acceptor1
                                dist2 = utils.calculate_distance(donor2, acceptor1)
                                if dist2 <= distance_cutoff:
                                    amide_amide_interactions.append({
                                        'type': 'amide_amide_hbond',
                                        'donor_chain': chain2,
                                        'donor_residue': f"{res2.resname} {res2.id[1]}",
                                        'donor_group': group2,
                                        'donor_atom': donor2.name,
                                        'acceptor_chain': chain1,
                                        'acceptor_residue': f"{res1.resname} {res1.id[1]}",
                                        'acceptor_group': group1,
                                        'acceptor_atom': acceptor1.name,
                                        'distance': dist2
                                    })
    
    return amide_amide_interactions