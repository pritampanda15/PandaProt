# pandaprot/interactions/disulfide.py
"""
Module for detecting disulfide bridges in protein structures.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from Bio.PDB import Residue

from . import utils


def find_disulfide_bridges(residues_by_chain: Dict[str, List[Residue]],
                          distance_cutoff: float = 2.2,
                          include_intrachain: bool = False) -> List[Dict]:
    """
    Find disulfide bridges between cysteine residues.
    
    Args:
        residues_by_chain: Dictionary of residues organized by chain
        distance_cutoff: Maximum distance for disulfide detection (Å)
        include_intrachain: Whether to include interactions within the same chain
        
    Returns:
        List of dictionaries containing disulfide bridge information
    """
    disulfide_bridges = []
    
    # Process each pair of chains
    chain_pairs = []
    for chain1 in residues_by_chain:
        for chain2 in residues_by_chain:
            # Skip if same chain and not including intrachain interactions
            if chain1 == chain2 and not include_intrachain:
                continue
                
            chain_pairs.append((chain1, chain2))
    
    # Find disulfide bridges
    for chain1, chain2 in chain_pairs:
        for res1 in residues_by_chain[chain1]:
            if res1.resname == 'CYS':
                # Get sulfur atom
                try:
                    sg1 = res1['SG']
                except KeyError:
                    continue
                
                for res2 in residues_by_chain[chain2]:
                    if res2.resname == 'CYS':
                        # Get sulfur atom
                        try:
                            sg2 = res2['SG']
                        except KeyError:
                            continue
                        
                        # Calculate distance
                        dist = utils.calculate_distance(sg1, sg2)
                        
                        if dist <= distance_cutoff:
                            disulfide_bridges.append({
                                'type': 'disulfide',
                                'chain1': chain1,
                                'residue1': f"CYS {res1.id[1]}",
                                'chain2': chain2,
                                'residue2': f"CYS {res2.id[1]}",
                                'distance': dist
                            })
    
    return disulfide_bridges