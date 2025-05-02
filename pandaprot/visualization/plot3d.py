# pandaprot/visualization/plot3d.py
"""
Module for 3D visualization of protein structures and interactions.
"""

from typing import Dict, List, Optional, Any
import py3Dmol
from Bio.PDB import Structure


def create_3d_visualization(pdb_file: str, 
                           structure: Structure,
                           interactions: Dict[str, List[Dict]]) -> Any:
    """
    Create an interactive 3D visualization of the protein structure with highlighted interactions.
    
    Args:
        pdb_file: Path to PDB file
        structure: Parsed structure object
        interactions: Dictionary of interactions
        
    Returns:
        py3Dmol view object
    """
    # Create a py3Dmol view
    view = py3Dmol.view(width=800, height=600)
    
    # Add the PDB structure
    with open(pdb_file, 'r') as f:
        pdb_data = f.read()
    
    view.addModel(pdb_data, 'pdb')
    
    # Style the protein
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    
    # Define colors for different interaction types
    interaction_colors = {
        'hydrogen_bonds': 'blue',
        'ionic_interactions': 'red',
        'hydrophobic_interactions': 'orange',
        'pi_stacking': 'purple',
        'pi_cation': 'green',
        'salt_bridges': 'yellow'
    }
    
    # Add interaction visualizations
    for interaction_type, interactions_list in interactions.items():
        color = interaction_colors.get(interaction_type, 'gray')
        
        for interaction in interactions_list:
            # Get coordinates based on interaction type
            if interaction_type == 'hydrogen_bonds':
                # Get donor and acceptor atoms
                donor_chain = interaction['donor_chain']
                donor_residue = int(interaction['donor_residue'].split()[1])
                donor_atom = interaction['donor_atom']
                
                acceptor_chain = interaction['acceptor_chain']
                acceptor_residue = int(interaction['acceptor_residue'].split()[1])
                acceptor_atom = interaction['acceptor_atom']
                
                # Get coordinates
                try:
                    donor_coord = structure[0][donor_chain][donor_residue][donor_atom].coord
                    acceptor_coord = structure[0][acceptor_chain][acceptor_residue][acceptor_atom].coord
                    
                    # Add cylinder
                    view.addCylinder({
                        'start': {'x': float(donor_coord[0]), 'y': float(donor_coord[1]), 'z': float(donor_coord[2])},
                        'end': {'x': float(acceptor_coord[0]), 'y': float(acceptor_coord[1]), 'z': float(acceptor_coord[2])},
                        'radius': 0.1,
                        'color': color,
                        'dashed': True
                    })
                except KeyError:
                    continue
                    
            elif interaction_type in ['ionic_interactions', 'salt_bridges']:
                # Handle similarly to hydrogen bonds with appropriate atom names
                # Implement based on the specific fields in the interaction dict
                # This can be adapted based on your interaction detection implementation
                pass
                
            elif interaction_type == 'hydrophobic_interactions':
                # Similar approach for hydrophobic interactions
                pass
                
            elif interaction_type in ['pi_stacking', 'pi_cation']:
                # More complex - may need to visualize ring centers
                pass
    
    # Set view parameters
    view.zoomTo()
    
    return view