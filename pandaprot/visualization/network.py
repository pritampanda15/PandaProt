# pandaprot/visualization/network.py
"""
Module for creating network visualizations of protein interactions.
"""

from typing import Dict, List, Tuple, Set, Optional
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import logging as logger
from Bio.PDB import PDBParser
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom  
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.vectors import Vector
# Set up logging
logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_interaction_network(structure: Structure, 
                              interactions: Dict[str, List[Dict]]) -> Tuple[nx.Graph, plt.Figure]:
    """
    Create a network visualization of protein interactions.
    
    Args:
        structure: Parsed structure object
        interactions: Dictionary of interactions
        
    Returns:
        Tuple of network graph and matplotlib figure
    """
    # Create a graph
    G = nx.Graph()
    
    # Add nodes (residues)
    residues_seen = set()
    
    for interaction_type, interactions_list in interactions.items():
        for interaction in interactions_list:
            # Extract residue information based on interaction type
            if interaction_type == 'hydrogen_bonds':
                res1 = interaction['donor_residue']
                chain1 = interaction['donor_chain']
                res2 = interaction['acceptor_residue']
                chain2 = interaction['acceptor_chain']

            elif interaction_type in ['ionic_interactions', 'salt_bridges']:
                if all(k in interaction for k in ['positive_residue', 'positive_chain', 'negative_residue', 'negative_chain']):
                    res1 = interaction['positive_residue']
                    chain1 = interaction['positive_chain']
                    res2 = interaction['negative_residue']
                    chain2 = interaction['negative_chain']
                elif all(k in interaction for k in ['residue1', 'chain1', 'residue2', 'chain2']):
                    res1 = interaction['residue1']
                    chain1 = interaction['chain1']
                    res2 = interaction['residue2']
                    chain2 = interaction['chain2']
                else:
                    logger.warning(f"Skipping malformed {interaction_type}: {interaction}")
                    continue

            # elif interaction_type in ['ionic_interactions', 'salt_bridges']:
            #     if 'positive_residue' in interaction:
            #         res1 = interaction['positive_residue']
            #         chain1 = interaction['positive_chain']
            #         res2 = interaction['negative_residue']
            #         chain2 = interaction['negative_chain']
            #     else:
            #         # Fallback for other format
            #         res1 = interaction['residue1']
            #         chain1 = interaction['chain1']
            #         res2 = interaction['residue2']
            #         chain2 = interaction['chain2']
            # else:
            #     # Generic handling - safer approach using only get()
            #     res1 = interaction.get('residue1', interaction.get('chain1_residue', 'unknown'))
            #     chain1 = interaction.get('chain1', 'unknown')
            #     res2 = interaction.get('residue2', interaction.get('chain2_residue', 'unknown'))
            #     chain2 = interaction.get('chain2', 'unknown')
            # else:
            #     # Skip interactions without proper residue mapping
            #     if not all(k in interaction for k in ['residue1', 'residue2', 'chain1', 'chain2']):
            #         continue  # or log warning
            #     res1 = interaction['residue1']
            #     chain1 = interaction['chain1']
            #     res2 = interaction['residue2']
            #     chain2 = interaction['chain2']

            # Create node IDs
            node1 = f"{chain1}:{res1}"
            node2 = f"{chain2}:{res2}"
            
            # Add nodes if not seen before
            if node1 not in residues_seen:
                G.add_node(node1, 
                          chain=chain1, 
                          residue=res1,
                          residue_type=res1.split()[0] if ' ' in res1 else '')
                residues_seen.add(node1)
                
            if node2 not in residues_seen:
                G.add_node(node2, 
                          chain=chain2, 
                          residue=res2,
                          residue_type=res2.split()[0] if ' ' in res2 else '')
                residues_seen.add(node2)
            
            # Add edge with interaction type
            if G.has_edge(node1, node2):
                # Add to existing edge attributes
                G[node1][node2]['types'].append(interaction_type)
                G[node1][node2]['count'] += 1
            else:
                # Create new edge
                G.add_edge(node1, node2, 
                          types=[interaction_type], 
                          count=1,
                          distance=interaction.get('distance', 0))
    
    
    # Create visualization
    fig = plt.figure(figsize=(12, 10))

    # Define node colors by chain
    chains = list(set(nx.get_node_attributes(G, 'chain').values()))
    chain_colors = plt.cm.tab10(np.linspace(0, 1, len(chains)))
    chain_color_map = dict(zip(chains, chain_colors))

    node_colors = [chain_color_map[G.nodes[node]['chain']] for node in G.nodes]

    # Define edge colors by interaction type
    edge_colors = []
    for u, v, attrs in G.edges(data=True):
        if 'hydrogen_bonds' in attrs['types']:
            edge_colors.append('blue')
        elif 'ionic_interactions' in attrs['types'] or 'salt_bridges' in attrs['types']:
            edge_colors.append('red')
        elif 'hydrophobic_interactions' in attrs['types']:
            edge_colors.append('orange')
        elif 'pi_stacking' in attrs['types'] or 'pi_cation' in attrs['types']:
            edge_colors.append('purple')
        else:
            edge_colors.append('gray')

    # Create layout
    pos = nx.spring_layout(G, k=0.3, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.5, alpha=0.7)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

    # Add legend
    plt.legend(['Chain ' + chain for chain in chains], 
            loc='upper right',
            title='Chains')

    plt.title("Protein Interaction Network")
    plt.axis('off')

    # ✅ Save the finalized figure here
    #fig.savefig("interaction_network.png", dpi=300, bbox_inches='tight')
    #plt.close(fig)  # Close the figure to free memory
    logger.info("Interaction network created successfully with %d nodes and %d edges.", 
                G.number_of_nodes(), G.number_of_edges())

    return G, fig
