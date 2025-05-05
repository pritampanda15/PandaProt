# pandaprot/core.py
"""
Enhanced core functionality for PandaProt with additional interaction types.
"""

import os
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Structure, Model, Chain, Residue, Atom
from Bio.PDB.vectors import Vector
import py3Dmol
import logging as logger
import networkx as nx


# Import all interaction modules
from .interactions import (
    hydrogen_bonds,
    ionic,
    hydrophobic,
    pi_interactions,
    salt_bridges,
    cation_pi,
    ch_pi,
    disulfide,
    sulfur_aromatic,
    water_mediated,
    metal_coordination,
    halogen_bonds,
    amide_aromatic,
    van_der_waals,
    amide_amide
)
from .visualization import plot3d, network
from .reports import generator


class PandaProt:
    """
    PandaProt: A comprehensive tool for mapping and visualizing 
    interactions at protein interfaces.
    """
    
    def __init__(self, pdb_file: str, chains: Optional[List[str]] = None):
        """
        Initialize PandaProt with a PDB file and optional chain specifications.
        
        Args:
            pdb_file: Path to PDB file
            chains: Optional list of chains to analyze (e.g., ['A', 'B'])
        """
        self.pdb_file = pdb_file
        self.chains = chains
        self.structure = None
        self.interactions = {}
        self.parser = PDBParser(QUIET=True)
        self._load_structure()
        
    def _load_structure(self):
        """Load the PDB structure using BioPython."""
        try:
            self.structure = self.parser.get_structure('complex', self.pdb_file)
            print(f"Successfully loaded structure from {self.pdb_file}")
            
            # If chains are specified, validate they exist
            if self.chains:
                available_chains = [chain.id for chain in self.structure[0]]
                for chain in self.chains:
                    if chain not in available_chains:
                        raise ValueError(f"Chain {chain} not found in structure. "
                                        f"Available chains: {', '.join(available_chains)}")
        except Exception as e:
            raise ValueError(f"Failed to load PDB file: {e}")
    
    def map_interactions(self, distance_cutoff: float = 4.5, include_intrachain: bool = False):
        """
        Map all types of interactions between specified chains.
        
        Args:
            distance_cutoff: Maximum distance cutoff for interaction detection
            include_intrachain: Whether to include interactions within the same chain
            
        Returns:
            Dictionary containing all detected interactions
        """
        # Get atoms and residues by chain
        atoms_by_chain = self._get_atoms_by_chain()
        residues_by_chain = self._get_residues_by_chain()
        
        # Map standard interaction types
        self.interactions['hydrogen_bonds'] = hydrogen_bonds.find_hydrogen_bonds(
            atoms_by_chain, include_intrachain=include_intrachain
        )
        
        self.interactions['ionic_interactions'] = ionic.find_ionic_interactions(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['hydrophobic_interactions'] = hydrophobic.find_hydrophobic_interactions(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['pi_stacking'] = pi_interactions.find_pi_stacking(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['pi_cation'] = pi_interactions.find_pi_cation(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['salt_bridges'] = salt_bridges.find_salt_bridges(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        # Map enhanced interaction types
        self.interactions['cation_pi'] = cation_pi.find_cation_pi_interactions(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['ch_pi'] = ch_pi.find_ch_pi_interactions(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['disulfide_bridges'] = disulfide.find_disulfide_bridges(
            residues_by_chain, include_intrachain=include_intrachain
        )
        
        self.interactions['sulfur_aromatic'] = sulfur_aromatic.find_sulfur_aromatic_interactions(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['water_mediated'] = water_mediated.find_water_mediated_interactions(
            self.structure, chains=self.chains, include_intrachain=include_intrachain
        )
        
        self.interactions['metal_coordination'] = metal_coordination.find_metal_coordination(
            self.structure, chains=self.chains, include_intrachain=include_intrachain
        )
        
        self.interactions['halogen_bonds'] = halogen_bonds.find_halogen_bonds(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['amide_aromatic'] = amide_aromatic.find_amide_aromatic_interactions(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['van_der_waals'] = van_der_waals.find_van_der_waals_interactions(
            residues_by_chain, include_intrachain=include_intrachain
        )
        
        self.interactions['amide_amide'] = amide_amide.find_amide_amide_interactions(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        # Print summary
        total_interactions = sum(len(interactions) for interactions in self.interactions.values())
        print(f"Found {total_interactions} interactions:")
        for interaction_type, interactions in self.interactions.items():
            print(f"  - {interaction_type}: {len(interactions)}")
        
        return self.interactions
    
    def filter_interactions(self, interaction_types: Optional[List[str]] = None,
                          chains: Optional[List[str]] = None,
                          residues: Optional[List[str]] = None,
                          distance_range: Optional[Tuple[float, float]] = None):
        """
        Filter interactions based on specified criteria.
        
        Args:
            interaction_types: Types of interactions to include
            chains: Chains to include
            residues: Residues to include (format: 'A:ASP32', where A is chain ID)
            distance_range: Range of distances to include (min, max)
            
        Returns:
            Dictionary containing filtered interactions
        """
        if not self.interactions:
            print("No interactions to filter. Run map_interactions() first.")
            return {}
        
        filtered_interactions = {}
        
        # Filter by interaction type
        if interaction_types:
            for interaction_type in interaction_types:
                if interaction_type in self.interactions:
                    filtered_interactions[interaction_type] = self.interactions[interaction_type]
        else:
            filtered_interactions = self.interactions.copy()
        
        # Filter by chain
        if chains:
            for interaction_type, interactions in list(filtered_interactions.items()):
                filtered = []
                for interaction in interactions:
                    # Different interaction types have different field names
                    chain1 = interaction.get('chain1', 
                                          interaction.get('donor_chain',
                                                       interaction.get('positive_chain',
                                                                    interaction.get('aromatic_chain',
                                                                                 interaction.get('sulfur_chain', '')))))
                    
                    chain2 = interaction.get('chain2', 
                                          interaction.get('acceptor_chain',
                                                       interaction.get('negative_chain',
                                                                    interaction.get('pi_chain',
                                                                                 interaction.get('cationic_chain', '')))))
                    
                    if chain1 in chains or chain2 in chains:
                        filtered.append(interaction)
                
                filtered_interactions[interaction_type] = filtered
        
        # Filter by residue
        if residues:
            residue_specs = []
            for res_spec in residues:
                if ':' in res_spec:
                    chain, res = res_spec.split(':')
                    residue_specs.append((chain, res))
                else:
                    # If no chain specified, just use the residue
                    residue_specs.append((None, res_spec))
            
            for interaction_type, interactions in list(filtered_interactions.items()):
                filtered = []
                for interaction in interactions:
                    # Extract residue information from interaction
                    res1 = interaction.get('residue1', 
                                        interaction.get('donor_residue',
                                                     interaction.get('positive_residue',
                                                                  interaction.get('aromatic_residue',
                                                                               interaction.get('sulfur_residue', '')))))
                    
                    chain1 = interaction.get('chain1', 
                                          interaction.get('donor_chain',
                                                       interaction.get('positive_chain',
                                                                    interaction.get('aromatic_chain',
                                                                                 interaction.get('sulfur_chain', '')))))
                    
                    res2 = interaction.get('residue2', 
                                        interaction.get('acceptor_residue',
                                                     interaction.get('negative_residue',
                                                                  interaction.get('pi_residue',
                                                                               interaction.get('cationic_residue', '')))))
                    
                    chain2 = interaction.get('chain2', 
                                          interaction.get('acceptor_chain',
                                                       interaction.get('negative_chain',
                                                                    interaction.get('pi_chain',
                                                                                 interaction.get('cationic_chain', '')))))
                    
                    # Check if either residue matches the specifications
                    for chain_spec, res_spec in residue_specs:
                        if (chain_spec is None or chain_spec == chain1) and (res_spec in res1):
                            filtered.append(interaction)
                            break
                        elif (chain_spec is None or chain_spec == chain2) and (res_spec in res2):
                            filtered.append(interaction)
                            break
                
                filtered_interactions[interaction_type] = filtered
        
        # Filter by distance
        if distance_range:
            min_dist, max_dist = distance_range
            for interaction_type, interactions in list(filtered_interactions.items()):
                filtered = []
                for interaction in interactions:
                    dist = interaction.get('distance', 0)
                    if min_dist <= dist <= max_dist:
                        filtered.append(interaction)
                
                filtered_interactions[interaction_type] = filtered
        
        # Print summary of filtered interactions
        total_filtered = sum(len(interactions) for interactions in filtered_interactions.values())
        print(f"Filtered to {total_filtered} interactions:")
        for interaction_type, interactions in filtered_interactions.items():
            print(f"  - {interaction_type}: {len(interactions)}")
        
        return filtered_interactions
    
    def _get_atoms_by_chain(self) -> Dict[str, List[Atom]]:
        """Get all atoms organized by chain."""
        atoms_by_chain = {}
        
        for model in self.structure:
            for chain in model:
                # Skip if chains are specified and this chain is not in the list
                if self.chains and chain.id not in self.chains:
                    continue
                    
                atoms_by_chain[chain.id] = []
                for residue in chain:
                    # Skip hetero-atoms and water
                    if residue.id[0] != ' ':
                        continue
                        
                    for atom in residue:
                        atoms_by_chain[chain.id].append(atom)
        
        return atoms_by_chain
    
    def _get_residues_by_chain(self) -> Dict[str, List[Residue]]:
        """Get all residues organized by chain."""
        residues_by_chain = {}
        
        for model in self.structure:
            for chain in model:
                # Skip if chains are specified and this chain is not in the list
                if self.chains and chain.id not in self.chains:
                    continue
                    
                residues_by_chain[chain.id] = []
                for residue in chain:
                    # Skip hetero-atoms and water
                    if residue.id[0] != ' ':
                        continue
                        
                    residues_by_chain[chain.id].append(residue)
        
        return residues_by_chain
    
    def visualize_3d(self, output_file: Optional[str] = None, interaction_types: Optional[List[str]] = None):
        """
        Generate 3D visualization of the complex with interactions highlighted.
        
        Args:
            output_file: Optional output file to save the visualization
            interaction_types: Types of interactions to visualize (default: all)
            
        Returns:
            Path to the saved HTML file
        """
        # Set default output file if not provided
        if not output_file:
            output_file = "pandaprot_visualization.html"
            
        # Filter interactions if specified
        interactions_to_visualize = self.interactions
        if interaction_types:
            interactions_to_visualize = {k: v for k, v in self.interactions.items() if k in interaction_types}
        
        # Call the updated visualization function
        html_file = plot3d.create_pandaprot_3d_viz(
            self.pdb_file,  # This can be a file path
            interactions_to_visualize,  # This is the interactions dictionary
            output_file  # This is where to save the output
        )
        
        print(f"3D visualization saved to {html_file}")
        return html_file
    
    def generate_report(self, output_file: Optional[str] = None, interaction_types=None):
        """
        Generate a detailed report of all interactions.
        Args:
            output_file: Optional output file to save the report
            interaction_types: Optional list of interaction types to include in the report
        """
        # If you want to filter interactions based on types:
        filtered_interactions = self.interactions
        if interaction_types:
            # Implement filtering logic here if needed
            # For example: filtered_interactions = [i for i in self.interactions if i.type in interaction_types]
            pass
            
        report_df = generator.create_interaction_report(filtered_interactions)
        if output_file and report_df is not None:
            report_df.to_csv(output_file, index=False)
            print(f"Interaction report saved to {output_file}")
        return report_df
    
    def get_interaction_statistics(self):
        """
        Get statistics about interactions.
        
        Returns:
            Dictionary containing interaction statistics
        """
        if not self.interactions:
            print("No interactions to analyze. Run map_interactions() first.")
            return {}
        
        stats = {
            'total_interactions': sum(len(interactions) for interactions in self.interactions.values()),
            'by_type': {k: len(v) for k, v in self.interactions.items()},
            'avg_distances': {},
            'residue_frequencies': {}
        }
        
        # Calculate average distances for each interaction type
        for interaction_type, interactions in self.interactions.items():
            if interactions:
                distances = [interaction['distance'] for interaction in interactions 
                           if 'distance' in interaction]
                if distances:
                    stats['avg_distances'][interaction_type] = sum(distances) / len(distances)
        
        # Calculate residue frequencies in interactions
        residue_counts = {}
        
        for interaction_type, interactions in self.interactions.items():
            for interaction in interactions:
                # Extract residue info based on interaction type
                res1 = interaction.get('residue1', 
                                     interaction.get('donor_residue',
                                                  interaction.get('positive_residue',
                                                               interaction.get('aromatic_residue',
                                                                            interaction.get('sulfur_residue', '')))))
                
                chain1 = interaction.get('chain1', 
                                       interaction.get('donor_chain',
                                                    interaction.get('positive_chain',
                                                                 interaction.get('aromatic_chain',
                                                                              interaction.get('sulfur_chain', '')))))
                
                res2 = interaction.get('residue2', 
                                     interaction.get('acceptor_residue',
                                                  interaction.get('negative_residue',
                                                               interaction.get('pi_residue',
                                                                            interaction.get('cationic_residue', '')))))
                
                chain2 = interaction.get('chain2', 
                                       interaction.get('acceptor_chain',
                                                    interaction.get('negative_chain',
                                                                 interaction.get('pi_chain',
                                                                              interaction.get('cationic_chain', '')))))
                
                # Count residues
                if res1 and chain1:
                    key = f"{chain1}:{res1}"
                    residue_counts[key] = residue_counts.get(key, 0) + 1
                
                if res2 and chain2:
                    key = f"{chain2}:{res2}"
                    residue_counts[key] = residue_counts.get(key, 0) + 1
        
        # Sort residues by frequency
        stats['residue_frequencies'] = dict(
            sorted(residue_counts.items(), key=lambda x: x[1], reverse=True)
        )
        
        return stats
    
    def sanitize_gml_attributes(self, graph):
        for node, attrs in graph.nodes(data=True):
            for key, value in attrs.items():
                if isinstance(value, (np.generic,)):
                    graph.nodes[node][key] = value.item()
        for u, v, attrs in graph.edges(data=True):
            for key, value in attrs.items():
                if isinstance(value, (np.generic,)):
                    graph[u][v][key] = value.item()

    def create_interaction_network(self, output_file: Optional[str] = None,
                                  interaction_types: Optional[List[str]] = None):
        """
        Create a network visualization of interactions.
        Args:
            output_file: Optional output file to save the network visualization
            interaction_types: Types of interactions to include in the network
        """
        if not self.interactions:
            print("No interactions to visualize. Run map_interactions() first.")
            return
        if interaction_types:
            filtered_interactions = {
                k: v for k, v in self.interactions.items() if k in interaction_types
            }
        else:
            filtered_interactions = self.interactions
        #network_graph = network.create_interaction_network(interactions=filtered_interactions)
        # network_graph = network.create_interaction_network(self.structure, filtered_interactions)
        # network.visualize_network(network_graph, output_file)        
        network_graph, fig = network.create_interaction_network(self.structure, filtered_interactions)
        if output_file:
            fig.savefig(output_file.replace(".html", "_network.png"), dpi=300)
        logger.info(f"Network visualization saved to {output_file.replace('.html', '_network.png')}")

        # Save the network graph as a file
        if output_file:
            self.sanitize_gml_attributes(network_graph)
            nx.write_gml(network_graph, output_file.replace(".html", "_network.gml"))

            # Save the network graph in GML format
            logger.info(f"Network graph saved to {output_file.replace('.html', '_network.gml')}")
        # Return the network graph object
        return network_graph