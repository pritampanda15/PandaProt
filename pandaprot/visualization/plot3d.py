from typing import Dict, List, Any, Optional
import py3Dmol
from Bio.PDB import Structure
import numpy as np
import os
import tempfile
import base64
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB import Selection
import os
import base64
import tempfile
import numpy as np
import logging as logger
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom  
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.vectors import Vector

def create_pandaprot_3d_viz(structure_or_interactions, interactions_or_output, output_file=None, width=800, height=600, show_surface=False, show_sidechains=True, interaction_types=None):
    """
    Create an interactive 3D visualization for PandaProt interactions.
    
    Parameters:
    -----------
    structure_or_interactions : Structure or dict
        Either a Bio.PDB.Structure object or a dictionary of interactions
    interactions_or_output : dict or str
        Either a dictionary of interactions or an output file path
    output_file : str, optional
        Path to save the output HTML file
    width : int, default=800
        Width of the visualization in pixels
    height : int, default=600
        Height of the visualization in pixels
    show_surface : bool, default=False
        Whether to show molecular surface
    show_sidechains : bool, default=True
        Whether to show sidechain atoms for interacting residues
    interaction_types : list, optional
        List of interaction types to include (if None, include all)
    """

    structure = None
    pdb_file = None
    interactions = None
    
    # Check first parameter
    if hasattr(structure_or_interactions, "level") and structure_or_interactions.level == "S":
        structure = structure_or_interactions
    elif isinstance(structure_or_interactions, dict):
        interactions = structure_or_interactions
    elif isinstance(structure_or_interactions, str) and os.path.exists(structure_or_interactions):
        pdb_file = structure_or_interactions
    
    # Check second parameter
    if isinstance(interactions_or_output, dict):
        interactions = interactions_or_output
    elif isinstance(interactions_or_output, str):
        if output_file is None:
            output_file = interactions_or_output
    elif hasattr(interactions_or_output, "level") and interactions_or_output.level == "S":
        structure = interactions_or_output
    
    # Default output file name
    if output_file is None:
        output_file = "pandaprot_visualization.html"
    
    # Make sure we have both structure and interactions
    if structure is None and pdb_file is None:
        raise ValueError("No structure or PDB file provided")
    if interactions is None:
        interactions = {}
    
    # Load structure from file path if needed
    if structure is None and pdb_file is not None:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
    
    # Define colors for different interaction types
    interaction_colors = {
        'hydrogen_bonds': 'blue',
        'hydrogen_bond': 'blue',
        'ionic_interactions': 'red',
        'ionic': 'red',
        'salt_bridges': 'yellow',
        'salt_bridge': 'yellow',
        'hydrophobic_interactions': 'orange',
        'hydrophobic': 'orange',
        'pi_stacking': 'purple',
        'pi_stack': 'purple',
        'pi_cation': 'green',
        'cation_pi': 'teal',
        'ch_pi': 'lightseagreen', 
        'disulfide': 'gold',
        'sulfur_aromatic': 'darkkhaki',
        'water_mediated': 'dodgerblue',
        'metal_coordination': 'silver',
        'halogen_bonds': 'darkturquoise',
        'halogen_bond': 'darkturquoise',
        'amide_aromatic': 'mediumorchid',
        'van_der_waals': 'lightslategray',
        'amide_amide': 'hotpink'
    }
    
    # Filter interaction types if specified
    if interaction_types:
        filtered_interactions = {}
        for itype in interaction_types:
            if itype in interactions:
                filtered_interactions[itype] = interactions[itype]
        interactions = filtered_interactions
    
    # Convert structure to PDB format for visualization
    if pdb_file and os.path.exists(pdb_file):
        # If we have a valid file path, read it directly
        with open(pdb_file, 'rb') as f:
            pdb_encoded = base64.b64encode(f.read()).decode()
    else:
        # Otherwise write structure to a temporary file
        temp_pdb = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
        io = PDBIO()
        io.set_structure(structure)
        io.save(temp_pdb.name)
        with open(temp_pdb.name, 'rb') as f:
            pdb_encoded = base64.b64encode(f.read()).decode()
        os.unlink(temp_pdb.name)
    
    # Debug output to see what interaction types are present
    for itype, ilist in interactions.items():
        print(f"DEBUG: {itype}: {len(ilist)} interactions")
        if ilist:
            print(f"DEBUG: Sample {itype}: {ilist[0]}")
    
    # Collect interaction data with careful formatting
    interaction_data = []
    legend_items = set()
    processed_counts = {}
    
    for interaction_type, interactions_list in interactions.items():
        if not interactions_list:
            continue
            
        # Get color for this interaction type
        color_key = interaction_type
        for key in interaction_colors:
            if interaction_type.lower().replace('_', '') == key.lower().replace('_', ''):
                color_key = key
                break
        
        color = interaction_colors.get(color_key, "gray")
        legend_name = interaction_type.replace('_', ' ').title()
        legend_items.add((legend_name, color))
        processed_count = 0
        # Collect interaction data with careful formatting
    interaction_data = []
    legend_items = set()
    processed_counts = {}
    
    for interaction_type, interactions_list in interactions.items():
        if not interactions_list:
            continue
            
        # Get color for this interaction type
        color_key = interaction_type
        for key in interaction_colors:
            if interaction_type.lower().replace('_', '') == key.lower().replace('_', ''):
                color_key = key
                break
        
        color = interaction_colors.get(color_key, "gray")
        legend_name = interaction_type.replace('_', ' ').title()
        legend_items.add((legend_name, color))
        processed_count = 0
        
        # Process each interaction
        for interaction in interactions_list:
            try:
                # Initialize variables for coordinates and label
                start_coords = None
                end_coords = None
                label = None
                
                # Extract coordinates based on interaction type
                if interaction_type in ['hydrogen_bonds', 'hydrogen_bond', 'amide_amide', 'amide_amide_hbond']:
                    # Get donor and acceptor info
                    donor_chain = interaction.get('donor_chain')
                    donor_res_str = interaction.get('donor_residue', '0')
                    if isinstance(donor_res_str, str) and ' ' in donor_res_str:
                        donor_res = int(donor_res_str.split()[-1])
                    else:
                        donor_res = int(donor_res_str) if isinstance(donor_res_str, (int, str)) else 0
                    donor_atom = interaction.get('donor_atom')
                    
                    acceptor_chain = interaction.get('acceptor_chain')
                    acceptor_res_str = interaction.get('acceptor_residue', '0')
                    if isinstance(acceptor_res_str, str) and ' ' in acceptor_res_str:
                        acceptor_res = int(acceptor_res_str.split()[-1])
                    else:
                        acceptor_res = int(acceptor_res_str) if isinstance(acceptor_res_str, (int, str)) else 0
                    acceptor_atom = interaction.get('acceptor_atom')
                    
                    if None in (donor_chain, donor_atom, acceptor_chain, acceptor_atom):
                        continue
                        
                    try:
                        start_coords = structure[0][donor_chain][donor_res][donor_atom].coord
                        end_coords = structure[0][acceptor_chain][acceptor_res][acceptor_atom].coord
                        label = f"{legend_name} ({donor_chain}:{donor_res}:{donor_atom} - {acceptor_chain}:{acceptor_res}:{acceptor_atom})"
                    except Exception as e:
                        print(f"Error getting H-bond coordinates: {e}")
                        continue
                
                elif interaction_type in ['ionic_interactions', 'ionic', 'salt_bridges', 'salt_bridge']:
                    # For ionic interactions, check various possible field names
                    if 'positive_chain' in interaction and 'negative_chain' in interaction:
                        pos_chain = interaction.get('positive_chain')
                        pos_res_str = interaction.get('positive_residue', '0')
                        if isinstance(pos_res_str, str) and ' ' in pos_res_str:
                            pos_res = int(pos_res_str.split()[-1])
                        else:
                            pos_res = int(pos_res_str) if isinstance(pos_res_str, (int, str)) else 0
                        pos_atom = interaction.get('positive_atom')
                        
                        neg_chain = interaction.get('negative_chain')
                        neg_res_str = interaction.get('negative_residue', '0')
                        if isinstance(neg_res_str, str) and ' ' in neg_res_str:
                            neg_res = int(neg_res_str.split()[-1])
                        else:
                            neg_res = int(neg_res_str) if isinstance(neg_res_str, (int, str)) else 0
                        neg_atom = interaction.get('negative_atom')
                    else:
                        # Try basic/acidic naming convention
                        pos_chain = interaction.get('basic_chain')
                        pos_res_str = interaction.get('basic_residue', '0')
                        if isinstance(pos_res_str, str) and ' ' in pos_res_str:
                            pos_res = int(pos_res_str.split()[-1])
                        else:
                            pos_res = int(pos_res_str) if isinstance(pos_res_str, (int, str)) else 0
                        pos_atom = interaction.get('basic_atom')
                        
                        neg_chain = interaction.get('acidic_chain')
                        neg_res_str = interaction.get('acidic_residue', '0')
                        if isinstance(neg_res_str, str) and ' ' in neg_res_str:
                            neg_res = int(neg_res_str.split()[-1])
                        else:
                            neg_res = int(neg_res_str) if isinstance(neg_res_str, (int, str)) else 0
                        neg_atom = interaction.get('acidic_atom')
                    
                    if None in (pos_chain, pos_atom, neg_chain, neg_atom):
                        continue
                        
                    try:
                        start_coords = structure[0][pos_chain][pos_res][pos_atom].coord
                        end_coords = structure[0][neg_chain][neg_res][neg_atom].coord
                        label = f"{legend_name} ({pos_chain}:{pos_res}:{pos_atom} - {neg_chain}:{neg_res}:{neg_atom})"
                    except Exception as e:
                        print(f"Error getting ionic coordinates: {e}")
                        continue
                
                elif interaction_type in ['hydrophobic_interactions', 'hydrophobic', 'van_der_waals']:
                    chain1 = interaction.get('chain1')
                    res1_str = interaction.get('residue1', '0')
                    if isinstance(res1_str, str) and ' ' in res1_str:
                        res1 = int(res1_str.split()[-1])
                    else:
                        res1 = int(res1_str) if isinstance(res1_str, (int, str)) else 0
                    atom1 = interaction.get('atom1')
                    
                    chain2 = interaction.get('chain2')
                    res2_str = interaction.get('residue2', '0')
                    if isinstance(res2_str, str) and ' ' in res2_str:
                        res2 = int(res2_str.split()[-1])
                    else:
                        res2 = int(res2_str) if isinstance(res2_str, (int, str)) else 0
                    atom2 = interaction.get('atom2')
                    
                    if None in (chain1, atom1, chain2, atom2):
                        continue
                        
                    try:
                        start_coords = structure[0][chain1][res1][atom1].coord
                        end_coords = structure[0][chain2][res2][atom2].coord
                        label = f"{legend_name} ({chain1}:{res1}:{atom1} - {chain2}:{res2}:{atom2})"
                    except Exception as e:
                        print(f"Error getting hydrophobic coordinates: {e}")
                        continue
                
                elif interaction_type in ['pi_stacking', 'pi_stack', 'pi_cation', 'amide_aromatic']:
                    # Handle pi-cation with aromatic center to cation atom
                    if 'aromatic_chain' in interaction and 'cationic_chain' in interaction:
                        aromatic_chain = interaction.get('aromatic_chain')
                        aromatic_res_str = interaction.get('aromatic_residue', '0')
                        if isinstance(aromatic_res_str, str) and ' ' in aromatic_res_str:
                            aromatic_res = int(aromatic_res_str.split()[-1])
                        else:
                            aromatic_res = int(aromatic_res_str) if isinstance(aromatic_res_str, (int, str)) else 0
                            
                        cationic_chain = interaction.get('cationic_chain')
                        cationic_res_str = interaction.get('cationic_residue', '0')
                        if isinstance(cationic_res_str, str) and ' ' in cationic_res_str:
                            cationic_res = int(cationic_res_str.split()[-1])
                        else:
                            cationic_res = int(cationic_res_str) if isinstance(cationic_res_str, (int, str)) else 0
                        cationic_atom = interaction.get('cationic_atom')
                        
                        # For aromatic residue, use CA atom as fallback
                        try:
                            # Try to get center of aromatic ring
                            start_coords = structure[0][aromatic_chain][aromatic_res]['CA'].coord
                            end_coords = structure[0][cationic_chain][cationic_res][cationic_atom].coord
                            label = f"{legend_name} ({aromatic_chain}:{aromatic_res} - {cationic_chain}:{cationic_res}:{cationic_atom})"
                        except Exception as e:
                            print(f"Error getting pi-cation coordinates: {e}")
                            continue
                    
                    # Handle amide-aromatic interactions
                    elif 'amide_chain' in interaction and 'aromatic_chain' in interaction:
                        amide_chain = interaction.get('amide_chain')
                        amide_res_str = interaction.get('amide_residue', '0')
                        if isinstance(amide_res_str, str) and ' ' in amide_res_str:
                            amide_res = int(amide_res_str.split()[-1])
                        else:
                            amide_res = int(amide_res_str) if isinstance(amide_res_str, (int, str)) else 0
                            
                        aromatic_chain = interaction.get('aromatic_chain')
                        aromatic_res_str = interaction.get('aromatic_residue', '0')
                        if isinstance(aromatic_res_str, str) and ' ' in aromatic_res_str:
                            aromatic_res = int(aromatic_res_str.split()[-1])
                        else:
                            aromatic_res = int(aromatic_res_str) if isinstance(aromatic_res_str, (int, str)) else 0
                        
                        # Use CA atoms for both
                        try:
                            start_coords = structure[0][amide_chain][amide_res]['CA'].coord
                            end_coords = structure[0][aromatic_chain][aromatic_res]['CA'].coord
                            label = f"{legend_name} ({amide_chain}:{amide_res} - {aromatic_chain}:{aromatic_res})"
                        except Exception as e:
                            print(f"Error getting amide-aromatic coordinates: {e}")
                            continue
                    else:
                        continue
                
                elif interaction_type in ['cation_pi', 'ch_pi']:
                    # Specialized handling for ch_pi
                    if 'ch_chain' in interaction and 'pi_chain' in interaction:
                        ch_chain = interaction.get('ch_chain')
                        ch_res_str = interaction.get('ch_residue', '0')
                        if isinstance(ch_res_str, str) and ' ' in ch_res_str:
                            ch_res = int(ch_res_str.split()[-1])
                        else:
                            ch_res = int(ch_res_str) if isinstance(ch_res_str, (int, str)) else 0
                        ch_atom = interaction.get('ch_atom')
                        
                        pi_chain = interaction.get('pi_chain')
                        pi_res_str = interaction.get('pi_residue', '0')
                        if isinstance(pi_res_str, str) and ' ' in pi_res_str:
                            pi_res = int(pi_res_str.split()[-1])
                        else:
                            pi_res = int(pi_res_str) if isinstance(pi_res_str, (int, str)) else 0
                        
                        # Use CH atom to CA of aromatic
                        try:
                            if ch_atom:
                                start_coords = structure[0][ch_chain][ch_res][ch_atom].coord
                            else:
                                start_coords = structure[0][ch_chain][ch_res]['CA'].coord
                                
                            end_coords = structure[0][pi_chain][pi_res]['CA'].coord
                            label = f"{legend_name} ({ch_chain}:{ch_res} - {pi_chain}:{pi_res})"
                        except Exception as e:
                            print(f"Error getting CH-pi coordinates: {e}")
                            continue
                    
                    # Handle cation-pi
                    elif 'cation_chain' in interaction and 'pi_chain' in interaction:
                        cation_chain = interaction.get('cation_chain')
                        cation_res_str = interaction.get('cation_residue', '0')
                        if isinstance(cation_res_str, str) and ' ' in cation_res_str:
                            cation_res = int(cation_res_str.split()[-1])
                        else:
                            cation_res = int(cation_res_str) if isinstance(cation_res_str, (int, str)) else 0
                        cation_atom = interaction.get('cation_atom')
                        
                        pi_chain = interaction.get('pi_chain')
                        pi_res_str = interaction.get('pi_residue', '0')
                        if isinstance(pi_res_str, str) and ' ' in pi_res_str:
                            pi_res = int(pi_res_str.split()[-1])
                        else:
                            pi_res = int(pi_res_str) if isinstance(pi_res_str, (int, str)) else 0
                        
                        # Use cation atom to CA of aromatic
                        try:
                            if cation_atom:
                                start_coords = structure[0][cation_chain][cation_res][cation_atom].coord
                            else:
                                start_coords = structure[0][cation_chain][cation_res]['CA'].coord
                                
                            end_coords = structure[0][pi_chain][pi_res]['CA'].coord
                            label = f"{legend_name} ({cation_chain}:{cation_res} - {pi_chain}:{pi_res})"
                        except Exception as e:
                            print(f"Error getting cation-pi coordinates: {e}")
                            continue
                    else:
                        continue
                
                elif interaction_type in ['water_mediated']:
                    # Special handling for water-mediated interactions
                    chain1 = interaction.get('chain1')
                    res1_str = interaction.get('residue1', '0')
                    if isinstance(res1_str, str) and ' ' in res1_str:
                        res1 = int(res1_str.split()[-1])
                    else:
                        res1 = int(res1_str) if isinstance(res1_str, (int, str)) else 0
                    atom1 = interaction.get('atom1')
                    
                    chain2 = interaction.get('chain2')
                    res2_str = interaction.get('residue2', '0')
                    if isinstance(res2_str, str) and ' ' in res2_str:
                        res2 = int(res2_str.split()[-1])
                    else:
                        res2 = int(res2_str) if isinstance(res2_str, (int, str)) else 0
                    atom2 = interaction.get('atom2')
                    
                    water_res = interaction.get('water_residue')
                    
                    # Skip if missing critical info
                    if None in (chain1, atom1, chain2, atom2, water_res):
                        continue
                    
                    # In PDB files, water molecules are often in a separate chain
                    # Try different ways to access the water molecule
                    water_chain = 'W'  # Default
                    water_atom = 'O'   # Water oxygen
                    
                    try:
                        # Method 1: Check if water is in chain W
                        try:
                            water_coords = structure[0][water_chain][water_res][water_atom].coord
                        except:
                            # Method 2: Try with no chain specifier
                            try:
                                for chain in structure[0]:
                                    if water_res in chain:
                                        water_chain = chain.id
                                        water_coords = structure[0][water_chain][water_res][water_atom].coord
                                        break
                            except:
                                # Method 3: Check HOH residues for matching ID
                                for chain in structure[0]:
                                    for residue in chain:
                                        if residue.resname == 'HOH' and residue.id[1] == water_res:
                                            water_chain = chain.id
                                            water_res = residue.id[1]
                                            water_coords = residue['O'].coord
                                            break
                        
                        # Create connections from each residue to water
                        start_coords = structure[0][chain1][res1][atom1].coord
                        end_coords = water_coords
                        label = f"{legend_name} ({chain1}:{res1}:{atom1} - HOH:{water_res})"
                        
                        # Add first connection
                        interaction_data.append({
                            'type': interaction_type,
                            'color': color,
                            'start': start_coords,
                            'end': end_coords,
                            'label': label
                        })
                        
                        # Add second connection (water to res2)
                        start_coords = water_coords
                        end_coords = structure[0][chain2][res2][atom2].coord
                        label = f"{legend_name} (HOH:{water_res} - {chain2}:{res2}:{atom2})"
                        
                        processed_count += 1
                        continue  # Skip the normal addition below
                    except Exception as e:
                        print(f"Error getting water-mediated coordinates: {e}")
                        continue
                
                elif interaction_type in ['disulfide', 'disulfide_bridges']:
                    chain1 = interaction.get('chain1')
                    res1_str = interaction.get('residue1', '0')
                    if isinstance(res1_str, str) and ' ' in res1_str:
                        res1 = int(res1_str.split()[-1])
                    else:
                        res1 = int(res1_str) if isinstance(res1_str, (int, str)) else 0
                    atom1 = interaction.get('atom1', 'SG')
                    
                    chain2 = interaction.get('chain2')
                    res2_str = interaction.get('residue2', '0')
                    if isinstance(res2_str, str) and ' ' in res2_str:
                        res2 = int(res2_str.split()[-1])
                    else:
                        res2 = int(res2_str) if isinstance(res2_str, (int, str)) else 0
                    atom2 = interaction.get('atom2', 'SG')
                    
                    if None in (chain1, chain2):
                        continue
                        
                    try:
                        start_coords = structure[0][chain1][res1][atom1].coord
                        end_coords = structure[0][chain2][res2][atom2].coord
                        label = f"{legend_name} ({chain1}:{res1}:{atom1} - {chain2}:{res2}:{atom2})"
                    except Exception as e:
                        print(f"Error getting disulfide coordinates: {e}")
                        continue
                
                # Skip if we couldn't get coordinates
                if start_coords is None or end_coords is None or label is None:
                    continue
                
                # Calculate distance
                distance = np.linalg.norm(np.array(start_coords) - np.array(end_coords))
                
                # Skip extremely long interactions (likely errors)
                if distance > 12.0:  # 12 Angstroms is a reasonable cutoff
                    print(f"Skipping long-distance interaction: {distance:.2f}Å - {label}")
                    continue
                
                # Add this interaction to the data
                interaction_data.append({
                    'type': interaction_type,
                    'color': color,
                    'start': start_coords,
                    'end': end_coords,
                    'label': f"{label} ({distance:.2f}Å)"
                })
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {interaction_type} interaction: {e}")
                continue
        
        processed_counts[interaction_type] = processed_count
    
    # Report processing statistics
    print("Processing summary:")
    for itype, count in processed_counts.items():
        total = len(interactions.get(itype, []))
        print(f"  {itype}: {count}/{total} interactions processed")
        
    # Collect all residues involved in interactions
    interacting_residues = set()
    
    for data in interaction_data:
        # Parse the label to extract residue info
        # Example label: "Hydrogen Bond (A:42:N - B:56:O) (2.85Å)"
        parts = data['label'].split(' - ')
        if len(parts) >= 2:
            res1_info = parts[0].split('(')[1].split(':')
            res2_info = parts[1].split(':')
            
            if len(res1_info) >= 2:
                chain1, res1 = res1_info[0], res1_info[1]
                interacting_residues.add(f"{chain1}:{res1}")
            
            if len(res2_info) >= 2:
                chain2, res2 = res2_info[0], res2_info[1].split(')')[0]
                interacting_residues.add(f"{chain2}:{res2}")
    
    # Create a minimal, clean HTML template
    # Modified HTML template with added residue display and tooltip functionality
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PandaProt 3D Visualization</title>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }}
        #container {{ width: 100%; height: 100%; position: relative; }}
        #controls {{ position: absolute; bottom: 10px; left: 10px; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.3); }}
        #legend {{ position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px; font-family: sans-serif; max-width: 200px; }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 5px; }}
        .color-box {{ width: 15px; height: 15px; margin-right: 8px; border: 1px solid #999; }}
        #tooltip {{ position: absolute; background: rgba(0,0,0,0.8); color: white; padding: 8px; border-radius: 4px; font-family: sans-serif; font-size: 14px; pointer-events: none; display: none; z-index: 1000; max-width: 300px; }}
    </style>
</head>
<body>
    <div id="container"></div>
    
    <div id="controls">
        <button onclick="resetView()">Reset View</button>
        <button onclick="exportImage()">Export PNG</button>
        <button onclick="clearLabels()">Clear Labels</button>
        <button onclick="toggleSidechains()">Toggle Sidechains</button>
        <button onclick="toggleSurface()">Toggle Surface</button>
    </div>
    
    <div id="legend">
        <h3>Interaction Types</h3>
        <div id="legend-items"></div>
    </div>
    
    <div id="tooltip"></div>
    
    <script>
        // Initialize variables
        let viewer;
        let currentLabel = null;
        let showingSidechains = {'true' if show_sidechains else 'false'};
        let showingSurface = {'true' if show_surface else 'false'};
        let tooltip = document.getElementById('tooltip');
        
        // Helper functions
        function resetView() {{
            viewer.zoomTo();
            viewer.render();
        }}
        
        function clearLabels() {{
            if (currentLabel) {{
                viewer.removeLabel(currentLabel);
                currentLabel = null;
                viewer.render();
            }}
        }}
        
        function showLabel(text, position) {{
            clearLabels();
            currentLabel = viewer.addLabel(text, {{
                position: position,
                backgroundColor: 'white',
                fontColor: 'black',
                showBackground: true,
                fontSize: 12,
                inFront: true
            }});
            viewer.render();
        }}
        
        function showTooltip(text, position) {{
            // Convert 3D position to screen coordinates
            let canvas = viewer.getCanvas();
            let rect = canvas.getBoundingClientRect();
            let pixelPosition = viewer.modelToScreen(position);
            
            tooltip.style.left = (rect.left + pixelPosition.x) + 'px';
            tooltip.style.top = (rect.top + pixelPosition.y - 40) + 'px';
            tooltip.innerHTML = text;
            tooltip.style.display = 'block';
            
            // Hide tooltip after 3 seconds
            setTimeout(() => {{
                tooltip.style.display = 'none';
            }}, 3000);
        }}
        
        function toggleSidechains() {{
            showingSidechains = !showingSidechains;
            updateStyles();
        }}
        
        function toggleSurface() {{
            showingSurface = !showingSurface;
            
            if (showingSurface) {{
                viewer.addSurface($3Dmol.VDW, {{opacity: 0.4, color: "white"}}, {{hetflag: false}});
            }} else {{
                viewer.removeSurface();
            }}
            
            viewer.render();
        }}
        
        function updateStyles() {{
            // Base styles
            viewer.setStyle({{}}, {{cartoon: {{color: 'lightgray', opacity: 0.8}}}});
            viewer.setStyle({{hetflag: true}}, {{stick: {{colorscheme: 'elementColors', radius: 0.3}}}});
            
            // Sidechain styles for interacting residues
            if (showingSidechains) {{
                // Add sidechains for interacting residues
                interactingResidues.forEach(function(res) {{
                    viewer.setStyle({{chain: res.chain, resi: res.resi}}, 
                        {{cartoon: {{color: 'lightgray', opacity: 0.8}},
                         stick: {{colorscheme: 'elementColors', radius: 0.2}}}}
                    );
                }});
            }}
            
            viewer.render();
        }}
        
        function exportImage() {{
            let link = document.createElement('a');
            link.download = 'pandaprot_visualization.png';
            link.href = viewer.getCanvas().toDataURL('image/png');
            link.click();
        }}
        
        // Setup visualization when document is ready
        $(document).ready(function() {{
            // Create viewer
            viewer = $3Dmol.createViewer($("#container"), {{
                backgroundColor: 'white',
                width: {width},
                height: {height}
            }});
            
            // Load PDB data
            viewer.addModel(atob("{pdb_encoded}"), "pdb");
            
            // Define interacting residues
            const interactingResidues = [
"""
    
    # Add interacting residues data
    for i, residue in enumerate(interacting_residues):
        chain, resi = residue.split(':')
        html += f"""                {{chain: "{chain}", resi: {resi}}}{',' if i < len(interacting_residues)-1 else ''}
"""
    
    html += """            ];
            
            // Set default style
            viewer.setStyle({}, {cartoon: {color: 'lightgray', opacity: 0.8}});
            viewer.setStyle({hetflag: true}, {stick: {colorscheme: 'elementColors', radius: 0.3}});
            
            // Add sidechains for interacting residues if enabled
            if (showingSidechains) {
                interactingResidues.forEach(function(res) {
                    viewer.setStyle({chain: res.chain, resi: res.resi}, 
                        {cartoon: {color: 'lightgray', opacity: 0.8},
                         stick: {colorscheme: 'elementColors', radius: 0.2}}
                    );
                });
            }
            
            // Add surface if requested
            if (showingSurface) {
                viewer.addSurface($3Dmol.VDW, {opacity: 0.4, color: "white"}, {hetflag: false});
            }
            
            // Add interactions
            const interactions = [
"""
    
    # Add each interaction as a clean JavaScript object
    for i, data in enumerate(interaction_data):
        html += f"""                {{
                    type: "{data['type']}",
                    color: "{data['color']}",
                    start: {{x: {data['start'][0]}, y: {data['start'][1]}, z: {data['start'][2]}}},
                    end: {{x: {data['end'][0]}, y: {data['end'][1]}, z: {data['end'][2]}}},
                    label: "{data['label']}"
                }}{'' if i == len(interaction_data)-1 else ','}
"""
    
    html += """            ];
            
            // Draw all interactions
            interactions.forEach(function(interaction) {
                // Add cylinder for the interaction
                viewer.addCylinder({
                    start: interaction.start,
                    end: interaction.end,
                    radius: 0.15,
                    color: interaction.color,
                    dashed: interaction.type.includes('hydrogen') || interaction.type.includes('salt')
                });
                
                // Add center point for label activation
                const midX = (interaction.start.x + interaction.end.x) / 2;
                const midY = (interaction.start.y + interaction.end.y) / 2;
                const midZ = (interaction.start.z + interaction.end.z) / 2;
                
                // Add invisible clickable sphere
                viewer.addSphere({
                    center: {x: midX, y: midY, z: midZ},
                    radius: 0.3,
                    color: interaction.color,
                    opacity: 0,
                    clickable: true,
                    callback: function() {
                        showLabel(interaction.label, {x: midX, y: midY, z: midZ});
                    }
                });
            });
            
            // Add clickable spheres for each residue for tooltips
            interactingResidues.forEach(function(res) {
                // Get the alpha carbon of the residue
                const atoms = viewer.getModel().selectedAtoms({chain: res.chain, resi: res.resi, atom: 'CA'});
                
                if (atoms.length > 0) {
                    const atom = atoms[0];
                    const resname = atom.resn;
                    
                    viewer.addSphere({
                        center: {x: atom.x, y: atom.y, z: atom.z},
                        radius: 0.5,
                        color: 'white',
                        opacity: 0.0,  // Invisible
                        clickable: true,
                        callback: function() {
                            showTooltip(`Residue: ${resname} ${res.resi} (Chain ${res.chain})`, 
                                        {x: atom.x, y: atom.y, z: atom.z});
                        }
                    });
                }
            });
            
            // Create legend
            const legendItems = [
"""
    
    # Add legend items
    for i, (name, color) in enumerate(sorted(legend_items)):
        html += f"""                {{name: "{name}", color: "{color}"}}{',' if i < len(legend_items)-1 else ''}
"""
    
    html += """            ];
            
            // Populate legend
            const legendContainer = document.getElementById('legend-items');
            legendItems.forEach(function(item) {
                const div = document.createElement('div');
                div.className = 'legend-item';
                
                const colorBox = document.createElement('div');
                colorBox.className = 'color-box';
                colorBox.style.backgroundColor = item.color;
                
                const label = document.createElement('span');
                label.textContent = item.name;
                
                div.appendChild(colorBox);
                div.appendChild(label);
                legendContainer.appendChild(div);
            });
            
            // Set initial view
            viewer.zoomTo();
            viewer.render();
        });
    </script>
</body>
</html>"""

    # Write the HTML file
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Enhanced 3D visualization saved to {output_file}")
    return output_file


