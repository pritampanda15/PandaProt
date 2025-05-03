from typing import Dict, List, Any
import py3Dmol
from Bio.PDB import Structure
import numpy as np


def create_3d_visualization(pdb_file: str,
                           structure: Structure,
                           interactions: Dict[str, List[Dict]]) -> Any:
    """
    Create a full-screen interactive 3D visualization of the protein structure with labeled interactions.
    
    Args:
        pdb_file: Path to PDB file
        structure: Parsed structure object
        interactions: Dictionary of interactions
        
    Returns:
        py3Dmol view object
    """
    view = py3Dmol.view(width='100%', height='100vh')

    # Load and display the protein
    with open(pdb_file, 'r') as f:
        pdb_data = f.read()
    view.addModel(pdb_data, 'pdb')
    
    # Set protein style
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.addStyle({}, {"stick": {"radius": 0.1, "colorscheme": "whiteCarbon"}})  # Makes atoms hoverable

    # Define colors for all interaction types
    interaction_colors = {
        'hydrogen_bonds': 'blue',
        'ionic_interactions': 'red',
        'salt_bridges': 'yellow',
        'hydrophobic_interactions': 'orange',
        'pi_stacking': 'purple',
        'pi_cation': 'green',
        'cation_pi': 'teal',
        'ch_pi': 'lightseagreen',
        'disulfide': 'gold',
        'sulfur_aromatic': 'darkkhaki',
        'water_mediated': 'dodgerblue',
        'metal_coordination': 'silver',
        'halogen_bonds': 'darkturquoise',
        'amide_aromatic': 'mediumorchid',
        'van_der_waals': 'lightslategray',
        'amide_amide': 'hotpink'
    }

    # Process each interaction type
    for interaction_type, interactions_list in interactions.items():
        color = interaction_colors.get(interaction_type, 'gray')
        dashed = interaction_type in ['hydrogen_bonds', 'salt_bridges', 'ionic_interactions', 
                                     'halogen_bonds', 'water_mediated', 'amide_amide']
        
        for interaction in interactions_list:
            try:
                # Get coordinates based on interaction type
                if interaction_type == 'hydrogen_bonds' or interaction_type == 'amide_amide':
                    donor_chain = interaction['donor_chain']
                    donor_res = int(interaction['donor_residue'].split()[1])
                    donor_atom = interaction['donor_atom']
                    acceptor_chain = interaction['acceptor_chain']
                    acceptor_res = int(interaction['acceptor_residue'].split()[1])
                    acceptor_atom = interaction['acceptor_atom']

                    start = structure[0][donor_chain][donor_res][donor_atom].coord
                    end = structure[0][acceptor_chain][acceptor_res][acceptor_atom].coord

                elif interaction_type in ['salt_bridges', 'ionic_interactions']:
                    pos_chain = interaction['basic_chain']
                    pos_res = int(interaction['basic_residue'].split()[1])
                    pos_atom = interaction['basic_atom']
                    neg_chain = interaction['acidic_chain']
                    neg_res = int(interaction['acidic_residue'].split()[1])
                    neg_atom = interaction['acidic_atom']

                    start = structure[0][pos_chain][pos_res][pos_atom].coord
                    end = structure[0][neg_chain][neg_res][neg_atom].coord

                elif interaction_type in ['hydrophobic_interactions', 'van_der_waals']:
                    res1_chain = interaction['chain1']
                    res1_res = int(interaction['residue1'].split()[1])
                    atom1 = interaction.get('atom1', 'CA')

                    res2_chain = interaction['chain2']
                    res2_res = int(interaction['residue2'].split()[1])
                    atom2 = interaction.get('atom2', 'CA')

                    start = structure[0][res1_chain][res1_res][atom1].coord
                    end = structure[0][res2_chain][res2_res][atom2].coord

                elif interaction_type in ['pi_stacking', 'pi_cation', 'cation_pi', 'ch_pi', 'sulfur_aromatic', 'amide_aromatic']:
                    chain1 = interaction['chain1']
                    res1 = int(interaction['residue1'].split()[1])
                    atoms1 = interaction.get('atoms1', ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'])

                    chain2 = interaction['chain2']
                    res2 = int(interaction['residue2'].split()[1])
                    atoms2 = interaction.get('atoms2', ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'])

                    # Try to get coordinates, falling back to CA atoms if needed
                    try:
                        coords1 = [structure[0][chain1][res1][atom].coord for atom in atoms1 if atom in structure[0][chain1][res1]]
                        if not coords1:
                            coords1 = [structure[0][chain1][res1]['CA'].coord]
                    except:
                        coords1 = [structure[0][chain1][res1]['CA'].coord]
                        
                    try:
                        coords2 = [structure[0][chain2][res2][atom].coord for atom in atoms2 if atom in structure[0][chain2][res2]]
                        if not coords2:
                            coords2 = [structure[0][chain2][res2]['CA'].coord]
                    except:
                        coords2 = [structure[0][chain2][res2]['CA'].coord]

                    start = np.mean(coords1, axis=0)
                    end = np.mean(coords2, axis=0)

                elif interaction_type == 'disulfide':
                    chain1 = interaction['chain1']
                    res1 = int(interaction['residue1'].split()[1])
                    atom1 = interaction.get('atom1', 'SG')

                    chain2 = interaction['chain2']
                    res2 = int(interaction['residue2'].split()[1])
                    atom2 = interaction.get('atom2', 'SG')

                    start = structure[0][chain1][res1][atom1].coord
                    end = structure[0][chain2][res2][atom2].coord

                elif interaction_type == 'water_mediated':
                    # For water-mediated interactions, we need to handle three points
                    # This is simplified to show just a connection to the water
                    chain1 = interaction['chain1']
                    res1 = int(interaction['residue1'].split()[1])
                    atom1 = interaction.get('atom1', 'O')
                    
                    water_chain = interaction.get('water_chain', 'W')
                    water_res = int(interaction.get('water_residue', '0').split()[1])
                    water_atom = interaction.get('water_atom', 'O')
                    
                    chain2 = interaction.get('chain2', None)
                    res2 = None if chain2 is None else int(interaction.get('residue2', '0').split()[1])
                    atom2 = interaction.get('atom2', 'O')

                    start = structure[0][chain1][res1][atom1].coord
                    water_coord = structure[0][water_chain][water_res][water_atom].coord
                    
                    # First connection: residue 1 to water
                    view.addCylinder({
                        'start': {'x': float(start[0]), 'y': float(start[1]), 'z': float(start[2])},
                        'end': {'x': float(water_coord[0]), 'y': float(water_coord[1]), 'z': float(water_coord[2])},
                        'radius': 0.15,
                        'color': color,
                        'dashed': True
                    })
                    
                    # If there's a second residue, add connection from water to residue 2
                    if chain2 is not None and res2 is not None:
                        end = structure[0][chain2][res2][atom2].coord
                        view.addCylinder({
                            'start': {'x': float(water_coord[0]), 'y': float(water_coord[1]), 'z': float(water_coord[2])},
                            'end': {'x': float(end[0]), 'y': float(end[1]), 'z': float(end[2])},
                            'radius': 0.15,
                            'color': color,
                            'dashed': True
                        })
                    
                    # Highlight the water molecule
                    view.addSphere({
                        'center': {'x': float(water_coord[0]), 'y': float(water_coord[1]), 'z': float(water_coord[2])},
                        'radius': 0.4,
                        'color': color,
                        'opacity': 0.7
                    })
                    
                    # Skip the normal cylinder addition that happens outside this if/else block
                    continue

                elif interaction_type == 'metal_coordination':
                    metal_chain = interaction['metal_chain']
                    metal_res = int(interaction['metal_residue'].split()[1])
                    metal_atom = interaction['metal_atom']
                    
                    ligand_chain = interaction['ligand_chain']
                    ligand_res = int(interaction['ligand_residue'].split()[1])
                    ligand_atom = interaction['ligand_atom']
                    
                    metal_coord = structure[0][metal_chain][metal_res][metal_atom].coord
                    ligand_coord = structure[0][ligand_chain][ligand_res][ligand_atom].coord
                    
                    # Highlight the metal atom with a sphere
                    view.addSphere({
                        'center': {'x': float(metal_coord[0]), 'y': float(metal_coord[1]), 'z': float(metal_coord[2])},
                        'radius': 0.5,
                        'color': 'silver',
                        'opacity': 0.8
                    })
                    
                    start = metal_coord
                    end = ligand_coord

                elif interaction_type == 'halogen_bonds':
                    donor_chain = interaction['donor_chain']
                    donor_res = int(interaction['donor_residue'].split()[1])
                    donor_atom = interaction['donor_atom']
                    
                    acceptor_chain = interaction['acceptor_chain']
                    acceptor_res = int(interaction['acceptor_residue'].split()[1])
                    acceptor_atom = interaction['acceptor_atom']
                    
                    start = structure[0][donor_chain][donor_res][donor_atom].coord
                    end = structure[0][acceptor_chain][acceptor_res][acceptor_atom].coord
                
                else:
                    # Skip unsupported types
                    print(f"Skipping unsupported interaction type: {interaction_type}")
                    continue

                # Draw interaction cylinder (for all types except those that handled their own visualization)
                view.addCylinder({
                    'start': {'x': float(start[0]), 'y': float(start[1]), 'z': float(start[2])},
                    'end': {'x': float(end[0]), 'y': float(end[1]), 'z': float(end[2])},
                    'radius': 0.15,
                    'color': color,
                    'dashed': dashed
                })

            except Exception as e:
                print(f"Error visualizing {interaction_type}: {e}")
                continue  # Skip failed interactions

    # Set initial view
    view.zoomTo()
    return view

def save_fullscreen_html(view, output_file: str, title: str = "3D Structure"):
    html = view._make_html().replace(
        '<div id="viewer" style="width:800px; height:600px;',
        '<div id="viewer" style="width:100%; height:100vh; position:absolute; top:0; left:0;'
    )
    html = html.replace(
        '<body>',
        f'<body style="margin:0; overflow:hidden;"><h1 style="display:none">{title}</h1>'
    )
    with open(output_file, 'w') as f:
        f.write(html)
def save_custom_html(view, output_file, title="Protein Interaction Visualization"):
    """
    Save the py3Dmol view as a custom HTML file with reliable legends and controls.
    
    Args:
        view: py3Dmol view object
        output_file: Path to save the HTML file
        title: Title for the HTML page
    """
    # Get the base HTML from py3Dmol
    html = view._make_html()
    
    # Define the interaction colors that match your visualization
    interaction_colors = {
        'Hydrogen Bonds': 'blue',
        'Ionic Interactions': 'red',
        'Salt Bridges': 'yellow',
        'Hydrophobic Interactions': 'orange',
        'Pi-Pi Stacking': 'purple',
        'Pi-Cation': 'green',
        'Cation-Pi': 'teal',
        'CH-Pi': 'lightseagreen',
        'Disulfide Bridges': 'gold',
        'Sulfur-Aromatic': 'darkkhaki',
        'Water-Mediated': 'dodgerblue',
        'Metal Coordination': 'silver',
        'Halogen Bonds': 'darkturquoise',
        'Amide-Aromatic': 'mediumorchid',
        'Van der Waals': 'lightslategray',
        'Amide-Amide': 'hotpink'
    }
    
    # Create the legend HTML
    legend_items = ''
    for name, color in interaction_colors.items():
        legend_items += f'<div class="legend-item"><span class="legend-color" style="background-color:{color};"></span>{name}</div>\n'
    
    # Complete HTML document with embedded viewer, legend, and controls
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol.js/2.0.3/3Dmol-min.js"></script>
    <style>
        body, html {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            height: 100%;
            width: 100%;
            font-family: Arial, sans-serif;
        }}
        #container {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
        }}
        #legend {{
            position: absolute;
            top: 20px;
            right: 20px;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 8px;
            font-size: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 100;
            max-height: 80vh;
            overflow-y: auto;
        }}
        .legend-header {{
            font-weight: bold;
            margin-bottom: 10px;
            text-align: center;
        }}
        .legend-item {{
            margin-bottom: 8px;
            display: flex;
            align-items: center;
        }}
        .legend-color {{
            display: inline-block;
            width: 16px;
            height: 16px;
            margin-right: 10px;
            border-radius: 3px;
        }}
        #controls {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            z-index: 100;
        }}
        .btn {{
            background-color: rgba(255, 255, 255, 0.9);
            border: none;
            padding: 10px 15px;
            margin-right: 10px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            transition: all 0.3s;
        }}
        .btn:hover {{
            background-color: rgba(240, 240, 240, 1);
            transform: translateY(-2px);
        }}
        #atom-label {{
            position: absolute;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            display: none;
            z-index: 200;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        #toggle-btn {{
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 101;
            background-color: rgba(255, 255, 255, 0.9);
            border: none;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    
    <div id="legend">
        <div class="legend-header">Interaction Types</div>
        {legend_items}
    </div>
    
    <button id="toggle-btn">Hide Legend</button>
    
    <div id="controls">
        <button class="btn" id="reset-btn">Reset View</button>
        <button class="btn" id="export-btn">Export PNG</button>
    </div>
    
    <div id="atom-label"></div>

    <script>
        // Store the HTML content inside a variable
        var viewerHTML = `{html.split('<div id="viewer"')[1].split('</div>')[0]}`;
        
        $(document).ready(function() {{
            // Insert the viewer
            $('#container').html('<div id="viewer"' + viewerHTML + '</div>');
            
            // Wait for the 3Dmol viewer to be initialized
            setTimeout(function() {{
                // Access the viewer
                var viewer = $3Dmol.viewers[0];
                if (!viewer) {{
                    console.error("Viewer not initialized!");
                    return;
                }}
                
                // Store the original view
                var originalView = viewer.getView();
                
                // Set up reset button
                $('#reset-btn').on('click', function() {{
                    viewer.setView(originalView);
                    viewer.render();
                }});
                
                // Set up export button
                $('#export-btn').on('click', function() {{
                    var canvas = $('#viewer canvas')[0];
                    if (canvas) {{
                        try {{
                            var link = document.createElement('a');
                            link.download = 'protein_structure.png';
                            link.href = canvas.toDataURL('image/png');
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                        }} catch (e) {{
                            console.error('Error exporting image:', e);
                            alert('Error exporting image. This might be due to security restrictions.');
                        }}
                    }}
                }});
                
                // Set up legend toggle
                $('#toggle-btn').on('click', function() {{
                    if ($('#legend').is(':visible')) {{
                        $('#legend').hide();
                        $(this).text('Show Legend');
                    }} else {{
                        $('#legend').show();
                        $(this).text('Hide Legend');
                    }}
                }});
                
                // Make atoms hoverable
                viewer.setStyle({{}}, {{"stick": {{"radius": 0.15, "colorscheme": "whiteCarbon"}}}});
                
                // Set up atom hover
                var atomLabel = $('#atom-label');
                var viewerElement = $('#viewer');
                
                // Manual hover tracking since the built-in hover might not work
                viewerElement.on('mousemove', function(event) {{
                    var x = event.clientX;
                    var y = event.clientY;
                    
                    // Get atom at this position
                    var atom = viewer.getAtomAtPosition({{x: x, y: y}});
                    
                    if (atom) {{
                        var label = atom.chain + ':' + atom.resi + ' ' + atom.resn + ' ' + atom.atom;
                        atomLabel.text(label);
                        atomLabel.css({{
                            'display': 'block',
                            'left': (x + 15) + 'px',
                            'top': (y - 15) + 'px'
                        }});
                    }} else {{
                        atomLabel.css('display', 'none');
                    }}
                }});
                
                // Re-render to apply all changes
                viewer.render();
                
                console.log("Viewer setup complete!");
            }}, 1000);
        }});
    </script>
</body>
</html>
"""
    
    # Write the complete HTML to file
    with open(output_file, 'w') as f:
        f.write(full_html)
    
    print(f"Visualization saved to {output_file}")