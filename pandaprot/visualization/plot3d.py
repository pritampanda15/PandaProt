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
    Create a fully self-contained HTML file with reliable legend and hover functionality.
    """
    # First, get the base HTML and extract the essential model data
    base_html = view._make_html()
    
    # Extract the model data (this is the crucial part from py3Dmol)
    import re
    model_data_match = re.search(r'var glviewer = null;\s*\$\(document\).ready\(function\(\) \{\s*glviewer = \$3Dmol\.createViewer\(.*?, \{.*?\}\);\s*(.*?)glviewer\.render\(\);', base_html, re.DOTALL)
    
    if not model_data_match:
        print("Warning: Could not extract model data from py3Dmol output. The visualization may be incomplete.")
        model_setup_code = ""
    else:
        model_setup_code = model_data_match.group(1)
    
    # Define our custom HTML with explicit positioning and debug logging
    html = f"""<!DOCTYPE html>
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
            height: 100%; 
            width: 100%; 
            overflow: hidden;
            font-family: Arial, sans-serif;
        }}
        
        #viewer_container {{
            position: absolute;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
        }}
        
        #viewer {{
            width: 100%;
            height: 100%;
            position: relative;
        }}
        
        #legend_panel {{
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
            max-height: 80vh;
            overflow-y: auto;
            width: 220px;
        }}
        
        .legend_title {{
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }}
        
        .legend_item {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }}
        
        .color_box {{
            width: 16px;
            height: 16px;
            margin-right: 8px;
            border-radius: 3px;
        }}
        
        #controls_panel {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            z-index: 1000;
        }}
        
        .control_button {{
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 8px 15px;
            margin-right: 10px;
            cursor: pointer;
            font-weight: bold;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .control_button:hover {{
            background-color: #f5f5f5;
        }}
        
        #atom_label {{
            position: fixed;
            display: none;
            background-color: rgba(255, 255, 255, 0.9);
            border: 1px solid #ccc;
            border-radius: 3px;
            padding: 5px 8px;
            font-size: 12px;
            z-index: 1000;
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <!-- The main 3D viewer container -->
    <div id="viewer_container">
        <div id="viewer"></div>
    </div>
    
    <!-- Fixed legend panel -->
    <div id="legend_panel">
        <div class="legend_title">Interaction Types</div>
        <div class="legend_item"><div class="color_box" style="background-color: blue;"></div>Hydrogen Bonds</div>
        <div class="legend_item"><div class="color_box" style="background-color: red;"></div>Ionic Interactions</div>
        <div class="legend_item"><div class="color_box" style="background-color: yellow;"></div>Salt Bridges</div>
        <div class="legend_item"><div class="color_box" style="background-color: orange;"></div>Hydrophobic Interactions</div>
        <div class="legend_item"><div class="color_box" style="background-color: purple;"></div>Pi-Pi Stacking</div>
        <div class="legend_item"><div class="color_box" style="background-color: green;"></div>Pi-Cation</div>
        <div class="legend_item"><div class="color_box" style="background-color: teal;"></div>Cation-Pi</div>
        <div class="legend_item"><div class="color_box" style="background-color: lightseagreen;"></div>CH-Pi</div>
        <div class="legend_item"><div class="color_box" style="background-color: gold;"></div>Disulfide Bridges</div>
        <div class="legend_item"><div class="color_box" style="background-color: darkkhaki;"></div>Sulfur-Aromatic</div>
        <div class="legend_item"><div class="color_box" style="background-color: dodgerblue;"></div>Water-Mediated</div>
        <div class="legend_item"><div class="color_box" style="background-color: silver;"></div>Metal Coordination</div>
        <div class="legend_item"><div class="color_box" style="background-color: darkturquoise;"></div>Halogen Bonds</div>
        <div class="legend_item"><div class="color_box" style="background-color: mediumorchid;"></div>Amide-Aromatic</div>
        <div class="legend_item"><div class="color_box" style="background-color: lightslategray;"></div>Van der Waals</div>
        <div class="legend_item"><div class="color_box" style="background-color: hotpink;"></div>Amide-Amide</div>
    </div>
    
    <!-- Control buttons -->
    <div id="controls_panel">
        <button class="control_button" id="reset_view_btn">Reset View</button>
        <button class="control_button" id="toggle_legend_btn">Hide Legend</button>
        <button class="control_button" id="export_png_btn">Export PNG</button>
    </div>
    
    <!-- Atom label for hover -->
    <div id="atom_label"></div>
    
    <script>
        // Create viewer when the page is fully loaded
        $(document).ready(function() {{
            console.log("Document ready");
            
            // Create a new 3Dmol viewer
            var viewer = $3Dmol.createViewer($("#viewer"), {{
                backgroundAlpha: 0,
                antialias: true
            }});
            
            // Log viewer creation
            console.log("Viewer created:", viewer);
            
            // Set up the model based on extracted py3Dmol code
            try {{
                {model_setup_code}
                console.log("Model setup complete");
            }} catch(e) {{
                console.error("Error setting up model:", e);
            }}
            
            // Make atoms hoverable with slightly larger sticks
            viewer.setStyle({{}}, {{"stick": {{"radius": 0.2, "colorscheme": "whiteCarbon"}}}});
            
            // Set up hover detection
            const atomLabel = document.getElementById('atom_label');
            
            // Store original view for reset
            var originalView = viewer.getView();
            console.log("Original view stored");
            
            // Function to handle mousemove for atom hover
            function handleMouseMove(event) {{
                try {{
                    // Get atom at mouse position
                    var atom = viewer.getAtomAtPosition({{x: event.clientX, y: event.clientY}});
                    
                    if (atom) {{
                        // Display atom label
                        var labelText = atom.chain + ":" + atom.resi + " " + atom.resn + " " + atom.atom;
                        atomLabel.textContent = labelText;
                        atomLabel.style.left = (event.clientX + 15) + 'px';
                        atomLabel.style.top = (event.clientY - 15) + 'px';
                        atomLabel.style.display = 'block';
                    }} else {{
                        // Hide atom label
                        atomLabel.style.display = 'none';
                    }}
                }} catch(e) {{
                    console.error("Error in hover:", e);
                }}
            }}
            
            // Add mousemove listener to the viewer container
            document.getElementById('viewer_container').addEventListener('mousemove', handleMouseMove);
            
            // Reset view button
            document.getElementById('reset_view_btn').addEventListener('click', function() {{
                viewer.setView(originalView);
                viewer.render();
                console.log("View reset");
            }});
            
            // Toggle legend button
            document.getElementById('toggle_legend_btn').addEventListener('click', function() {{
                var legend = document.getElementById('legend_panel');
                var button = document.getElementById('toggle_legend_btn');
                
                if (legend.style.display === 'none') {{
                    legend.style.display = 'block';
                    button.textContent = 'Hide Legend';
                }} else {{
                    legend.style.display = 'none';
                    button.textContent = 'Show Legend';
                }}
                console.log("Legend toggled");
            }});
            
            // Export PNG button
            document.getElementById('export_png_btn').addEventListener('click', function() {{
                try {{
                    var canvas = document.querySelector('#viewer canvas');
                    if (canvas) {{
                        var link = document.createElement('a');
                        link.download = 'protein_structure.png';
                        link.href = canvas.toDataURL('image/png');
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                        console.log("PNG exported");
                    }} else {{
                        console.error("Canvas not found for export");
                    }}
                }} catch(e) {{
                    console.error("Error exporting PNG:", e);
                    alert("Error exporting image. This might be due to security restrictions.");
                }}
            }});
            
            // Final render
            viewer.render();
            console.log("Initial render complete");
            
            // Add a small delay and re-render to ensure everything is displayed properly
            setTimeout(function() {{
                viewer.render();
                console.log("Secondary render complete");
            }}, 500);
        }});
    </script>
</body>
</html>
"""
    
    # Write the HTML to the output file
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Enhanced visualization saved to {output_file}")