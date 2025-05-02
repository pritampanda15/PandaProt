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
    view.setStyle({'cartoon': {'color': 'spectrum'}})

    interaction_colors = {
        'hydrogen_bonds': 'blue',
        'ionic_interactions': 'red',
        'salt_bridges': 'yellow',
        'hydrophobic_interactions': 'orange',
        'pi_stacking': 'purple',
        'pi_cation': 'green'
    }

    for interaction_type, interactions_list in interactions.items():
        color = interaction_colors.get(interaction_type, 'gray')

        for interaction in interactions_list:
            try:
                if interaction_type == 'hydrogen_bonds':
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

                elif interaction_type == 'hydrophobic_interactions':
                    res1_chain = interaction['chain1']
                    res1_res = int(interaction['residue1'].split()[1])
                    atom1 = interaction.get('atom1', 'CA')

                    res2_chain = interaction['chain2']
                    res2_res = int(interaction['residue2'].split()[1])
                    atom2 = interaction.get('atom2', 'CA')

                    start = structure[0][res1_chain][res1_res][atom1].coord
                    end = structure[0][res2_chain][res2_res][atom2].coord

                elif interaction_type in ['pi_stacking', 'pi_cation']:
                    chain1 = interaction['chain1']
                    res1 = int(interaction['residue1'].split()[1])
                    atoms1 = interaction.get('atoms1', ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'])

                    chain2 = interaction['chain2']
                    res2 = int(interaction['residue2'].split()[1])
                    atoms2 = interaction.get('atoms2', ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'])

                    coords1 = [structure[0][chain1][res1][atom].coord for atom in atoms1 if atom in structure[0][chain1][res1]]
                    coords2 = [structure[0][chain2][res2][atom].coord for atom in atoms2 if atom in structure[0][chain2][res2]]

                    if not coords1 or not coords2:
                        continue

                    start = np.mean(coords1, axis=0)
                    end = np.mean(coords2, axis=0)

                else:
                    continue  # Skip unsupported types

                # Draw interaction cylinder
                view.addCylinder({
                    'start': {'x': float(start[0]), 'y': float(start[1]), 'z': float(start[2])},
                    'end': {'x': float(end[0]), 'y': float(end[1]), 'z': float(end[2])},
                    'radius': 0.15,
                    'color': color,
                    'dashed': interaction_type in ['hydrogen_bonds', 'salt_bridges', 'ionic_interactions']
                })

                # Add label at midpoint
                midpoint = (start + end) / 2
                view.addLabel(interaction_type.replace('_', ' ').title(), {
                    'position': {'x': float(midpoint[0]), 'y': float(midpoint[1]), 'z': float(midpoint[2])},
                    'backgroundColor': 'white',
                    'fontColor': color,
                    'fontSize': 12,
                    'inFront': True,
                    'showBackground': True
                })

            except Exception:
                continue  # Silently skip failed interactions

    view.zoomTo()
    save_fullscreen_html(view, "3d.html")
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
def save_custom_html(view, output_file):
    # Generate the base HTML
    html = view._make_html()

    # Custom CSS for the legend
    custom_css = """
    <style>
    #legend {
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 5px;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    .legend-item {
        margin-bottom: 5px;
    }
    .legend-color {
        display: inline-block;
        width: 12px;
        height: 12px;
        margin-right: 5px;
        vertical-align: middle;
    }
    </style>
    """

    # Custom JavaScript for hover labels and legend
    custom_js = """
    <script>
    function addCustomFeatures(viewer) {
        // Define interaction types and their colors
        const interactions = {
            'Hydrogen Bonds': 'blue',
            'Ionic Interactions': 'red',
            'Salt Bridges': 'yellow',
            'Hydrophobic Interactions': 'orange',
            'Pi Stacking': 'purple',
            'Pi Cation': 'green'
        };

        // Add legend
        const legend = document.createElement('div');
        legend.id = 'legend';
        for (const [name, color] of Object.entries(interactions)) {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `<span class="legend-color" style="background-color:${color};"></span>${name}`;
            legend.appendChild(item);
        }
        document.body.appendChild(legend);

        // Add hover labels
        viewer.setHoverable({}, true,
            function(atom, viewer) {
                if (!atom.label) {
                    atom.label = viewer.addLabel(atom.resn + ":" + atom.atom, {
                        position: atom,
                        backgroundColor: 'mintcream',
                        fontColor: 'black'
                    });
                }
            },
            function(atom) {
                if (atom.label) {
                    viewer.removeLabel(atom.label);
                    delete atom.label;
                }
            }
        );
        viewer.render();
    }

    // Wait for the viewer to be ready
    window.addEventListener('load', function() {
        if (typeof viewer !== 'undefined') {
            addCustomFeatures(viewer);
        }
    });
    </script>
    """

    # Inject custom CSS and JS before closing </body> tag
    html = html.replace('</body>', custom_css + custom_js + '</body>')

    # Save the modified HTML
    with open(output_file, 'w') as f:
        f.write(html)

