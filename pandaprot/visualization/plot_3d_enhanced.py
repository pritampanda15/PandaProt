from typing import Dict, List, Any
import py3Dmol
from Bio.PDB import Structure
import numpy as np


def create_3d_visualization(pdb_file: str,
                           structure: Structure,
                           interactions: Dict[str, List[Dict]],
                           output_file: str = "3d.html") -> Any:
    """
    Create a full-screen interactive 3D visualization of the protein structure with labeled interactions.
    
    Args:
        pdb_file: Path to PDB file
        structure: Parsed structure object
        interactions: Dictionary of interactions
        output_file: Output HTML file path
        
    Returns:
        py3Dmol view object
    """
    view = py3Dmol.view(width='100%', height='100vh')

    # Load and display the protein
    with open(pdb_file, 'r') as f:
        pdb_data = f.read()
    view.addModel(pdb_data, 'pdb')
    
    # Set style with chain coloring
    chains = set()
    for model in structure:
        for chain in model:
            chains.add(chain.id)
    
    # Create spectrum of colors for chains
    chain_colors = {}
    for i, chain_id in enumerate(sorted(chains)):
        # Create a color based on position in rainbow
        hue = i / max(1, len(chains)-1) * 360
        chain_colors[chain_id] = f'hsl({hue}, 80%, 60%)'
    
    # Apply styles to each chain
    for chain_id, color in chain_colors.items():
        view.setStyle({'chain': chain_id}, {
            'cartoon': {'color': color},
            'stick': {'colorscheme': 'yellowCarbon', 'radius': 0.15}
        })

    interaction_colors = {
        'hydrogen_bonds': 'blue',
        'ionic_interactions': 'red',
        'salt_bridges': 'yellow',
        'hydrophobic_interactions': 'orange',
        'pi_stacking': 'purple',
        'pi_cation': 'green',
        'ch_pi': 'magenta',
        'amide_aromatic': 'teal',
        'amide_amide': 'lime',
        'disulfide_bridges': 'gold',
        'cation_pi': 'darkgreen',
        'sulfur_aromatic': 'darkblue',
        'water_mediated': 'cyan',
        'halogen_bonds': 'brown',
        'metal_coordination': 'silver',
        'van_der_waals': 'lightgray'
    }

    # Track drawn interactions for hover data
    interaction_data = []

    for interaction_type, interactions_list in interactions.items():
        color = interaction_colors.get(interaction_type, 'gray')

        for interaction in interactions_list:
            try:
                # Try to extract residue info based on interaction type
                start, end, details = extract_interaction_coords(interaction_type, interaction, structure)
                
                if start is None or end is None:
                    continue

                # Draw interaction cylinder
                cylinder_id = view.addCylinder({
                    'start': {'x': float(start[0]), 'y': float(start[1]), 'z': float(start[2])},
                    'end': {'x': float(end[0]), 'y': float(end[1]), 'z': float(end[2])},
                    'radius': 0.15,
                    'color': color,
                    'dashed': interaction_type in ['hydrogen_bonds', 'salt_bridges', 'ionic_interactions']
                })

                # Store interaction data for hover info
                midpoint = (start + end) / 2
                interaction_data.append({
                    'type': interaction_type,
                    'position': {
                        'x': float(midpoint[0]), 
                        'y': float(midpoint[1]), 
                        'z': float(midpoint[2])
                    },
                    'color': color,
                    'details': details
                })

            except Exception as e:
                print(f"Error drawing {interaction_type}: {str(e)}")
                continue  # Skip failed interactions

    view.zoomTo()
    
    # Use the custom HTML saving function
    save_enhanced_html(view, output_file, interaction_data, interaction_colors, chain_colors)
    return view

def extract_interaction_coords(interaction_type, interaction, structure):
    """
    Extract coordinates and details for an interaction
    
    Returns:
        tuple: (start_coords, end_coords, details_dict)
    """
    details = {'type': interaction_type}
    
    try:
        if interaction_type == 'hydrogen_bonds':
            donor_chain = interaction['donor_chain']
            donor_res = int(interaction['donor_residue'].split()[1])
            donor_atom = interaction['donor_atom']
            acceptor_chain = interaction['acceptor_chain']
            acceptor_res = int(interaction['acceptor_residue'].split()[1])
            acceptor_atom = interaction['acceptor_atom']
            
            details.update({
                'donor': f"{interaction['donor_residue']} ({donor_chain})",
                'acceptor': f"{interaction['acceptor_residue']} ({acceptor_chain})",
                'distance': interaction.get('distance', 'N/A'),
                'angle': interaction.get('angle', 'N/A')
            })

            start = structure[0][donor_chain][donor_res][donor_atom].coord
            end = structure[0][acceptor_chain][acceptor_res][acceptor_atom].coord

        elif interaction_type in ['salt_bridges', 'ionic_interactions']:
            try:
                pos_chain = interaction['basic_chain']
                pos_res = int(interaction['basic_residue'].split()[1])
                pos_atom = interaction['basic_atom']
                neg_chain = interaction['acidic_chain']
                neg_res = int(interaction['acidic_residue'].split()[1])
                neg_atom = interaction['acidic_atom']
                
                details.update({
                    'basic': f"{interaction['basic_residue']} ({pos_chain})",
                    'acidic': f"{interaction['acidic_residue']} ({neg_chain})",
                    'distance': interaction.get('distance', 'N/A')
                })
                
                start = structure[0][pos_chain][pos_res][pos_atom].coord
                end = structure[0][neg_chain][neg_res][neg_atom].coord
            except KeyError:
                # Try alternative keys
                pos_chain = interaction.get('positive_chain', interaction.get('chain1', ''))
                pos_res = int(interaction.get('positive_residue', interaction.get('residue1', '')).split()[1])
                neg_chain = interaction.get('negative_chain', interaction.get('chain2', ''))
                neg_res = int(interaction.get('negative_residue', interaction.get('residue2', '')).split()[1])
                
                details.update({
                    'positive': f"{interaction.get('positive_residue', interaction.get('residue1', ''))} ({pos_chain})",
                    'negative': f"{interaction.get('negative_residue', interaction.get('residue2', ''))} ({neg_chain})",
                    'distance': interaction.get('distance', 'N/A')
                })
                
                # Use CA atoms if specific atoms not found
                start = structure[0][pos_chain][pos_res]['CA'].coord
                end = structure[0][neg_chain][neg_res]['CA'].coord

        elif interaction_type == 'hydrophobic_interactions':
            res1_chain = interaction.get('chain1', '')
            res1_res = int(interaction.get('residue1', '').split()[1])
            atom1 = interaction.get('atom1', 'CA')

            res2_chain = interaction.get('chain2', '')
            res2_res = int(interaction.get('residue2', '').split()[1])
            atom2 = interaction.get('atom2', 'CA')
            
            details.update({
                'residue1': f"{interaction.get('residue1', '')} ({res1_chain})",
                'residue2': f"{interaction.get('residue2', '')} ({res2_chain})",
                'distance': interaction.get('distance', 'N/A')
            })

            start = structure[0][res1_chain][res1_res][atom1].coord
            end = structure[0][res2_chain][res2_res][atom2].coord

        elif interaction_type in ['pi_stacking', 'pi_cation', 'ch_pi', 'cation_pi']:
            chain1 = interaction.get('chain1', '')
            res1 = int(interaction.get('residue1', '').split()[1])
            chain2 = interaction.get('chain2', '')
            res2 = int(interaction.get('residue2', '').split()[1])
            
            details.update({
                'residue1': f"{interaction.get('residue1', '')} ({chain1})",
                'residue2': f"{interaction.get('residue2', '')} ({chain2})",
                'distance': interaction.get('distance', 'N/A'),
                'angle': interaction.get('angle', 'N/A')
            })
            
            # Try to get aromatic ring centers or CA atoms
            try:
                aromatic_atoms = ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']
                coords1 = [structure[0][chain1][res1][atom].coord for atom in aromatic_atoms 
                          if atom in structure[0][chain1][res1]]
                coords2 = [structure[0][chain2][res2][atom].coord for atom in aromatic_atoms 
                          if atom in structure[0][chain2][res2]]
                
                if coords1 and coords2:
                    start = np.mean(coords1, axis=0)
                    end = np.mean(coords2, axis=0)
                else:
                    # Fall back to CA atoms
                    start = structure[0][chain1][res1]['CA'].coord
                    end = structure[0][chain2][res2]['CA'].coord
            except KeyError:
                # Fall back to CA atoms
                start = structure[0][chain1][res1]['CA'].coord
                end = structure[0][chain2][res2]['CA'].coord
                
        else:
            # Generic handling for other interaction types
            try:
                chain1 = interaction.get('chain1', '')
                res1 = int(interaction.get('residue1', '').split()[1])
                chain2 = interaction.get('chain2', '')
                res2 = int(interaction.get('residue2', '').split()[1])
                
                details.update({
                    'residue1': f"{interaction.get('residue1', '')} ({chain1})",
                    'residue2': f"{interaction.get('residue2', '')} ({chain2})",
                    'distance': interaction.get('distance', 'N/A')
                })
                
                # Use CA atoms for generic interactions
                start = structure[0][chain1][res1]['CA'].coord
                end = structure[0][chain2][res2]['CA'].coord
            except (KeyError, ValueError):
                return None, None, {}
        
        return start, end, details
        
    except Exception as e:
        print(f"Error extracting coordinates for {interaction_type}: {str(e)}")
        return None, None, {}

# def save_enhanced_html(view, output_file, interaction_data, interaction_colors, chain_colors):
#     """
#     Save an enhanced HTML visualization with hover effects and detailed legend
#     """
#     # Generate the base HTML
#     html = view._make_html()
    
#     # Convert NumPy types to standard Python types for JSON serialization
#     def convert_to_serializable(obj):
#         if isinstance(obj, dict):
#             return {k: convert_to_serializable(v) for k, v in obj.items()}
#         elif isinstance(obj, list):
#             return [convert_to_serializable(item) for item in obj]
#         elif hasattr(obj, 'item'):  # Handle NumPy scalar types
#             return obj.item()
#         elif hasattr(obj, 'tolist'):  # Handle NumPy arrays
#             return obj.tolist()
#         else:
#             return obj
    
#     # Convert interaction data to JSON for JavaScript
#     import json
#     serializable_interaction_data = convert_to_serializable(interaction_data)
#     interaction_json = json.dumps(serializable_interaction_data)
#     interaction_colors_json = json.dumps(interaction_colors)
#     chain_colors_json = json.dumps(chain_colors)
    
#     # Rest of the function remains the same...
    
#     # Custom CSS for the legend and hover info
#     custom_css = """
#     <style>
#     #legend {
#         position: absolute;
#         top: 10px;
#         right: 10px;
#         background-color: rgba(255, 255, 255, 0.9);
#         padding: 10px;
#         border-radius: 5px;
#         box-shadow: 0 2px 5px rgba(0,0,0,0.2);
#         font-family: Arial, sans-serif;
#         font-size: 14px;
#         max-height: 80vh;
#         overflow-y: auto;
#         z-index: 1000;
#     }
#     .legend-section {
#         margin-bottom: 10px;
#     }
#     .legend-header {
#         font-weight: bold;
#         margin-bottom: 5px;
#         border-bottom: 1px solid #ccc;
#     }
#     .legend-item {
#         margin: 3px 0;
#         cursor: pointer;
#         padding: 2px;
#         border-radius: 3px;
#     }
#     .legend-item:hover {
#         background-color: #f0f0f0;
#     }
#     .legend-color {
#         display: inline-block;
#         width: 12px;
#         height: 12px;
#         margin-right: 5px;
#         vertical-align: middle;
#         border-radius: 2px;
#     }
#     .interaction-count {
#         font-size: 0.8em;
#         color: #666;
#         margin-left: 5px;
#     }
#     #hover-info {
#         display: none;
#         position: absolute;
#         background-color: rgba(255, 255, 255, 0.9);
#         padding: 10px;
#         border-radius: 5px;
#         box-shadow: 0 2px 5px rgba(0,0,0,0.2);
#         font-family: Arial, sans-serif;
#         font-size: 14px;
#         z-index: 1001;
#         max-width: 300px;
#     }
#     .hover-title {
#         font-weight: bold;
#         margin-bottom: 5px;
#         border-bottom: 1px solid #ccc;
#     }
#     .hover-detail {
#         margin: 3px 0;
#     }
#     .hover-value {
#         font-weight: bold;
#     }
#     #toggle-legend {
#         position: absolute;
#         top: 10px;
#         right: 10px;
#         background-color: rgba(255, 255, 255, 0.9);
#         padding: 5px 10px;
#         border-radius: 5px;
#         box-shadow: 0 2px 5px rgba(0,0,0,0.2);
#         font-family: Arial, sans-serif;
#         font-size: 14px;
#         cursor: pointer;
#         z-index: 1002;
#         display: none;
#     }
#     </style>
#     """

#     # Custom JavaScript for hover labels, interactive legend, and hover info
#     custom_js = f"""
#     <script>
#     // Store interaction data
#     const interactionData = {interaction_json};
#     const interactionColors = {interaction_colors_json};
#     const chainColors = {chain_colors_json};
    
#     // Wait for the viewer to be ready
#     window.addEventListener('load', function() {{
#         if (typeof viewer !== 'undefined') {{
#             enhanceViewer(viewer);
#         }}
#     }});
    
#     function enhanceViewer(viewer) {{
#         // Create legend
#         createLegend();
        
#         // Add atom hover effect
#         addAtomHoverEffect(viewer);
        
#         // Add interaction hover effect
#         addInteractionHoverEffect(viewer);
#     }}
    
#     function createLegend() {{
#         // Create legend container
#         const legend = document.createElement('div');
#         legend.id = 'legend';
        
#         // Create chain section
#         const chainSection = document.createElement('div');
#         chainSection.className = 'legend-section';
        
#         const chainHeader = document.createElement('div');
#         chainHeader.className = 'legend-header';
#         chainHeader.textContent = 'Chains';
#         chainSection.appendChild(chainHeader);
        
#         for (const [chainId, color] of Object.entries(chainColors)) {{
#             const item = document.createElement('div');
#             item.className = 'legend-item';
#             item.innerHTML = `<span class="legend-color" style="background-color:${{color}};"></span>Chain ${{chainId}}`;
#             item.onclick = function() {{
#                 highlightChain(chainId);
#             }};
#             chainSection.appendChild(item);
#         }}
        
#         legend.appendChild(chainSection);
        
#         // Create interaction section with counts
#         const interactionSection = document.createElement('div');
#         interactionSection.className = 'legend-section';
        
#         const interactionHeader = document.createElement('div');
#         interactionHeader.className = 'legend-header';
#         interactionHeader.textContent = 'Interactions';
#         interactionSection.appendChild(interactionHeader);
        
#         // Count interactions by type
#         const interactionCounts = {{}};
#         interactionData.forEach(interaction => {{
#             const type = interaction.type;
#             interactionCounts[type] = (interactionCounts[type] || 0) + 1;
#         }});
        
#         for (const [type, color] of Object.entries(interactionColors)) {{
#             const count = interactionCounts[type] || 0;
#             if (count > 0) {{
#                 const item = document.createElement('div');
#                 item.className = 'legend-item';
#                 const displayName = type.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
#                 item.innerHTML = `<span class="legend-color" style="background-color:${{color}};"></span>${{displayName}} <span class="interaction-count">(${{count}})</span>`;
#                 item.onclick = function() {{
#                     highlightInteractionType(type);
#                 }};
#                 interactionSection.appendChild(item);
#             }}
#         }}
        
#         legend.appendChild(interactionSection);
        
#         // Add toggle button
#         const toggleButton = document.createElement('div');
#         toggleButton.id = 'toggle-legend';
#         toggleButton.textContent = 'Show Legend';
#         toggleButton.onclick = function() {{
#             if (legend.style.display === 'none') {{
#                 legend.style.display = 'block';
#                 this.textContent = 'Hide Legend';
#             }} else {{
#                 legend.style.display = 'none';
#                 this.textContent = 'Show Legend';
#             }}
#         }};
        
#         document.body.appendChild(legend);
#         document.body.appendChild(toggleButton);
        
#         // Create hover info container
#         const hoverInfo = document.createElement('div');
#         hoverInfo.id = 'hover-info';
#         document.body.appendChild(hoverInfo);
#     }}
    
#     function addAtomHoverEffect(viewer) {{
#         viewer.setHoverable({{}}, true, 
#             function(atom, viewer) {{
#                 if (!atom.label) {{
#                     const residueName = atom.resn;
#                     const residueNumber = atom.resi;
#                     const chain = atom.chain;
#                     const atomName = atom.atom;
                    
#                     const labelText = `${{residueName}} ${{residueNumber}}:${{atomName}} (Chain ${{chain}})`;
#                     atom.label = viewer.addLabel(labelText, {{
#                         position: atom,
#                         backgroundColor: 'mintcream',
#                         fontColor: 'black',
#                         fontSize: 12,
#                         padding: 2
#                     }});
#                 }}
#             }},
#             function(atom) {{
#                 if (atom.label) {{
#                     viewer.removeLabel(atom.label);
#                     delete atom.label;
#                 }}
#             }}
#         );
#     }}
    
#     function addInteractionHoverEffect(viewer) {{
#         // Add hover info for interactions
#         const hoverInfo = document.getElementById('hover-info');
#         const canvas = document.getElementById('canvas');
        
#         canvas.addEventListener('mousemove', function(event) {{
#             const rect = canvas.getBoundingClientRect();
#             const x = event.clientX - rect.left;
#             const y = event.clientY - rect.top;
            
#             // Convert to 3D coordinates
#             const pos = viewer.screenToModel(x, y);
            
#             // Check if near any interaction
#             const nearestInteraction = findNearestInteraction(pos, 3.0); // threshold in Angstroms
            
#             if (nearestInteraction) {{
#                 // Show hover info
#                 showInteractionInfo(nearestInteraction, event.clientX, event.clientY);
#             }} else {{
#                 // Hide hover info
#                 hoverInfo.style.display = 'none';
#             }}
#         }});
#     }}
    
#     function findNearestInteraction(position, threshold) {{
#         let nearestInteraction = null;
#         let minDistance = threshold;
        
#         interactionData.forEach(interaction => {{
#             const dx = position.x - interaction.position.x;
#             const dy = position.y - interaction.position.y;
#             const dz = position.z - interaction.position.z;
#             const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);
            
#             if (distance < minDistance) {{
#                 minDistance = distance;
#                 nearestInteraction = interaction;
#             }}
#         }});
        
#         return nearestInteraction;
#     }}
    
#     function showInteractionInfo(interaction, x, y) {{
#         const hoverInfo = document.getElementById('hover-info');
        
#         // Position the hover info
#         hoverInfo.style.display = 'block';
#         hoverInfo.style.left = `${{x + 10}}px`;
#         hoverInfo.style.top = `${{y + 10}}px`;
        
#         // Format the type for display
#         const displayType = interaction.type.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
        
#         // Create content for hover info
#         let content = `<div class="hover-title" style="color:${{interaction.color}}">${{displayType}}</div>`;
        
#         for (const [key, value] of Object.entries(interaction.details)) {{
#             if (key !== 'type') {{
#                 const displayKey = key.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
#                 content += `<div class="hover-detail"><span class="hover-label">${{displayKey}}:</span> <span class="hover-value">${{value}}</span></div>`;
#             }}
#         }}
        
#         hoverInfo.innerHTML = content;
#     }}
    
#     function highlightChain(chainId) {{
#         // Implement chain highlighting
#         // This would require more integration with the 3Dmol.js API
#         console.log(`Highlighting chain ${{chainId}}`);
#     }}
    
#     function highlightInteractionType(type) {{
#         // Implement interaction type highlighting
#         // This would require more integration with the 3Dmol.js API
#         console.log(`Highlighting interactions of type ${{type}}`);
#     }}
#     </script>
#     """

#     # Inject custom CSS and JS before closing </body> tag
#     html = html.replace('</body>', custom_css + custom_js + '</body>')

#     # Save the modified HTML
#     with open(output_file, 'w') as f:
#         f.write(html)

def save_enhanced_html(view, output_file, interaction_data, interaction_colors, chain_colors):
    # Generate the base HTML
    html = view._make_html()
    
    # Convert NumPy types to standard Python types
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):  # Handle NumPy scalar types
            return obj.item()
        elif hasattr(obj, 'tolist'):  # Handle NumPy arrays
            return obj.tolist()
        else:
            return obj
    
    # Convert interaction data to JSON for JavaScript
    import json
    serializable_interaction_data = convert_to_serializable(interaction_data)
    interaction_json = json.dumps(serializable_interaction_data)
    interaction_colors_json = json.dumps(interaction_colors)
    chain_colors_json = json.dumps(chain_colors)
    
    # Simple CSS for legend
    custom_css = """
    <style>
    #simple-legend {
        position: fixed;
        top: 20px;
        right: 20px;
        background-color: white;
        border: 1px solid black;
        padding: 10px;
        z-index: 9999;
    }
    .legend-item {
        margin: 5px 0;
    }
    .color-box {
        display: inline-block;
        width: 15px;
        height: 15px;
        margin-right: 5px;
    }
    </style>
    """
    
    # Simple JavaScript for static legend
    custom_js = f"""
    <script>
    window.addEventListener('load', function() {{
        const interactionColors = {interaction_colors_json};
        
        // Create a simple legend
        const legend = document.createElement('div');
        legend.id = 'simple-legend';
        
        // Add legend title
        const title = document.createElement('div');
        title.style.fontWeight = 'bold';
        title.textContent = 'Interaction Types';
        legend.appendChild(title);
        
        // Add legend items
        for (const [type, color] of Object.entries(interactionColors)) {{
            const item = document.createElement('div');
            item.className = 'legend-item';
            
            const colorBox = document.createElement('span');
            colorBox.className = 'color-box';
            colorBox.style.backgroundColor = color;
            
            const text = document.createTextNode(type.replace(/_/g, ' '));
            
            item.appendChild(colorBox);
            item.appendChild(text);
            legend.appendChild(item);
        }}
        
        // Add to document
        document.body.appendChild(legend);
        
        // Basic hover labels for atoms (simplified)
        if (typeof viewer !== 'undefined') {{
            viewer.setHoverable({{}}, true,
                function(atom, viewer) {{
                    if (!atom.label) {{
                        atom.label = viewer.addLabel(atom.resn + ":" + atom.atom, {{
                            position: atom,
                            backgroundColor: 'white',
                            fontColor: 'black'
                        }});
                    }}
                }},
                function(atom) {{
                    if (atom.label) {{
                        viewer.removeLabel(atom.label);
                        delete atom.label;
                    }}
                }}
            );
        }}
    }});
    </script>
    """
    
    # Inject custom CSS and JS before closing </body> tag
    html = html.replace('</body>', custom_css + custom_js + '</body>')
    
    # Save the modified HTML
    with open(output_file, 'w') as f:
        f.write(html)