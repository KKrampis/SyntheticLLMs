import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure with a clean, academic style
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Define colors consistent with other figures in the paper
root_color = '#4472C4'      # Blue for root concepts
level1_color = '#70AD47'    # Green for intermediate concepts  
level2_color = '#FFC000'    # Orange for specific concepts
level3_color = '#C65911'    # Darker orange for leaf concepts

# Define positions and sizes
box_width = 1.8
box_height = 0.6
level_spacing = 1.8

# Root level (depth 0) - Animal
root_x, root_y = 4.1, 6.5
root_box = FancyBboxPatch((root_x, root_y), box_width, box_height,
                         boxstyle="round,pad=0.05", 
                         facecolor=root_color, 
                         edgecolor='black', 
                         linewidth=1.5)
ax.add_patch(root_box)
ax.text(root_x + box_width/2, root_y + box_height/2, 'Animal', 
        ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# Level 1 (depth 1) - Mammal, Bird
level1_positions = [(1.5, 4.7), (6.7, 4.7)]
level1_labels = ['Mammal', 'Bird']

level1_boxes = []
for i, (x, y) in enumerate(level1_positions):
    box = FancyBboxPatch((x, y), box_width, box_height,
                        boxstyle="round,pad=0.05",
                        facecolor=level1_color,
                        edgecolor='black',
                        linewidth=1.5)
    ax.add_patch(box)
    level1_boxes.append(box)
    ax.text(x + box_width/2, y + box_height/2, level1_labels[i],
           ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Level 2 (depth 2) - Dog, Cat under Mammal; Eagle, Sparrow under Bird
level2_positions = [(0.2, 2.9), (2.8, 2.9), (5.4, 2.9), (8.0, 2.9)]
level2_labels = ['Dog', 'Cat', 'Eagle', 'Sparrow']

level2_boxes = []
for i, (x, y) in enumerate(level2_positions):
    box = FancyBboxPatch((x, y), box_width, box_height,
                        boxstyle="round,pad=0.05",
                        facecolor=level2_color,
                        edgecolor='black',
                        linewidth=1.5)
    ax.add_patch(box)
    level2_boxes.append(box)
    ax.text(x + box_width/2, y + box_height/2, level2_labels[i],
           ha='center', va='center', fontsize=10, fontweight='bold', color='black')

# Level 3 (depth 3) - Specific breeds/species
level3_positions = [(0.2, 1.1), (2.8, 1.1), (5.4, 1.1), (8.0, 1.1)]
level3_labels = ['Golden\nRetriever', 'Persian\nCat', 'Bald\nEagle', 'House\nSparrow']

level3_boxes = []
for i, (x, y) in enumerate(level3_positions):
    box = FancyBboxPatch((x, y), box_width, box_height,
                        boxstyle="round,pad=0.05",
                        facecolor=level3_color,
                        edgecolor='black',
                        linewidth=1.5)
    ax.add_patch(box)
    level3_boxes.append(box)
    ax.text(x + box_width/2, y + box_height/2, level3_labels[i],
           ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Draw connections
# Root to Level 1
for i, (x, y) in enumerate(level1_positions):
    ax.plot([root_x + box_width/2, x + box_width/2], 
           [root_y, y + box_height], 
           'k-', linewidth=2, alpha=0.7)

# Level 1 to Level 2
# Mammal to Dog and Cat
mammal_x = level1_positions[0][0] + box_width/2
mammal_y = level1_positions[0][1]
for i in range(2):
    dog_cat_x = level2_positions[i][0] + box_width/2
    dog_cat_y = level2_positions[i][1] + box_height
    ax.plot([mammal_x, dog_cat_x], [mammal_y, dog_cat_y], 
           'k-', linewidth=2, alpha=0.7)

# Bird to Eagle and Sparrow
bird_x = level1_positions[1][0] + box_width/2
bird_y = level1_positions[1][1]
for i in range(2, 4):
    bird_species_x = level2_positions[i][0] + box_width/2
    bird_species_y = level2_positions[i][1] + box_height
    ax.plot([bird_x, bird_species_x], [bird_y, bird_species_y], 
           'k-', linewidth=2, alpha=0.7)

# Level 2 to Level 3
for i in range(4):
    level2_x = level2_positions[i][0] + box_width/2
    level2_y = level2_positions[i][1]
    level3_x = level3_positions[i][0] + box_width/2
    level3_y = level3_positions[i][1] + box_height
    ax.plot([level2_x, level3_x], [level2_y, level3_y], 
           'k-', linewidth=2, alpha=0.7)

# Add depth labels on the right
ax.text(9.5, 6.8, 'Depth 0\n(Root)', ha='center', va='center', 
        fontsize=10, style='italic', color='gray')
ax.text(9.5, 5.0, 'Depth 1\n(Categories)', ha='center', va='center', 
        fontsize=10, style='italic', color='gray')
ax.text(9.5, 3.2, 'Depth 2\n(Types)', ha='center', va='center', 
        fontsize=10, style='italic', color='gray')
ax.text(9.5, 1.4, 'Depth 3\n(Instances)', ha='center', va='center', 
        fontsize=10, style='italic', color='gray')

# Add title
ax.text(5, 7.5, 'Hierarchical Concept Organization in Synthetic Feature Trees', 
        ha='center', va='center', fontsize=14, fontweight='bold')

# Add statistics annotation
stats_text = """Tree Structure Statistics:
• 128 root features (depth 0)
• 512 features at depth 1  
• 2,048 features at depth 2
• 8,192 leaf features at depth 3
Total: 10,884 hierarchical features"""

ax.text(0.2, 0.2, stats_text, ha='left', va='bottom', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))

# Save the figure
plt.tight_layout()
plt.savefig('/Users/bioitx/Documents/GitHub/SyntheticLLMs/figures/fig1.png', 
           dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

print("Figure saved as figures/fig1.png")