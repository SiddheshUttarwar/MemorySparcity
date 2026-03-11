import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 7))

# Title
ax.text(0.5, 0.95, 'Multi-Tile Network-on-Chip (NoC) SNN Architecture', 
        fontsize=16, fontweight='bold', ha='center', color='#2c3e50', family='monospace')

def draw_tile(ax, x, y, title, color='#f0f4c3'):
    # Tile outline
    tile = patches.Rectangle((x, y), 0.25, 0.35, fill=True, color=color, ec='#2c3e50', lw=2)
    ax.add_patch(tile)
    ax.text(x+0.125, y+0.32, title, fontsize=12, fontweight='bold', ha='center', family='monospace')
    
    # Internal components (Miniaturized)
    sram = patches.Rectangle((x+0.02, y+0.18), 0.1, 0.1, fill=True, color='#ffcc80', ec='black')
    ax.add_patch(sram)
    ax.text(x+0.07, y+0.23, 'SRAM\nWeight\nBank', ha='center', va='center', fontsize=7)
    
    mac = patches.Rectangle((x+0.13, y+0.18), 0.1, 0.1, fill=True, color='#ce93d8', ec='black')
    ax.add_patch(mac)
    ax.text(x+0.18, y+0.23, 'Sparse\nMAC', ha='center', va='center', fontsize=7)
    
    lif = patches.Rectangle((x+0.02, y+0.05), 0.21, 0.1, fill=True, color='#90caf9', ec='black')
    ax.add_patch(lif)
    ax.text(x+0.125, y+0.1, 'LIF Neuron Array', ha='center', va='center', fontsize=8, fontweight='bold')

def draw_ping_pong(ax, x, y):
    box = patches.Rectangle((x, y), 0.15, 0.2, fill=True, color='#e0f7fa', ec='#006064', lw=1.5, ls='--')
    ax.add_patch(box)
    ax.text(x+0.075, y+0.16, 'Ping-Pong\nSpike Buffer', ha='center', va='center', fontsize=8, fontweight='bold')
    
    bank_a = patches.Rectangle((x+0.02, y+0.08), 0.11, 0.05, fill=True, color='#b2ebf2', ec='black')
    ax.add_patch(bank_a)
    ax.text(x+0.075, y+0.105, 'Bank A', ha='center', va='center', fontsize=7)
    
    bank_b = patches.Rectangle((x+0.02, y+0.02), 0.11, 0.05, fill=True, color='#80deea', ec='black')
    ax.add_patch(bank_b)
    ax.text(x+0.075, y+0.045, 'Bank B', ha='center', va='center', fontsize=7)

# Draw Tiles
draw_tile(ax, 0.05, 0.5, 'TILE 1 (Conv1)')
draw_tile(ax, 0.7, 0.5, 'TILE 2 (Conv2)')

draw_tile(ax, 0.05, 0.05, 'TILE 3 (FC1)')
draw_tile(ax, 0.7, 0.05, 'TILE 4 (FC2)')

# Draw Ping Pong Buffers
draw_ping_pong(ax, 0.425, 0.575) # Between T1 and T2
draw_ping_pong(ax, 0.425, 0.125) # Between T3 and T4
draw_ping_pong(ax, 0.05, 0.225) # Between T2 and T3 (Pretend routing goes here for diagram flow)

# Abstract NoC Router Mesh (Center)
router = patches.Rectangle((0.4, 0.35), 0.2, 0.15, fill=True, color='#fff59d', ec='#f57f17', lw=2)
ax.add_patch(router)
ax.text(0.5, 0.425, 'Global NoC\nRouting Mesh', ha='center', va='center', fontsize=10, fontweight='bold')

# Arrows T1 -> Buffer -> T2
ax.arrow(0.3, 0.675, 0.1, 0, head_width=0.02, head_length=0.02, fc='black', ec='black', lw=1.5)
ax.text(0.35, 0.69, 'T spikes', ha='center', fontsize=8)

ax.arrow(0.575, 0.675, 0.1, 0, head_width=0.02, head_length=0.02, fc='black', ec='black', lw=1.5)
ax.text(0.625, 0.69, 'T-1 spikes', ha='center', fontsize=8)

# Vertical routing T2 -> Router -> Buffer -> T3
# T2 to router
ax.arrow(0.825, 0.5, 0, -0.05, head_width=0.02, head_length=0.02, fc='gray', ec='gray', lw=1.5)
ax.plot([0.825, 0.825], [0.45, 0.425], color='gray', lw=1.5)
ax.plot([0.825, 0.6], [0.425, 0.425], color='gray', lw=1.5)
ax.arrow(0.6, 0.425, -0.0, 0, head_width=0.02, head_length=0.02, fc='gray', ec='gray', lw=1.5)

# Router to buffer T3 
ax.plot([0.4, 0.125], [0.425, 0.425], color='gray', lw=1.5)
ax.arrow(0.125, 0.425, 0, -0.18, head_width=0.02, head_length=0.02, fc='gray', ec='gray', lw=1.5)

# T3 to Buffer
ax.arrow(0.3, 0.225, 0.1, 0, head_width=0.02, head_length=0.02, fc='black', ec='black', lw=1.5)
# Buffer to T4
ax.arrow(0.575, 0.225, 0.1, 0, head_width=0.02, head_length=0.02, fc='black', ec='black', lw=1.5)


ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.axis('off')

plt.tight_layout()
plt.savefig('d:/Courses/ECE274-NeuromorphicComputing/Project/Hardware_Architecture/multi_tile_noc_diagram.png', dpi=300, bbox_inches='tight')
