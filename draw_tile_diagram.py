import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 6))

# Main Tile Box
box = patches.Rectangle((0.15, 0.1), 0.7, 0.8, fill=True, color='#f0f4c3', ec='#2c3e50', lw=2)
ax.add_patch(box)
ax.text(0.5, 0.85, 'SNN Hardware Tile (snn_tile.v)', fontsize=14, fontweight='bold', ha='center', color='#2c3e50', family='monospace')

# Input / Output Interfaces
ax.arrow(0.02, 0.7, 0.11, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
ax.text(0.08, 0.73, 'rx_spike\n(NoC In)', ha='center', va='bottom', fontsize=9, family='monospace')

ax.arrow(0.85, 0.3, 0.1, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
ax.text(0.92, 0.33, 'tx_spike\n(NoC Out)', ha='center', va='bottom', fontsize=9, family='monospace')

# Components
sram = patches.Rectangle((0.2, 0.6), 0.2, 0.15, fill=True, color='#ffcc80', ec='black', lw=1.5)
ax.add_patch(sram)
ax.text(0.3, 0.675, 'Local SRAM\nWeight Bank', ha='center', va='center', fontsize=9, fontweight='bold')

mac = patches.Rectangle((0.55, 0.6), 0.2, 0.15, fill=True, color='#ce93d8', ec='black', lw=1.5)
ax.add_patch(mac)
ax.text(0.65, 0.675, 'Sparse MAC\nAdder Tree', ha='center', va='center', fontsize=9, fontweight='bold')

lif = patches.Rectangle((0.55, 0.2), 0.2, 0.2, fill=True, color='#90caf9', ec='black', lw=1.5)
ax.add_patch(lif)
ax.text(0.65, 0.3, 'LIF Neuron\nArray (256)', ha='center', va='center', fontsize=9, fontweight='bold')

router = patches.Rectangle((0.75, 0.25), 0.1, 0.1, fill=True, color='#a5d6a7', ec='black', lw=1.5)
ax.add_patch(router)
ax.text(0.8, 0.3, 'Spike\nEncoder', ha='center', va='center', fontsize=8, fontweight='bold')

global_ctrl = patches.Rectangle((0.2, 0.2), 0.2, 0.1, fill=True, color='#e0e0e0', ec='black', lw=1.5)
ax.add_patch(global_ctrl)
ax.text(0.3, 0.25, 'Global Control\n(timestep_tick)', ha='center', va='center', fontsize=9, fontweight='bold')

# Internal Wiring
ax.arrow(0.4, 0.675, 0.13, 0, head_width=0.015, head_length=0.02, fc='black', ec='black') # SRAM to MAC
ax.text(0.47, 0.69, '1024-bit\nWeight Bus', ha='center', va='bottom', fontsize=8)

ax.arrow(0.65, 0.6, 0, -0.18, head_width=0.015, head_length=0.02, fc='black', ec='black') # MAC to LIF
ax.text(0.67, 0.5, 'V_updates\n(INT16)', ha='left', va='center', fontsize=8)

ax.arrow(0.75, 0.3, 0.0, 0, head_width=0.01, head_length=0.01, fc='black', ec='black') # LIF to Router (stub to indicate connection)
ax.text(0.75, 0.22, '256-bit\nFire Vector', ha='center', va='top', fontsize=8)

# Control Wires
ax.plot([0.3, 0.3], [0.3, 0.45], color='gray', ls='--')
ax.arrow(0.3, 0.45, 0.25, 0, head_width=0.015, head_length=0.02, fc='gray', ec='gray', ls='--') # Control to LIF

ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.axis('off')

plt.tight_layout()
plt.savefig('d:/Courses/ECE274-NeuromorphicComputing/Project/Hardware_Architecture/snn_tile_diagram.png', dpi=300, bbox_inches='tight')
