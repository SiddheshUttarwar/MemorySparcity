import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 6))

# Main Module Box
box = patches.Rectangle((0.2, 0.1), 0.6, 0.8, fill=True, color='#f8f9fa', ec='#2c3e50', lw=2)
ax.add_patch(box)
ax.text(0.5, 0.85, 'ping_pong_spike_buffer', fontsize=16, fontweight='bold', ha='center', color='#2c3e50', family='monospace')
ax.text(0.5, 0.80, 'ADDR_WIDTH=11, BUFFER_DEPTH=256', fontsize=10, ha='center', color='gray')

# Internal Memories
mem_A = patches.Rectangle((0.45, 0.5), 0.2, 0.2, fill=True, color='#e1bee7', ec='black', lw=1.5)
ax.add_patch(mem_A)
ax.text(0.55, 0.6, 'Memory Bank A\nreg [10:0] mem_A [0:255]', ha='center', va='center', fontsize=9, family='monospace')

mem_B = patches.Rectangle((0.45, 0.2), 0.2, 0.2, fill=True, color='#c5cae9', ec='black', lw=1.5)
ax.add_patch(mem_B)
ax.text(0.55, 0.3, 'Memory Bank B\nreg [10:0] mem_B [0:255]', ha='center', va='center', fontsize=9, family='monospace')

# Control FSM
fsm = patches.Rectangle((0.25, 0.45), 0.1, 0.15, fill=True, color='#fff9c4', ec='black', lw=1.5)
ax.add_patch(fsm)
ax.text(0.3, 0.525, 'Ping-Pong\nState\nToggle', ha='center', va='center', fontsize=9, fontweight='bold')

# MUX
mux_read = patches.Polygon([[0.7, 0.5], [0.75, 0.45], [0.75, 0.35], [0.7, 0.3]], fill=True, color='#ffe0b2', ec='black')
ax.add_patch(mux_read)
ax.text(0.725, 0.4, 'MUX', ha='center', va='center', fontsize=8, fontweight='bold', rotation=270)

# Input Arrows
inputs = [
    ('clk', 0.85), ('rst_n', 0.78), ('timestep_tick', 0.525), 
    ('write_en', 0.35), ('write_spike_id[10:0]', 0.28),
    ('read_en', 0.15)
]
for text, y in inputs:
    ax.arrow(0.05, y, 0.13, 0, head_width=0.015, head_length=0.02, fc='black', ec='black')
    ax.text(0.04, y, text, ha='right', va='center', fontsize=11, family='monospace')

# Output Arrows
outputs = [
    ('buffer_full', 0.78), ('buffer_empty', 0.72), ('read_spike_id[10:0]', 0.4)
]
for text, y in outputs:
    ax.arrow(0.8, y, 0.13, 0, head_width=0.015, head_length=0.02, fc='black', ec='black')
    ax.text(0.95, y, text, ha='left', va='center', fontsize=11, family='monospace')

# Internal wiring
ax.arrow(0.35, 0.525, 0.08, 0, head_width=0.01, head_length=0.015, fc='gray', ec='gray', ls='--') # FSM to A
ax.arrow(0.35, 0.525, 0.03, 0, ec='gray', ls='--')
ax.plot([0.38, 0.38], [0.525, 0.3], color='gray', ls='--')
ax.arrow(0.38, 0.3, 0.05, 0, head_width=0.01, head_length=0.015, fc='gray', ec='gray', ls='--') # FSM to B

# To MUX
ax.plot([0.38, 0.38], [0.525, 0.7], color='gray', ls='--')
ax.plot([0.38, 0.725], [0.7, 0.7], color='gray', ls='--')
ax.arrow(0.725, 0.7, 0, -0.19, head_width=0.01, head_length=0.015, fc='gray', ec='gray', ls='--') # FSM to MUX sel

# Mux inputs
ax.arrow(0.65, 0.6, 0.035, 0, head_width=0.01, head_length=0.015, fc='black', ec='black')
ax.plot([0.65, 0.65], [0.6, 0.45], color='black')
ax.arrow(0.65, 0.3, 0.035, 0, head_width=0.01, head_length=0.015, fc='black', ec='black')

ax.arrow(0.75, 0.4, 0.03, 0, head_width=0.01, head_length=0.015, fc='black', ec='black') # Mux Out

ax.set_xlim(0, 1.2)
ax.set_ylim(0, 0.95)
ax.axis('off')

plt.tight_layout()
plt.savefig('d:/Courses/ECE274-NeuromorphicComputing/Project/Hardware_Architecture/ping_pong_diagram_rtl.png', dpi=300, bbox_inches='tight')
