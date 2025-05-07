from pathlib import Path
import gymnasium as gym
import ale_py
import gymnasium.wrappers as wrappers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Get your observation

env = gym.make("ALE/Pacman-v5", frameskip=1)
env = wrappers.AtariPreprocessing(env)
env = wrappers.FrameStackObservation(env, stack_size=4)
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")  # (4, 84, 84)

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Downsample for better visualization
downsample = 1  # Use every 4th pixel
downsampled = obs[:, ::downsample, ::downsample]

# Create meshgrid for the pixel coordinates
y, x = np.mgrid[0:downsampled.shape[1], 0:downsampled.shape[2]]

# Plot each frame as a surface at different z-heights with spacing
spacing = 1000  # Space between layers
for i in range(4):
    # Normalize for better visibility
    frame = downsampled[i].astype(float)
    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-10)
    
    # Plot as a surface with the actual frame as height
    # Each frame is offset in z-axis by spacing
    surf = ax.plot_surface(x, y, frame + i*spacing, 
                          rstride=1, cstride=1,
                          cmap='gray',
                          alpha=0.9)
    
    # Add a border around each layer
    max_x, max_y = downsampled.shape[2]-1, downsampled.shape[1]-1
    z_level = i*spacing
    
    # Draw border lines around the perimeter of each layer
    # Bottom edge
    border_width = 5
    ax.plot([0, max_x], [0, 0], [z_level, z_level], 'k-', linewidth=border_width)
    # Top edge
    ax.plot([0, max_x], [max_y, max_y], [z_level, z_level], 'k-', linewidth=border_width)
    # Left edge
    ax.plot([0, 0], [0, max_y], [z_level, z_level], 'k-', linewidth=border_width)
    # Right edge
    ax.plot([max_x, max_x], [0, max_y], [z_level, z_level], 'k-', linewidth=border_width)

# Clean up the plot
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.axis('off')  # Hide all axes

# Make the figure background transparent for better contrast
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

ax.view_init(elev=250, azim=0, roll=-90)

# Save the figure
plt.tight_layout()
plt.savefig(Path("notes/_media") / "pacman_frame_stack_3d_layered.png")

plt.close()
