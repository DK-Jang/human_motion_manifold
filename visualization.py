
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D, axes3d
import numpy as np
import torch
import sys
import h5py
import os
from skeleton_h36m import skeleton_H36M
import argparse

def render_animation(data, skeleton, fps, output='interactive', bitrate=1000):
    """
    Render or show an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    x = 0
    y = 1
    z = 2
    radius = torch.max(skeleton.offsets()).item() * 7 # Heuristic that works well with many skeletons
    
    skeleton_parents = skeleton.parents()

    plt.ioff()
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20., azim=30)

    ax.set_xlim3d([-radius/2, radius/2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius/2, radius/2])
    ax.set_aspect('auto')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    ax.dist = 7.5

    # Get rid of the ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    ax.w_zaxis.set_pane_color(white)

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)

    lines = []
    initialized = False
    
    trajectory = data[:, 0, [0, 2]]
    avg_segment_length = np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) + 1e-3
    # draw_offset = int(25/avg_segment_length)
    draw_offset = 0
    spline_line, = ax.plot(*trajectory.T)
    camera_pos = trajectory
    height_offset = np.min(data[:, :, 1]) # Min height
    data = data.copy()
    data[:, :, 1] -= height_offset
    
    def update(frame):
        nonlocal initialized
        ax.set_xlim3d([-radius/2 + camera_pos[frame, 0], radius/2 + camera_pos[frame, 0]])
        ax.set_ylim3d([-radius/2 + camera_pos[frame, 1], radius/2 + camera_pos[frame, 1]])

        positions_world = data[frame]
        for i in range(positions_world.shape[0]):
            if skeleton_parents[i] == -1:
                continue
            if not initialized:
                col = 'forestgreen' if i in skeleton.joints_right() else 'forestgreen'
                lines.append(ax.plot([positions_world[i, x], positions_world[skeleton_parents[i], x]],
                        [positions_world[i, y], positions_world[skeleton_parents[i], y]],
                        [positions_world[i, z], positions_world[skeleton_parents[i], z]], zdir='y', c=col, lw=4, alpha=1.0,
                        path_effects=[pe.withStroke(linewidth=5, foreground='darkgreen'), pe.Normal()]))
            else:
                lines[i-1][0].set_xdata([positions_world[i, x], positions_world[skeleton_parents[i], x]])
                lines[i-1][0].set_ydata([positions_world[i, y], positions_world[skeleton_parents[i], y]])
                lines[i-1][0].set_3d_properties([positions_world[i, z], positions_world[skeleton_parents[i], z]], zdir='y')
        l = max(frame-draw_offset, 0)
        r = min(frame+draw_offset, trajectory.shape[0])
        spline_line.set_xdata(trajectory[l:r, 0])
        spline_line.set_ydata(np.zeros_like(trajectory[l:r, 0]))
        spline_line.set_3d_properties(trajectory[l:r, 1], zdir='y')
        initialized = True

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=np.arange(0, data.shape[0]), interval=1000/fps, repeat=False)
    if output == 'interactive':
        plt.show()
        return anim
    elif output == 'html':
        return anim.to_html5_video()
    elif output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only html, .mp4, and .gif are supported)')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz_path', 
                        type=str, 
                        default='./pretrained/output/recon/m_recon.hdf5')
    args = parser.parse_args()
    
    for jj in range(60):    # # of test motions: 60
        with h5py.File(args.viz_path, 'r') as h5f:
            xyz = h5f['batch{0}'.format(jj + 1)][:]
            xyz = np.reshape(xyz, (xyz.shape[0], -1, 3))
            xyz = xyz[:-1]  # drop the last few frames
            render_animation(xyz, skeleton_H36M, 25)
