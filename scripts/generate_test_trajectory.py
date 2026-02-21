#!/usr/bin/env python3
"""Generate a synthetic arm trajectory for testing Blender rendering."""
import numpy as np

N = 90  # 6 seconds at 15 Hz
fps = 15.0
t = np.linspace(0, 2 * np.pi, N)

# 5 arm joints: smooth sinusoidal reaching motion
joint_angles = np.zeros((N, 5), dtype=np.float64)
joint_angles[:, 0] = 0.4 * np.sin(t)          # joint1: base yaw
joint_angles[:, 1] = -0.5 + 0.3 * np.sin(t)   # joint2: shoulder
joint_angles[:, 2] = 0.6 * np.sin(t * 0.5)    # joint3: elbow
joint_angles[:, 3] = 0.3 * np.sin(t * 1.5)    # joint4: wrist pitch
joint_angles[:, 4] = 0.2 * np.sin(t * 2.0)    # joint5: wrist roll

# Gripper: open → close → open
gripper_angles = np.where(
    (t > np.pi * 0.6) & (t < np.pi * 1.4), 0.4, 0.0
).astype(np.float64)

out_path = "renders/blender_vla/trajectory.npz"
np.savez(out_path, joint_angles=joint_angles, gripper_angles=gripper_angles, fps=fps)
print(f"Saved test trajectory: {out_path} ({N} frames, {fps} Hz)")
