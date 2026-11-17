"""KITTI raw drives -> bags that follow the AP1.26 sensing contract.

kitti2mcap turns a KITTI raw drive into an MCAP with KITTI's own topic and frame
names. ``adapter`` rewrites that MCAP into a rosbag2 bag that looks like a
kart (or a filtered CARLA) recording: ICD topic names, AP1 frame names,
Velodyne-style XYZIRT clouds, latched /tf_static, and ground truth kept off
/tf on /ground_truth/*. Perception nodes then read KITTI, CARLA and kart bags
the same way. See the repo README.
"""
