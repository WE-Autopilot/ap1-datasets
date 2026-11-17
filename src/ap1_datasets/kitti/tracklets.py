"""KITTI raw tracklet labels (tracklet_labels.xml) -> per-frame 3D boxes.

Format from the KITTI raw devkit (parseTrackletXML.py): one <item> per object
with objectType, box size h/w/l, first_frame and one pose per frame. A pose is
the box's bottom centre (tx, ty, tz) and yaw rz, in the Velodyne frame.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

# devkit truncation states
TRUNC_IN_IMAGE = 0
TRUNC_TRUNCATED = 1
TRUNC_OUT_IMAGE = 2
TRUNC_BEHIND_IMAGE = 99


@dataclass
class Box:
    """One labelled object in one frame, in the Velodyne frame."""

    track_id: int
    kitti_class: str
    center: np.ndarray          # box centre (x, y, z), metres
    yaw: float                  # about z, radians
    size: np.ndarray            # (l, w, h): x, y, z extent before yaw, metres
    truncation: int
    occlusion: int


def parse_tracklets(path):
    """Boxes per frame index: {frame: [Box, ...]}."""
    root = ET.parse(path).getroot()
    tracklets = root.find('tracklets')
    if tracklets is None:
        raise ValueError(f'{path}: no <tracklets> element')
    frames = {}
    for track_id, item in enumerate(tracklets.findall('item')):
        kitti_class = item.findtext('objectType')
        h, w, length = (float(item.findtext(k)) for k in ('h', 'w', 'l'))
        first = int(item.findtext('first_frame'))
        for i, pose in enumerate(item.find('poses').findall('item')):
            tx, ty, tz, rz = (float(pose.findtext(k)) for k in ('tx', 'ty', 'tz', 'rz'))
            frames.setdefault(first + i, []).append(Box(
                track_id=track_id,
                kitti_class=kitti_class,
                center=np.array([tx, ty, tz + h / 2]),  # tz is the bottom of the box
                yaw=rz,
                size=np.array([length, w, h]),
                truncation=int(pose.findtext('truncation')),
                occlusion=int(pose.findtext('occlusion')),
            ))
    return frames


def visible_in_camera(box):
    """True if the box is at least partly inside the labelled camera image."""
    return box.truncation in (TRUNC_IN_IMAGE, TRUNC_TRUNCATED)


def map_class(kitti_class, class_map):
    """AP1 class string for a KITTI class, or None if it isn't scored."""
    return class_map.get(kitti_class)
