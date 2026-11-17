"""Unit tests for KITTI tracklet parsing and ground-truth boxes. No ROS needed."""
import numpy as np
import pytest

from ap1_datasets.kitti import geometry as geo
from ap1_datasets.kitti.adapter import ground_truth_boxes, sensor_tree
from ap1_datasets.kitti.tracklets import TRUNC_OUT_IMAGE, parse_tracklets

CLASSES = {'Pedestrian': 'pedestrian', 'Cyclist': 'unknown'}


def pose_xml(tx, ty, tz, rz, truncation=0):
    return (f'<item><tx>{tx}</tx><ty>{ty}</ty><tz>{tz}</tz><rx>0</rx><ry>0</ry><rz>{rz}</rz>'
            f'<state>1</state><occlusion>0</occlusion><occlusion_kf>0</occlusion_kf>'
            f'<truncation>{truncation}</truncation><amt_occlusion>0</amt_occlusion>'
            f'<amt_occlusion_kf>0</amt_occlusion_kf><amt_border_l>0</amt_border_l>'
            f'<amt_border_r>0</amt_border_r><amt_border_kf>0</amt_border_kf></item>')


def tracklet_xml(cls, h, w, length, first, poses):
    return (f'<item><objectType>{cls}</objectType><h>{h}</h><w>{w}</w><l>{length}</l>'
            f'<first_frame>{first}</first_frame><poses class_id="2" tracking_level="0" '
            f'version="0"><count>{len(poses)}</count><item_version>2</item_version>'
            + ''.join(poses) + '</poses><finished>1</finished></item>')


def write_tracklets(path, items):
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n'
        '<!DOCTYPE boost_serialization>\n<boost_serialization signature="serialization::archive" '
        'version="9"><tracklets class_id="0" tracking_level="0" version="0">'
        f'<count>{len(items)}</count><item_version>1</item_version>'
        + ''.join(items) + '</tracklets></boost_serialization>')
    return path


@pytest.fixture
def labels(tmp_path):
    return parse_tracklets(write_tracklets(tmp_path / 'tracklet_labels.xml', [
        tracklet_xml('Pedestrian', 1.8, 0.6, 0.8, 1, [
            pose_xml(10, 2, -1.7, 0.5), pose_xml(10.5, 2, -1.7, 0.5)]),
        tracklet_xml('Car', 1.5, 1.6, 4.0, 0, [pose_xml(20, -3, -1.6, 0)]),
        tracklet_xml('Cyclist', 1.7, 0.6, 1.8, 2, [pose_xml(8, 6, -1.7, 0, TRUNC_OUT_IMAGE)]),
    ]))


def test_parse_frames_and_box_centre(labels):
    assert sorted(labels) == [0, 1, 2]
    ped = labels[1][0]
    assert ped.kitti_class == 'Pedestrian' and ped.track_id == 0
    assert np.allclose(ped.center, [10, 2, -1.7 + 0.9])   # tz is the box bottom
    assert np.allclose(ped.size, [0.8, 0.6, 1.8])         # l, w, h
    assert labels[2][0].center[0] == 10.5 and labels[2][1].kitti_class == 'Cyclist'


def test_ground_truth_filters_classes_and_visibility(labels):
    t_base_lidar = geo.make_transform([0.8, -0.3, 0.8], [0, 0, 0, 1])
    assert ground_truth_boxes(labels[0], t_base_lidar, CLASSES, True) == []   # Car not scored
    frame2 = ground_truth_boxes(labels[2], t_base_lidar, CLASSES, True)
    assert [(tid, cls) for tid, cls, _, _ in frame2] == [(0, 'pedestrian')]   # cyclist outside
    with_hidden = ground_truth_boxes(labels[2], t_base_lidar, CLASSES, False)
    assert [cls for _, cls, _, _ in with_hidden] == ['pedestrian', 'unknown']


def test_ground_truth_box_in_base_link(labels):
    t_base_lidar = geo.make_transform([0.8, -0.3, 0.8], geo.yaw_quat(np.pi / 2))
    (_, _, t_box, size), = ground_truth_boxes(labels[1], t_base_lidar, CLASSES, True)
    # lidar point (10, 2, -0.8) rotated +90 deg about z, then shifted
    assert np.allclose(t_box[:3, 3], [0.8 - 2, -0.3 + 10, 0.8 - 0.8])
    yaw = np.arctan2(t_box[1, 0], t_box[0, 0])
    assert yaw == pytest.approx(0.5 + np.pi / 2)
    assert np.allclose(size, [0.8, 0.6, 1.8])


def test_sensor_tree_composes_back_to_kitti_calibration():
    frames = {'base': 'base_link', 'lidar': 'lidar_link', 'imu': 'imu_link',
              'camera': 'camera_link', 'camera_optical': 'camera_optical_frame'}
    t_lidar = geo.make_transform([0.81, -0.31, 0.80], geo.quat_from_rpy(0.01, 0, 0.002))
    t_cam_opt = geo.make_transform([1.08, -0.25, 0.73], [-0.4992, 0.4966, -0.5029, 0.5013])
    tree = {(p, c): t for p, c, t in sensor_tree(t_lidar, t_cam_opt, frames)}
    assert np.allclose(tree[('base_link', 'lidar_link')], t_lidar)
    assert np.allclose(tree[('base_link', 'imu_link')], np.eye(4))
    composed = (tree[('base_link', 'camera_link')]
                @ tree[('camera_link', 'camera_optical_frame')])
    assert np.allclose(composed, t_cam_opt, atol=1e-9)
    # KITTI's camera looks forward, so the camera body x axis is close to base x
    assert tree[('base_link', 'camera_link')][0, 0] > 0.99
