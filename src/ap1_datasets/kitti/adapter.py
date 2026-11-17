"""Rewrite a kitti2mcap MCAP into an AP1.26 sensing-contract rosbag2 bag.

    python -m ap1_datasets.kitti.adapter <drive>_kitti.mcap <out_dir> \\
        [--tracklets tracklet_labels.xml] [--config config.yaml]

What changes (names come from config.yaml):
  - camera, lidar and IMU topics and frame ids renamed to the ICD ones
  - the cloud becomes Velodyne-style XYZIRT with reconstructed ring and time
    (see lidar.py)
  - /tf_static rebuilt as base_link -> lidar, IMU, camera body -> camera
    optical frame, written latched (transient_local) so tf2 listeners and RViz
    receive it on playback; kitti2mcap's MCAP has no QoS metadata
  - KITTI's OXTS pose (map -> base_link on /tf) moves to /ground_truth/tf and
    /ground_truth/odometry, so /tf stays free for localization (ICD #12)
  - with --tracklets, the labels become /ground_truth/objects
    (vision_msgs/Detection3DArray in base_link), one array per camera frame
Other topics are copied through unchanged.

Needs ROS 2 Jazzy (rosbag2_py with the MCAP plugin): the pixi environment.
ROS modules are imported inside the functions that use them so the pure
helpers test without ROS.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from ap1_datasets.kitti import geometry as geo
from ap1_datasets.kitti.lidar import XYZIRT, to_xyzirt, xyzi_from_buffer
from ap1_datasets.kitti.tracklets import map_class, parse_tracklets, visible_in_camera

CONFIG_PATH = Path(__file__).with_name('config.yaml')
TOOL_VERSION = 1   # bump when the output changes; recorded in provenance.yaml
STREAMS = ('image', 'camera_info', 'lidar', 'imu')

# Ground-truth box covariance: tracklets are hand labels, so small but never
# all zero (consumers read zero as perfect certainty; Object Detection Message).
GT_VARIANCE = 1e-4


# ---------- config ----------

def load_config(path):
    """Load and check the config; missing keys are errors, not silent defaults."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    required = {
        'input.topics': ['image', 'camera_info', 'lidar', 'imu', 'tf', 'tf_static'],
        'input.frames': ['base', 'lidar', 'camera_optical', 'map'],
        'output.topics': ['image', 'camera_info', 'lidar', 'imu', 'ground_truth_tf',
                          'ground_truth_odometry', 'ground_truth_objects'],
        'output.frames': ['base', 'lidar', 'imu', 'camera', 'camera_optical', 'map'],
        'lidar': ['sweep_period_s', 'counter_clockwise'],
        'ground_truth': ['only_visible_in_camera', 'classes'],
    }
    for section, keys in required.items():
        node = cfg
        for part in section.split('.'):
            node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            raise SystemExit(f'{path}: missing section {section}')
        missing = [k for k in keys if k not in node]
        if missing:
            raise SystemExit(f'{path}: {section} is missing {", ".join(missing)}')
    return cfg


# ---------- pure helpers ----------

def sensor_tree(t_base_lidar, t_base_cam_optical, frames):
    """Static transforms of the output bag: [(parent, child, 4x4 transform)].

    base -> lidar and base -> camera body come from KITTI's calibration; the
    IMU is base_link itself in kitti2mcap, so base -> imu is identity.
    """
    t_base_cam = geo.body_from_optical(t_base_cam_optical)
    t_cam_optical = geo.make_transform(np.zeros(3), geo.OPTICAL_IN_BODY)
    return [
        (frames['base'], frames['lidar'], t_base_lidar),
        (frames['base'], frames['imu'], np.eye(4)),
        (frames['base'], frames['camera'], t_base_cam),
        (frames['camera'], frames['camera_optical'], t_cam_optical),
    ]


def ground_truth_boxes(boxes, t_base_lidar, class_map, only_visible):
    """Tracklet boxes -> [(track_id, ap1_class, 4x4 base -> box, size)] in base_link."""
    out = []
    for box in boxes:
        if only_visible and not visible_in_camera(box):
            continue
        ap1_class = map_class(box.kitti_class, class_map)
        if ap1_class is None:
            continue
        t_box = t_base_lidar @ geo.make_transform(box.center, geo.yaw_quat(box.yaw))
        out.append((box.track_id, ap1_class, t_box, box.size))
    return out


# ---------- ROS message helpers ----------

def stamp_to_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def transform_to_matrix(tf):
    t, r = tf.translation, tf.rotation
    return geo.make_transform([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])


def fill_transform(msg_tf, t):
    """Write a 4x4 transform into a geometry_msgs/Transform."""
    trans, quat = geo.split_transform(t)
    msg_tf.translation.x, msg_tf.translation.y, msg_tf.translation.z = map(float, trans)
    (msg_tf.rotation.x, msg_tf.rotation.y,
     msg_tf.rotation.z, msg_tf.rotation.w) = map(float, quat)


def fill_pose(pose, t):
    """Write a 4x4 transform into a geometry_msgs/Pose."""
    trans, quat = geo.split_transform(t)
    pose.position.x, pose.position.y, pose.position.z = map(float, trans)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(
        float, quat)


def tf_static_message(stamp, tree):
    from geometry_msgs.msg import TransformStamped
    from tf2_msgs.msg import TFMessage
    msg = TFMessage()
    for parent, child, t in tree:
        ts = TransformStamped()
        ts.header.stamp = stamp
        ts.header.frame_id = parent
        ts.child_frame_id = child
        fill_transform(ts.transform, t)
        msg.transforms.append(ts)
    return msg


def xyzirt_cloud(header, frame_id, pts):
    from sensor_msgs.msg import PointCloud2, PointField
    msg = PointCloud2()
    msg.header.stamp = header.stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = 1, len(pts)
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='ring', offset=16, datatype=PointField.UINT16, count=1),
        PointField(name='time', offset=20, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = XYZIRT.itemsize
    msg.row_step = XYZIRT.itemsize * len(pts)
    msg.is_dense = True
    msg.data = pts.tobytes()
    return msg


def convert_cloud(msg, frame_id, lidar_cfg):
    """KITTI x, y, z, intensity cloud -> (XYZIRT cloud in frame_id, ring method)."""
    fields = {f.name: f.offset for f in msg.fields}
    missing = {'x', 'y', 'z', 'intensity'} - fields.keys()
    if missing:
        raise SystemExit(f'lidar cloud has no {", ".join(sorted(missing))} field')
    xyzi = xyzi_from_buffer(msg.data, fields, msg.point_step, msg.width * msg.height)
    pts, method = to_xyzirt(xyzi, lidar_cfg['sweep_period_s'], lidar_cfg['counter_clockwise'])
    return xyzirt_cloud(msg.header, frame_id, pts), method


def detections_message(header, base_frame, gt_boxes):
    from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
    arr = Detection3DArray()
    arr.header.stamp = header.stamp
    arr.header.frame_id = base_frame
    cov = np.zeros(36)
    cov[[0, 7, 14, 21, 28, 35]] = GT_VARIANCE
    for track_id, ap1_class, t_box, size in gt_boxes:
        det = Detection3D()
        det.header = arr.header
        det.id = str(track_id)
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = ap1_class
        hyp.hypothesis.score = 1.0
        fill_pose(hyp.pose.pose, t_box)
        hyp.pose.covariance = cov.tolist()
        det.results.append(hyp)
        fill_pose(det.bbox.center, t_box)
        det.bbox.size.x, det.bbox.size.y, det.bbox.size.z = map(float, size)
        arr.detections.append(det)
    return arr


def odometry_message(stamp, map_frame, base_frame, t_map_base, twist):
    from nav_msgs.msg import Odometry
    odom = Odometry()
    odom.header.stamp = stamp
    odom.header.frame_id = map_frame
    odom.child_frame_id = base_frame
    fill_pose(odom.pose.pose, t_map_base)
    v, w = twist
    odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = map(float, v)
    odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = map(
        float, w)
    return odom


# ---------- rosbag2 I/O ----------

def topic_metadata(name, type_name, latched=False):
    import rosbag2_py
    kwargs = {}
    if latched:
        qos_cls = getattr(rosbag2_py, 'QoS', None) or rosbag2_py._storage.QoS  # Jazzy: _storage
        kwargs['offered_qos_profiles'] = [qos_cls(1).reliable().transient_local()]
    try:
        return rosbag2_py.TopicMetadata(id=0, name=name, type=type_name,
                                        serialization_format='cdr', **kwargs)
    except TypeError:   # rosbag2 before Jazzy has no id
        return rosbag2_py.TopicMetadata(name=name, type=type_name,
                                        serialization_format='cdr', **kwargs)


def open_reader(uri, topics=None):
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(uri), storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    if topics:
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    return reader


def read_static(uri, cfg):
    """(bag start time, base -> lidar, base -> camera optical) from the input bag."""
    from rclpy.serialization import deserialize_message
    from tf2_msgs.msg import TFMessage
    tin, fin = cfg['input']['topics'], cfg['input']['frames']
    reader = open_reader(uri, [tin['tf_static']])
    metadata = reader.get_metadata()
    if metadata.message_count == 0:
        raise SystemExit(f'{uri} has no messages')
    start_ns = metadata.starting_time.nanoseconds
    static = {}
    while reader.has_next():
        _, data, _ = reader.read_next()
        for ts in deserialize_message(data, TFMessage).transforms:
            static[(ts.header.frame_id, ts.child_frame_id)] = transform_to_matrix(ts.transform)
    t_lidar = static.get((fin['base'], fin['lidar']))
    t_cam = static.get((fin['base'], fin['camera_optical']))
    if t_lidar is None or t_cam is None:
        raise SystemExit(f'{uri}: {tin["tf_static"]} needs {fin["base"]} -> {fin["lidar"]} and '
                         f'{fin["base"]} -> {fin["camera_optical"]}; found {sorted(static)}')
    return start_ns, t_lidar, t_cam


def convert(in_uri, out_uri, cfg, tracklets_path=None):
    """Write the AP1 bag; returns (per-output-topic message counts, ring method counts)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from rosidl_runtime_py.utilities import get_message
    from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2
    from tf2_msgs.msg import TFMessage

    tin, fin = cfg['input']['topics'], cfg['input']['frames']
    tout, fout = cfg['output']['topics'], cfg['output']['frames']
    gt_cfg = cfg['ground_truth']

    first_t, t_base_lidar, t_base_cam = read_static(in_uri, cfg)
    tree = sensor_tree(t_base_lidar, t_base_cam, fout)
    labels = parse_tracklets(tracklets_path) if tracklets_path else None

    reader = open_reader(in_uri)
    in_types = {m.name: m.type for m in reader.get_all_topics_and_types()}
    for key in STREAMS:
        if tin[key] not in in_types:
            raise SystemExit(f'{in_uri} has no {key} topic {tin[key]}')

    writer = rosbag2_py.SequentialWriter()
    # zstd_fast: chunk-compressed MCAP, about the size of kitti2mcap's input
    writer.open(rosbag2_py.StorageOptions(uri=str(out_uri), storage_id='mcap',
                                          storage_preset_profile='zstd_fast'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    out_types = {
        tout['image']: 'sensor_msgs/msg/Image',
        tout['camera_info']: 'sensor_msgs/msg/CameraInfo',
        tout['lidar']: 'sensor_msgs/msg/PointCloud2',
        tout['imu']: 'sensor_msgs/msg/Imu',
        tout['ground_truth_tf']: 'tf2_msgs/msg/TFMessage',
        tout['ground_truth_odometry']: 'nav_msgs/msg/Odometry',
    }
    if labels is not None:
        out_types[tout['ground_truth_objects']] = 'vision_msgs/msg/Detection3DArray'
    writer.create_topic(topic_metadata('/tf_static', 'tf2_msgs/msg/TFMessage', latched=True))
    for name, type_name in out_types.items():
        writer.create_topic(topic_metadata(name, type_name))
    handled = {tin[k] for k in STREAMS} | {tin['tf'], tin['tf_static']}
    passthrough = {n: ty for n, ty in in_types.items() if n not in handled}
    for name, type_name in passthrough.items():
        writer.create_topic(topic_metadata(name, type_name))

    counts, ring_methods = {}, {}

    def write(topic, msg, t):
        writer.write(topic, serialize_message(msg), t)
        counts[topic] = counts.get(topic, 0) + 1

    from builtin_interfaces.msg import Time
    write('/tf_static', tf_static_message(Time(sec=first_t // 1_000_000_000,
                                               nanosec=first_t % 1_000_000_000), tree), first_t)

    frame_index = 0
    prev_pose = None   # (stamp_ns, map -> base) for the odometry twist
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == tin['image']:
            msg = deserialize_message(data, Image)
            msg.header.frame_id = fout['camera_optical']
            write(tout['image'], msg, t)
            if labels is not None:
                boxes = ground_truth_boxes(labels.get(frame_index, []), t_base_lidar,
                                           gt_cfg['classes'], gt_cfg['only_visible_in_camera'])
                write(tout['ground_truth_objects'],
                      detections_message(msg.header, fout['base'], boxes), t)
            frame_index += 1
        elif topic == tin['camera_info']:
            msg = deserialize_message(data, CameraInfo)
            msg.header.frame_id = fout['camera_optical']
            write(tout['camera_info'], msg, t)
        elif topic == tin['lidar']:
            cloud, method = convert_cloud(deserialize_message(data, PointCloud2),
                                          fout['lidar'], cfg['lidar'])
            ring_methods[method] = ring_methods.get(method, 0) + 1
            write(tout['lidar'], cloud, t)
        elif topic == tin['imu']:
            msg = deserialize_message(data, Imu)
            msg.header.frame_id = fout['imu']
            write(tout['imu'], msg, t)
        elif topic == tin['tf']:
            msg = deserialize_message(data, TFMessage)
            for ts in msg.transforms:
                if (ts.header.frame_id, ts.child_frame_id) != (fin['map'], fin['base']):
                    continue
                ts.header.frame_id = fout['map']
                ts.child_frame_id = fout['base']
                t_map_base = transform_to_matrix(ts.transform)
                stamp_ns = stamp_to_ns(ts.header.stamp)
                if prev_pose is not None and stamp_ns > prev_pose[0]:
                    twist = geo.body_twist(prev_pose[1], t_map_base,
                                           (stamp_ns - prev_pose[0]) / 1e9)
                else:
                    twist = (np.zeros(3), np.zeros(3))
                prev_pose = (stamp_ns, t_map_base)
                write(tout['ground_truth_tf'], TFMessage(transforms=[ts]), t)
                write(tout['ground_truth_odometry'],
                      odometry_message(ts.header.stamp, fout['map'], fout['base'],
                                       t_map_base, twist), t)
        elif topic == tin['tf_static']:
            continue   # rebuilt above
        else:
            msg = deserialize_message(data, get_message(passthrough[topic]))
            write(topic, msg, t)
    writer.close()   # writes metadata.yaml
    return counts, ring_methods


def source_commit():
    """Commit of this checkout, '+dirty' if it has uncommitted changes; None outside git."""
    here = Path(__file__).parent
    try:
        sha = subprocess.run(['git', '-C', str(here), 'rev-parse', 'HEAD'], check=True,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(here), 'status', '--porcelain'], check=True,
                               capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ('+dirty' if dirty else '')


def write_provenance(out_uri, in_uri, tracklets_path, cfg, counts, ring_methods):
    """provenance.yaml next to metadata.yaml: where the bag came from and how it differs."""
    info = {
        'tool': 'https://github.com/WE-Autopilot/ap1-datasets (ap1_datasets.kitti.adapter)',
        'tool_version': TOOL_VERSION,
        'tool_commit': source_commit(),
        'input': str(Path(in_uri).name),
        'tracklets': str(Path(tracklets_path).name) if tracklets_path else None,
        'license': 'KITTI raw data: CC BY-NC-SA 3.0 (attribution, non-commercial, share-alike)',
        'deviations_from_icd': [
            'camera 1242x375, rectified (ICD #6: 1920x1080)',
            'lidar ring and time are reconstructed (see ring_methods): not for LIO-SAM',
            'IMU at 10 Hz (synced OXTS; ICD: 100 Hz)',
            'base_link is the OXTS/IMU unit, not the AP1 vehicle origin',
        ],
        'frames': cfg['output']['frames'],
        'message_counts': counts,
        # sweeps whose rings came from KITTI's laser order vs nominal elevations
        'ring_methods': ring_methods,
    }
    with open(Path(out_uri) / 'provenance.yaml', 'w') as f:
        yaml.safe_dump(info, f, sort_keys=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('input', help='kitti2mcap output (.mcap)')
    p.add_argument('output', help='output rosbag2 directory (must not exist)')
    p.add_argument('--tracklets', help="the drive's tracklet_labels.xml -> /ground_truth/objects")
    p.add_argument('--config', type=Path, default=CONFIG_PATH, help='default: %(default)s')
    args = p.parse_args(argv)

    if Path(args.output).exists():
        raise SystemExit(f'{args.output} already exists')
    if not Path(args.input).is_file():
        raise SystemExit(f'{args.input} not found')
    cfg = load_config(args.config)
    counts, ring_methods = convert(args.input, args.output, cfg, args.tracklets)
    write_provenance(args.output, args.input, args.tracklets, cfg, counts, ring_methods)
    width = max(len(k) for k in counts)
    for topic, n in sorted(counts.items()):
        print(f'  {topic:{width}}  {n}')
    print(f'  lidar rings: {ring_methods}')
    print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
