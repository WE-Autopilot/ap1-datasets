"""End-to-end test of the adapter on a tiny synthetic kitti2mcap-style bag.

Needs ROS 2 (rosbag2_py with the MCAP plugin, vision_msgs, nav_msgs), which
the pixi environment provides; skipped without ROS.
"""
import numpy as np
import pytest
import yaml

rosbag2_py = pytest.importorskip('rosbag2_py')

from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2, PointField
from test_kitti_tracklets import pose_xml, tracklet_xml, write_tracklets
from tf2_msgs.msg import TFMessage

from ap1_datasets.kitti import adapter
from ap1_datasets.kitti import geometry as geo
from ap1_datasets.kitti.lidar import XYZIRT

T0 = 1_317_042_272_000_000_000   # 2011-09-26, like drive 0005
N_FRAMES = 3
LIDAR_IN_BASE = ([0.81, -0.31, 0.80], [0.0, 0.0, 0.0, 1.0])
CAMERA_OPTICAL_IN_BASE = ([1.08, -0.25, 0.73], list(geo.OPTICAL_IN_BODY))


def stamp(ns):
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def transform(parent, child, t_ns, xyz, quat):
    ts = TransformStamped()
    ts.header.stamp, ts.header.frame_id, ts.child_frame_id = stamp(t_ns), parent, child
    ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = xyz
    (ts.transform.rotation.x, ts.transform.rotation.y,
     ts.transform.rotation.z, ts.transform.rotation.w) = quat
    return ts


def kitti_cloud(t_ns):
    """64 lasers x 8 points, laser by laser, KITTI order, x/y/z/intensity float32."""
    rows = []
    for elev in np.radians(np.linspace(-24, 2, 64)):
        for az in np.radians(np.linspace(10, 350, 8)):
            rows.append([10 * np.cos(elev) * np.cos(az), 10 * np.cos(elev) * np.sin(az),
                         10 * np.sin(elev), 0.3])
    data = np.asarray(rows, dtype='<f4')
    msg = PointCloud2()
    msg.header.stamp, msg.header.frame_id = stamp(t_ns), 'velodyne'
    msg.height, msg.width = 1, len(data)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(['x', 'y', 'z', 'intensity'])]
    msg.point_step, msg.row_step = 16, 16 * len(data)
    msg.is_dense = True
    msg.data = data.tobytes()
    return msg


def write_input(path):
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    topics = {'/tf_static': 'tf2_msgs/msg/TFMessage', '/tf': 'tf2_msgs/msg/TFMessage',
              '/kitti/camera_color_left/image_raw': 'sensor_msgs/msg/Image',
              '/kitti/camera_color_left/camera_info': 'sensor_msgs/msg/CameraInfo',
              '/kitti/velodyne_points': 'sensor_msgs/msg/PointCloud2',
              '/kitti/imu': 'sensor_msgs/msg/Imu'}
    for name, type_name in topics.items():
        writer.create_topic(adapter.topic_metadata(name, type_name))

    def write(topic, msg, t):
        writer.write(topic, serialize_message(msg), t)

    write('/tf_static', TFMessage(transforms=[
        transform('base_link', 'velodyne', T0, *LIDAR_IN_BASE),
        transform('base_link', 'camera_color_left', T0, *CAMERA_OPTICAL_IN_BASE)]), T0)
    for k in range(N_FRAMES):
        t = T0 + k * 100_000_000
        img = Image(height=2, width=4, encoding='rgb8', step=12, data=bytes(24))
        img.header.stamp, img.header.frame_id = stamp(t), 'camera_color_left'
        write('/kitti/camera_color_left/image_raw', img, t)
        info = CameraInfo(height=2, width=4)
        info.header.stamp, info.header.frame_id = stamp(t), 'camera_color_left'
        write('/kitti/camera_color_left/camera_info', info, t)
        write('/kitti/velodyne_points', kitti_cloud(t), t)
        imu = Imu()
        imu.header.stamp, imu.header.frame_id = stamp(t), 'imu'
        write('/kitti/imu', imu, t)
        # driving +x at 10 m/s
        write('/tf', TFMessage(transforms=[
            transform('map', 'base_link', t, [k * 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])]), t)
    writer.close()
    return path


def read_bag(uri):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(uri), storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    meta = {t.topic_metadata.name: t.topic_metadata for t in
            reader.get_metadata().topics_with_message_count}
    msgs = {}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msgs.setdefault(topic, []).append(deserialize_message(data, get_message(meta[topic].type)))
    return meta, msgs


@pytest.fixture(scope='module')
def converted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('kitti')
    in_bag = write_input(tmp / 'drive_kitti')
    labels = write_tracklets(tmp / 'tracklet_labels.xml', [
        tracklet_xml('Pedestrian', 1.8, 0.6, 0.8, 0, [pose_xml(10, 2, -1.7, 0.0)] * N_FRAMES),
        tracklet_xml('Car', 1.5, 1.6, 4.0, 0, [pose_xml(20, -3, -1.6, 0)]),
    ])
    out = tmp / 'drive_ap1'
    assert adapter.main([str(next(in_bag.glob('*.mcap'))), str(out),
                         '--tracklets', str(labels)]) == 0
    return out


def test_topics_types_and_latched_tf_static(converted):
    meta, msgs = read_bag(converted)
    assert {name: m.type for name, m in meta.items()} == {
        '/tf_static': 'tf2_msgs/msg/TFMessage',
        '/sensing/camera/front/image_raw': 'sensor_msgs/msg/Image',
        '/sensing/camera/front/camera_info': 'sensor_msgs/msg/CameraInfo',
        '/sensing/lidar/pointcloud': 'sensor_msgs/msg/PointCloud2',
        '/sensing/imu/data': 'sensor_msgs/msg/Imu',
        '/ground_truth/tf': 'tf2_msgs/msg/TFMessage',
        '/ground_truth/odometry': 'nav_msgs/msg/Odometry',
        '/ground_truth/objects': 'vision_msgs/msg/Detection3DArray',
    }
    assert '/tf' not in msgs
    # ros2 bag play takes offered QoS from metadata.yaml: /tf_static must be latched
    topics = yaml.safe_load((converted / 'metadata.yaml').read_text())[
        'rosbag2_bagfile_information']['topics_with_message_count']
    qos = {t['topic_metadata']['name']: t['topic_metadata']['offered_qos_profiles']
           for t in topics}
    if isinstance(qos['/tf_static'], str):   # older rosbag2 stores it as a YAML string
        qos['/tf_static'] = yaml.safe_load(qos['/tf_static'])
    tf_static_qos, = qos['/tf_static']
    assert tf_static_qos['durability'] == 'transient_local'
    assert tf_static_qos['reliability'] == 'reliable'
    pairs = {(t.header.frame_id, t.child_frame_id) for t in msgs['/tf_static'][0].transforms}
    assert pairs == {('base_link', 'lidar_link'), ('base_link', 'imu_link'),
                     ('base_link', 'camera_link'),
                     ('camera_link', 'camera_optical_frame')}


def test_sensor_frames_and_xyzirt(converted):
    _, msgs = read_bag(converted)
    assert {m.header.frame_id for m in msgs['/sensing/camera/front/image_raw']} == {
        'camera_optical_frame'}
    assert {m.header.frame_id for m in msgs['/sensing/imu/data']} == {'imu_link'}
    cloud = msgs['/sensing/lidar/pointcloud'][0]
    assert cloud.header.frame_id == 'lidar_link'
    assert [f.name for f in cloud.fields] == list(XYZIRT.names)
    pts = np.frombuffer(bytes(cloud.data), dtype=XYZIRT)
    assert np.array_equal(np.unique(pts['ring']), np.arange(64))
    assert np.all((pts['time'] >= 0) & (pts['time'] < 0.1))


def test_ground_truth(converted):
    _, msgs = read_bag(converted)
    tf = msgs['/ground_truth/tf'][-1].transforms[0]
    assert (tf.header.frame_id, tf.child_frame_id) == ('kitti_map', 'base_link')
    odom = msgs['/ground_truth/odometry'][-1]
    assert odom.twist.twist.linear.x == pytest.approx(10.0)
    objects = msgs['/ground_truth/objects']
    assert len(objects) == N_FRAMES
    det, = objects[0].detections            # the car is not scored
    assert objects[0].header.frame_id == 'base_link'
    assert det.results[0].hypothesis.class_id == 'pedestrian'
    p = det.bbox.center.position
    assert (p.x, p.y, p.z) == pytest.approx((10.81, 1.69, -1.7 + 0.9 + 0.80))
    assert any(det.results[0].pose.covariance)


def test_provenance(converted):
    info = yaml.safe_load((converted / 'provenance.yaml').read_text())
    assert info['ring_methods'] == {'scan_order': N_FRAMES}
    assert info['message_counts']['/sensing/lidar/pointcloud'] == N_FRAMES


def test_refuses_existing_output(converted, tmp_path):
    with pytest.raises(SystemExit):
        adapter.main([str(tmp_path / 'missing.mcap'), str(converted)])
