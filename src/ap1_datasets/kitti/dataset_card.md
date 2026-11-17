---
license: cc-by-nc-sa-3.0
tags:
- ros2
- autonomy
- robotics
pretty_name: KITTI raw drives as AP1 ROS 2 bags (MCAP)
---

# KITTI raw drives as AP1 ROS 2 bags

Drives from the [KITTI raw dataset](https://www.cvlibs.net/datasets/kitti/raw_data.php),
converted to ROS 2 Jazzy rosbag2 bags (MCAP storage) that follow the
WE-Autopilot AP1.26 sensing contract: AP1 topic and frame names, latched
`/tf_static`, and ground truth kept off `/tf`. AP1 perception reads them the
same way it reads CARLA and kart recordings.

Built by [WE-Autopilot/ap1-datasets](https://github.com/WE-Autopilot/ap1-datasets);
every bag's `provenance.yaml` records the source drive and the tool commit.

## Layout

One rosbag2 directory per drive:

```
<date>_drive_<NNNN>/
  <date>_drive_<NNNN>_0.mcap
  metadata.yaml
  provenance.yaml
```

| Drive | Duration | Labelled classes |
|---|---|---|
| `2011_09_26_drive_0005` | 15.8 s | Pedestrian, Cyclist, Car, Van |
| `2011_09_26_drive_0060` | 8.0 s | Pedestrian, Cyclist, Car |

## Topics

| Topic | Type | frame_id | Notes |
|---|---|---|---|
| `/sensing/camera/front/image_raw` | `sensor_msgs/Image` | `camera_optical_frame` | left colour camera, rgb8 1242x375, 10 Hz, rectified |
| `/sensing/camera/front/camera_info` | `sensor_msgs/CameraInfo` | `camera_optical_frame` | zero distortion |
| `/sensing/lidar/pointcloud` | `sensor_msgs/PointCloud2` | `lidar_link` | Velodyne HDL-64E, fields x, y, z, intensity, ring, time |
| `/sensing/imu/data` | `sensor_msgs/Imu` | `imu_link` | OXTS, 10 Hz |
| `/tf_static` | `tf2_msgs/TFMessage` | | `base_link` -> `lidar_link`, `imu_link`, `camera_link` -> `camera_optical_frame`; transient_local |
| `/ground_truth/tf` | `tf2_msgs/TFMessage` | `kitti_map` -> `base_link` | OXTS pose |
| `/ground_truth/odometry` | `nav_msgs/Odometry` | `kitti_map` | OXTS pose and body-frame twist |
| `/ground_truth/objects` | `vision_msgs/Detection3DArray` | `base_link` | tracklet boxes, one array per image |

## Differences from KITTI and from the AP1 vehicle

- `base_link` is KITTI's OXTS/IMU unit, not the AP1 vehicle origin. Take
  sensor poses from `/tf_static`.
- `ring` and `time` are reconstructed: rings from KITTI's laser-by-laser point
  order (nominal HDL-64E elevations for the few sweeps where that fails),
  times from the azimuth of the counter-clockwise sweep, starting at the
  forward-facing stamp. Fine for perception, not for lidar deskewing.
- Ground-truth classes are AP1 class strings: Pedestrian and Person (sitting)
  are `pedestrian`, Cyclist and Misc are `unknown`; cars, vans, trucks and
  trams are dropped. Only boxes at least partly inside the camera image are kept.
- KITTI's map -> base_link pose is on `/ground_truth/tf`, not `/tf`. Remap it
  to `/tf` on playback when no localization runs.

## Use

```bash
ros2 bag play 2011_09_26_drive_0005 --clock
```

In the AP1 repo, `scripts/fetch_bags.py` downloads pinned revisions of these
bags into `bags/`.

## License and attribution

The KITTI dataset is copyright its authors (Karlsruhe Institute of Technology
and Toyota Technological Institute at Chicago) and published under
[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). These
bags are a derived work under the same license: non-commercial use only, with
attribution, and anything built on them must be shared under the same
license.

If you use them, cite the KITTI raw dataset:

```bibtex
@article{Geiger2013IJRR,
  author = {Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun},
  title = {Vision meets Robotics: The KITTI Dataset},
  journal = {International Journal of Robotics Research (IJRR)},
  year = {2013}
}
```
