"""Unit tests for KITTI lidar ring/time reconstruction. No ROS needed."""
import numpy as np

from ap1_datasets.kitti.lidar import (
    HDL64_ELEVATIONS_DEG,
    XYZIRT,
    assign_rings,
    point_times,
    rings_from_elevation,
    rings_from_scan_order,
    to_xyzirt,
    xyzi_from_buffer,
)


def points_at(elev_deg, azimuth_deg, r=20.0):
    e, a = np.radians(elev_deg), np.radians(azimuth_deg)
    return np.stack([r * np.cos(e) * np.cos(a), r * np.cos(e) * np.sin(a), r * np.sin(e)], axis=1)


def kitti_like_sweep(elevations, n_per_laser=90, gap=None):
    """Laser by laser, each run starting just left of forward and turning counter-clockwise."""
    runs = []
    for elev in elevations:
        az = np.linspace(0.5, 359.5, n_per_laser)
        if gap is not None:
            az = az[(az < gap[0]) | (az > gap[1])]
        az = np.where(az > 180, az - 360, az)     # atan2 range
        runs.append(points_at(np.full(len(az), elev), az))
    return np.vstack(runs)


def test_beam_table_is_hdl64():
    assert len(HDL64_ELEVATIONS_DEG) == 64
    assert np.all(np.diff(HDL64_ELEVATIONS_DEG) > 0)
    assert HDL64_ELEVATIONS_DEG[-1] == 2.0


def test_scan_order_rings_rank_lasers_by_elevation():
    rng = np.random.default_rng(0)
    elev = rng.permutation(np.linspace(-24, 2, 64))   # storage order isn't elevation order
    xyz = kitti_like_sweep(elev)
    rings, method = assign_rings(xyz)
    assert method == 'scan_order'
    per_laser = rings.reshape(64, -1)
    assert np.all(per_laser == per_laser[:, :1])       # one ring per laser run
    assert np.array_equal(per_laser[:, 0], np.argsort(np.argsort(elev)))
    assert rings.dtype == np.uint16


def test_scan_order_survives_missing_returns_near_the_front():
    xyz = kitti_like_sweep(np.linspace(-24, 2, 64), gap=(340, 359.9))
    assert rings_from_scan_order(xyz) is not None


def test_falls_back_to_elevation_when_a_laser_is_missing():
    xyz = kitti_like_sweep(HDL64_ELEVATIONS_DEG[:63])
    assert rings_from_scan_order(xyz) is None
    rings, method = assign_rings(xyz)
    assert method == 'elevation'
    assert np.array_equal(np.unique(rings), np.arange(63))


def test_elevation_rings_snap_to_nearest_beam():
    elev = HDL64_ELEVATIONS_DEG[[0, 10, 40, 63]] + [0.05, 0.1, -0.1, 0.05]
    rings = rings_from_elevation(points_at(elev, [0.0, 30.0, -120.0, 90.0]))
    assert list(rings) == [0, 10, 40, 63]


def test_times_counter_clockwise_from_forward_stamp():
    t = point_times(points_at(np.zeros(4), [0.0, 90.0, 180.0, -90.0]))
    assert np.allclose(t, [0.0, 0.025, 0.05, 0.075], atol=1e-6)
    assert t.dtype == np.float32
    t_cw = point_times(points_at(np.zeros(2), [90.0, -90.0]), counter_clockwise=False)
    assert np.allclose(t_cw, [0.075, 0.025], atol=1e-6)


def test_to_xyzirt_layout_matches_mapping():
    xyzi = np.hstack([kitti_like_sweep(np.linspace(-24, 2, 64), 8),
                      np.full((64 * 8, 1), 0.5)]).astype(np.float32)
    out, method = to_xyzirt(xyzi)
    assert method == 'scan_order'
    assert out.dtype == XYZIRT and XYZIRT.itemsize == 24
    assert [XYZIRT.fields[n][1] for n in XYZIRT.names] == [0, 4, 8, 12, 16, 20]
    assert np.all(out['intensity'] == np.float32(0.5))
    assert np.all((out['time'] >= 0) & (out['time'] < 0.1))


def test_xyzi_from_buffer_respects_offsets_and_padding():
    # KITTI-like layout with 4 bytes of padding after intensity
    dt = np.dtype({'names': ['x', 'y', 'z', 'intensity'], 'formats': ['<f4'] * 4,
                   'offsets': [0, 4, 8, 12], 'itemsize': 20})
    raw = np.zeros(3, dtype=dt)
    for i, name in enumerate(dt.names):
        raw[name] = np.arange(3) + 10 * i
    xyzi = xyzi_from_buffer(raw.tobytes(), {'x': 0, 'y': 4, 'z': 8, 'intensity': 12}, 20, 3)
    assert np.array_equal(xyzi[:, 3], [30, 31, 32])
    assert np.array_equal(xyzi[:, 0], [0, 1, 2])
