"""KITTI Velodyne HDL-64E clouds -> Velodyne-style XYZIRT, the ICD lidar layout.

ICD #7 (/sensing/lidar/pointcloud) requires per-point ``ring`` and ``time``.
KITTI's clouds carry only x, y, z, intensity, so both are reconstructed. The
facts used are from the KITTI raw devkit readme: the scanner rotates
counter-clockwise, the cameras fire when it faces forward, and the points are
not motion compensated.

- ring: KITTI stores each sweep laser by laser, each laser starting at the
  front and turning counter-clockwise. A laser boundary is where the azimuth
  steps from just right of forward (negative) back to just left of it. The 64
  runs are ranked by elevation so ring 0 is the lowest beam (the numbering of
  mapping's filter). If a sweep doesn't split into exactly 64 runs, rings fall
  back to the nearest nominal HDL-64E elevation, which is rougher.
- time: seconds since the header stamp. The stamp is the forward-facing
  moment (camera trigger), which is also where each laser's run starts, so a
  point at counter-clockwise azimuth a was measured a / 2pi of a sweep later.

Good enough for perception and visualization, not for LIO-SAM deskewing.
"""
import numpy as np

N_BEAMS = 64

# Nominal HDL-64E elevations (degrees), fallback only: upper block +2.0 down in
# 1/3 deg steps, lower block -8.83 down in 1/2 deg steps. Ascending = ring.
HDL64_ELEVATIONS_DEG = np.sort(np.concatenate([
    2.0 - np.arange(32) / 3.0,
    -8.83 - 0.5 * np.arange(32),
]))

# Same layout as mapping's filtered bags (simulation/filter_bag.py, XYZIRT)
XYZIRT = np.dtype({
    'names': ['x', 'y', 'z', 'intensity', 'ring', 'time'],
    'formats': ['<f4', '<f4', '<f4', '<f4', '<u2', '<f4'],
    'offsets': [0, 4, 8, 12, 16, 20],
    'itemsize': 24,
})


def elevations_deg(xyz):
    return np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))


def rings_from_scan_order(xyz, n_beams=N_BEAMS):
    """Rings from KITTI's laser-by-laser point order, or None if it doesn't fit.

    Each laser's run starts just left of forward (azimuth >= 0) and ends just
    right of it (azimuth < 0), so a boundary is a small negative -> positive
    step. Runs are ranked by median elevation: ring 0 is the lowest.
    """
    if len(xyz) < n_beams:
        return None
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    step = np.diff(az)
    starts = np.flatnonzero((az[:-1] < 0) & (az[1:] >= 0) & (step < np.pi / 2)) + 1
    if len(starts) != n_beams - 1:
        return None
    run = np.zeros(len(xyz), dtype=np.int64)
    run[starts] = 1
    run = np.cumsum(run)
    elev = elevations_deg(xyz)
    median = np.array([np.median(elev[run == k]) for k in range(n_beams)])
    rank = np.empty(n_beams, dtype=np.uint16)
    rank[np.argsort(median)] = np.arange(n_beams)
    return rank[run]


def rings_from_elevation(xyz, beams_deg=HDL64_ELEVATIONS_DEG):
    """Ring of each point: the nominal beam closest in elevation (fallback)."""
    edges = (beams_deg[1:] + beams_deg[:-1]) / 2
    return np.searchsorted(edges, elevations_deg(xyz)).astype(np.uint16)


def assign_rings(xyz):
    """(rings, method): scan order when the sweep allows it, else nominal elevation."""
    rings = rings_from_scan_order(xyz)
    if rings is not None:
        return rings, 'scan_order'
    return rings_from_elevation(xyz), 'elevation'


def point_times(xyz, sweep_period_s=0.1, counter_clockwise=True):
    """Seconds since the forward-facing stamp, in [0, sweep_period_s).

    Azimuth 0 is forward (+x), counter-clockwise positive (REP 103).
    """
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    if not counter_clockwise:
        azimuth = -azimuth
    turned = np.mod(azimuth, 2 * np.pi)
    return (turned / (2 * np.pi) * sweep_period_s).astype(np.float32)


def to_xyzirt(xyzi, sweep_period_s=0.1, counter_clockwise=True):
    """(XYZIRT array, ring method) from an (N, 4) x, y, z, intensity array."""
    out = np.zeros(len(xyzi), dtype=XYZIRT)
    for i, name in enumerate(['x', 'y', 'z', 'intensity']):
        out[name] = xyzi[:, i]
    out['ring'], method = assign_rings(xyzi[:, :3])
    out['time'] = point_times(xyzi[:, :3], sweep_period_s, counter_clockwise)
    return out, method


def xyzi_from_buffer(data, fields, point_step, count):
    """(N, 4) float array of x, y, z, intensity from raw PointCloud2 bytes.

    fields maps name -> byte offset; every field must be float32.
    """
    raw = np.frombuffer(bytes(data), dtype=np.uint8).reshape(count, point_step)
    out = np.empty((count, 4), dtype=np.float32)
    for i, name in enumerate(['x', 'y', 'z', 'intensity']):
        off = fields[name]
        out[:, i] = raw[:, off:off + 4].copy().view('<f4').ravel()
    return out
