# ap1-datasets

Builds the datasets WE-Autopilot publishes on Hugging Face under
[WEAP26](https://huggingface.co/WEAP26) for AP1 development. The AP1 repo
only downloads the results (`scripts/fetch_bags.py` there, pinned to a
dataset commit); everything that produces them lives here.

| Dataset | Source | Code |
|---|---|---|
| [WEAP26/KITTI_mcap](https://huggingface.co/datasets/WEAP26/KITTI_mcap) | KITTI raw drives | `src/ap1_datasets/kitti/` |

Bags follow the AP1.26 sensing contract (ICD topic and frame names, latched
`/tf_static`, ground truth on `/ground_truth/*` instead of `/tf`), so AP1
nodes consume them like CARLA or kart recordings.

## Setup

The environment is [pixi](https://pixi.sh): ROS 2 Jazzy from RoboStack, no
system ROS or Docker. Linux x86-64 only for now (WSL works); add platforms to
`[tool.pixi.workspace]` in `pyproject.toml` to support others.

```bash
pixi install
pixi run test
pixi run lint
```

## KITTI

```bash
pixi run build-kitti            # all drives in drives.yaml; --drive 0005 for one
hf auth login                   # write token for the WEAP26 org, once
pixi run upload-kitti
```

`build-kitti` downloads each drive listed in `src/ap1_datasets/kitti/drives.yaml`
(sha256-checked), converts it with the pinned kitti2mcap, then rewrites it into
an AP1 bag with `adapter.py` (topic and frame names in `config.yaml`). The
result is staged in `out/kitti/hf/`: one rosbag2 directory per drive plus
`dataset_card.md` as `README.md`. `upload-kitti` uploads that folder as one
commit to the dataset. Build from a clean, pushed checkout: each bag's
`provenance.yaml` records this repo's commit (`+dirty` with local changes).

After uploading, update `revision` in AP1's `scripts/bags.yaml` to the new
dataset commit (shown on the dataset's "Files and versions" tab).

Adding a drive: append it to `drives.yaml` with the sha256 of its zips, rebuild
and upload, and list it in `bags.yaml` in AP1.

What the adapter does and where it differs from KITTI is in the dataset card
(`src/ap1_datasets/kitti/dataset_card.md`). KITTI is CC BY-NC-SA 3.0, so the
bags are too.
