#!/usr/bin/env python3
# Companion for the gz-gui PointCloud panel. Two upstream quirks of that
# plugin make it useless on its own for our Kinect:
#   1. Its render loop is bounded by floatVMsg.data().size(), so it draws
#      nothing unless a matching Float_V is also streaming.
#   2. It never sets marker.parent, so the cloud always renders at the
#      world origin instead of following the sensor.
#
# This helper fixes both. It subscribes to /kinect/points (sensor-local
# frame) and the Gazebo pose streams for camera_link_optical, transforms
# the points into world frame, republishes them on /kinect/points_world,
# and publishes a matching Float_V on /kinect/points_colors. The plugin
# then renders /kinect/points_world at the world origin — which is where
# the points actually are — and the cloud follows the robot correctly.
#
# Subsampling + rate cap keep CPU bounded regardless of the camera update
# rate. numpy is used for the per-point math (rotation + translation)
# because Python-level loops over ~300k floats are too slow.

import os
import signal
import sys
import time

import numpy as np

from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked
from gz.msgs10.float_v_pb2 import Float_V
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node


POINT_CLOUD_IN = os.environ.get("WANIS_POINTCLOUD_IN", "/kinect/points")
POINT_CLOUD_OUT = os.environ.get("WANIS_POINTCLOUD_OUT", "/kinect/points_world")
FLOAT_V_OUT = os.environ.get("WANIS_FLOAT_V_OUT", "/kinect/points_colors")
WORLD_NAME = os.environ.get("WANIS_WORLD_NAME", "wanis_apartment")
MODEL_NAME = os.environ.get("WANIS_MODEL_NAME", "wanis_4x4")

# camera_link_optical pose relative to the model origin, from model.sdf:
#   <pose>0.16 0.09 0.348 -1.5708 0 -1.5708</pose>
# Fixed joint, so the offset is constant — composing it with the model's
# world pose gives the sensor's world pose without needing Gazebo to
# publish the nested link pose directly (the pose/info stream publishes
# link poses relative to the model, not to the world, which is why the
# earlier straight-read version placed the cloud next to the robot).
SENSOR_IN_MODEL_XYZ = (0.16, 0.09, 0.348)
SENSOR_IN_MODEL_RPY = (0.0, 0.0 , 0.0)

MAX_POINTS = 4000
MIN_PERIOD_S = 0.15

_model_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
_model_pose_seen = False
_pose_names_logged = False


def _on_pose(msg: Pose_V) -> None:
    global _model_pose, _model_pose_seen, _pose_names_logged
    if not _pose_names_logged:
        names = [p.name for p in msg.pose]
        print(f"[pointcloud_colorizer] pose/info names: {names}", flush=True)
        _pose_names_logged = True
    for p in msg.pose:
        if p.name == MODEL_NAME:
            _model_pose = np.array([
                p.position.x, p.position.y, p.position.z,
                p.orientation.x, p.orientation.y,
                p.orientation.z, p.orientation.w,
            ])
            if not _model_pose_seen:
                print(f"[pointcloud_colorizer] first model pose: "
                      f"pos=({p.position.x:.3f},{p.position.y:.3f},"
                      f"{p.position.z:.3f}) "
                      f"quat=({p.orientation.x:.3f},{p.orientation.y:.3f},"
                      f"{p.orientation.z:.3f},{p.orientation.w:.3f})",
                      flush=True)
                _model_pose_seen = True
            return


def _field_offsets(msg: PointCloudPacked):
    x_off = y_off = z_off = None
    for f in msg.field:
        if f.name == "x":
            x_off = f.offset
        elif f.name == "y":
            y_off = f.offset
        elif f.name == "z":
            z_off = f.offset
    return x_off, y_off, z_off


def _quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw),     1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw),     2 * (qy*qz + qx*qw),     1 - 2 * (qx*qx + qy*qy)],
    ], dtype=np.float32)


def _rpy_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


# Static rotation and translation of camera_link_optical in the model frame.
_R_SENSOR_IN_MODEL = _rpy_to_matrix(*SENSOR_IN_MODEL_RPY)
_T_SENSOR_IN_MODEL = np.array(SENSOR_IN_MODEL_XYZ, dtype=np.float32)


def main() -> int:
    node = Node()

    cloud_pub = node.advertise(POINT_CLOUD_OUT, PointCloudPacked)
    if not cloud_pub:
        print(f"[pointcloud_colorizer] advertise {POINT_CLOUD_OUT} failed",
              file=sys.stderr, flush=True)
        return 1
    float_pub = node.advertise(FLOAT_V_OUT, Float_V)
    if not float_pub:
        print(f"[pointcloud_colorizer] advertise {FLOAT_V_OUT} failed",
              file=sys.stderr, flush=True)
        return 1

    # Subscribe to both pose streams. dynamic_pose/info updates each tick
    # while the entity is moving; pose/info carries the full snapshot at a
    # slower cadence and covers the stationary case.
    for topic in (f"/world/{WORLD_NAME}/dynamic_pose/info",
                  f"/world/{WORLD_NAME}/pose/info"):
        if not node.subscribe(Pose_V, topic, _on_pose):
            print(f"[pointcloud_colorizer] subscribe {topic} failed",
                  file=sys.stderr, flush=True)

    last_publish = [0.0]
    pushed = [0]

    def on_cloud(msg: PointCloudPacked) -> None:
        now = time.monotonic()
        if now - last_publish[0] < MIN_PERIOD_S:
            return

        point_step = msg.point_step
        if point_step == 0 or not msg.data:
            return
        num_points = len(msg.data) // point_step
        if num_points == 0:
            return
        x_off, y_off, z_off = _field_offsets(msg)
        if None in (x_off, y_off, z_off):
            return

        stride = max(1, num_points // MAX_POINTS)
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        rows = np.arange(0, num_points, stride, dtype=np.int64)
        base = rows * point_step
        # Gather x,y,z floats per selected point. When x,y,z are contiguous
        # 4-byte floats (true for Gazebo rgbd_camera output) we pull 12
        # bytes in one shot; otherwise we pull each axis individually.
        if y_off == x_off + 4 and z_off == x_off + 8:
            idx = (base[:, None] + x_off + np.arange(12)).ravel()
            xyz = buf[idx].view(np.float32).reshape(-1, 3)
        else:
            def pull(offset):
                idx = (base[:, None] + offset + np.arange(4)).ravel()
                return buf[idx].view(np.float32)
            xyz = np.stack([pull(x_off), pull(y_off), pull(z_off)], axis=1)

        # Drop NaN/Inf and out-of-range points.
        finite = np.isfinite(xyz).all(axis=1)
        zmask = (xyz[:, 2] > 0.0) & (xyz[:, 2] < 8.0)
        xyz = xyz[finite & zmask]
        n = xyz.shape[0]
        if n == 0:
            return

        # Compose world pose of the sensor:
        #   T_world_sensor = T_world_model @ T_model_sensor
        # T_world_model comes from the pose stream (updates as the robot
        # moves); T_model_sensor is the SDF-fixed offset for camera_link_optical.
        px, py, pz, qx, qy, qz, qw = _model_pose
        R_model = _quat_to_matrix(qx, qy, qz, qw)
        R_sensor_world = (R_model @ _R_SENSOR_IN_MODEL).astype(np.float32)
        T_sensor_world = (
            R_model @ _T_SENSOR_IN_MODEL
            + np.array([px, py, pz], dtype=np.float32)
        ).astype(np.float32)

        # Transform sensor-local points into world frame.
        world = xyz @ R_sensor_world.T + T_sensor_world

        # Build an output PointCloudPacked with a compact xyz-only layout.
        out = PointCloudPacked()
        out.height = 1
        out.width = n
        out.point_step = 12
        out.row_step = 12 * n
        out.is_dense = True
        fx = out.field.add(); fx.name = "x"; fx.offset = 0
        fx.datatype = PointCloudPacked.Field.FLOAT32; fx.count = 1
        fy = out.field.add(); fy.name = "y"; fy.offset = 4
        fy.datatype = PointCloudPacked.Field.FLOAT32; fy.count = 1
        fz = out.field.add(); fz.name = "z"; fz.offset = 8
        fz.datatype = PointCloudPacked.Field.FLOAT32; fz.count = 1
        out.data = world.astype(np.float32).tobytes()

        cloud_pub.publish(out)

        floats = Float_V()
        floats.data.extend([1.0] * n)
        float_pub.publish(floats)

        last_publish[0] = now
        pushed[0] += 1
        if pushed[0] in (1, 3, 10) or pushed[0] % 60 == 0:
            print(f"[pointcloud_colorizer] n={pushed[0]} pts={n} "
                  f"sensor=({px:.2f},{py:.2f},{pz:.2f})", flush=True)

    if not node.subscribe(PointCloudPacked, POINT_CLOUD_IN, on_cloud):
        print(f"[pointcloud_colorizer] subscribe {POINT_CLOUD_IN} failed",
              file=sys.stderr, flush=True)
        return 1

    print(f"[pointcloud_colorizer] {POINT_CLOUD_IN} -> {POINT_CLOUD_OUT} + "
          f"{FLOAT_V_OUT} (model={MODEL_NAME}, "
          f"period={MIN_PERIOD_S}s, max_pts={MAX_POINTS})", flush=True)

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())
