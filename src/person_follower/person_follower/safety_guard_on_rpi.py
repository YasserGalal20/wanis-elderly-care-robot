#!/usr/bin/env python3
"""
safety_guard.py  –  Last-line-of-defence collision prevention for the Wanis robot.

Architecture
────────────
  other nodes ──► /cmd_vel_raw ──► THIS NODE ──► /cmd_vel ──► twist_stamper ──► motors

The node builds a rectangular safety polygon from the URDF-derived robot
footprint plus a configurable margin, publishes it to RViz2, and checks
every incoming LaserScan against it.  If *any* scan point falls inside the
safety polygon it:
  1. Vetoes the velocity command in the dangerous direction(s)
  2. Publishes zero (or reduced) velocity so no other node can override
  3. Logs a warning with the nearest obstacle distance and angle

The polygon changes colour in RViz2:
  • GREEN  – all clear
  • YELLOW – obstacle within warning zone
  • RED    – obstacle inside safety zone, motion vetoed
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist, Point32, PolygonStamped, Point
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _rect_polygon(half_x: float, half_y: float, n_side: int = 4):
    """Return corners of an axis-aligned rectangle centred at origin."""
    return [
        (+half_x, +half_y),
        (+half_x, -half_y),
        (-half_x, -half_y),
        (-half_x, +half_y),
    ]


def _inflate_rect(half_x: float, half_y: float, margin: float):
    """Inflate the rectangle by *margin* on every side."""
    return _rect_polygon(half_x + margin, half_y + margin)


def _point_in_rect(px: float, py: float, half_x: float, half_y: float) -> bool:
    return abs(px) <= half_x and abs(py) <= half_y


# ────────────────────────────────────────────────────────────────────
# ROS 2 Node
# ────────────────────────────────────────────────────────────────────

class SafetyGuard(Node):

    def __init__(self):
        super().__init__("safety_guard")

        # ── Robot footprint (from URDF: chassis 0.80 × 0.41 m) ──────
        self.declare_parameter("robot_length", 0.80)   # x extent  (front-back)
        self.declare_parameter("robot_width",  0.55)    # y extent  (left-right, wheels included)

        # ── Safety / warning margins (metres) ───────────────────────
        self.declare_parameter("safety_margin",  0.25)  # hard-stop zone around footprint
        self.declare_parameter("warning_margin", 0.50)  # yellow-warning zone

        # ── Behaviour tuning ────────────────────────────────────────
        self.declare_parameter("scan_topic",       "/scan")
        self.declare_parameter("cmd_vel_in_topic", "/cmd_vel_raw")
        self.declare_parameter("cmd_vel_out_topic", "/cmd_vel")
        self.declare_parameter("base_frame",       "base_footprint")
        self.declare_parameter("rate_hz",          20.0)        # marker publish rate
        # Reverse-escape is off by default: backing up blindly can hit things
        # the rear lidar beam can't see (low obstacles behind).  Prefer STOP.
        self.declare_parameter("allow_reverse_escape", True)
        self.declare_parameter("escape_speed",     -0.10)       # m/s reverse escape cap
        # If we have never heard a scan or PCL after startup, it is safer to
        # refuse to move than to drive blind.
        self.declare_parameter("require_sensor_before_motion", True)

        # ── 3D front obstacle check (from Kinect point cloud) ───────
        # The 2D lidar sits at ~0.47 m height so it misses overhangs,
        # low tables and other obstacles that the 2D scan can't see.
        # The point cloud check catches them.
        self.declare_parameter("pcl_topic",           "/kinect2/points")
        self.declare_parameter("pcl_enabled",          True)
        self.declare_parameter("pcl_z_min",            0.04)    # ignore floor
        self.declare_parameter("pcl_z_max",            1.80)    # ignore ceiling / high overhangs
        self.declare_parameter("pcl_min_points",       3)       # reject single-point noise (small clusters OK)
        self.declare_parameter("pcl_subsample",        2)       # numpy path is fast, keep dense
        self.declare_parameter("pcl_stale_timeout",    0.4)     # sec; if PCL stops, fail-safe block forward
        self.declare_parameter("brake_decel",          0.8)     # m/s^2 assumed deceleration
        self.declare_parameter("reaction_time",        0.25)    # s latency between detect and stop
        # Tight dead-ahead corridor: any PCL point here (even a single one)
        # triggers a hard block — catches thin obstacles like chair/table legs.
        self.declare_parameter("pcl_center_corridor_half_y", 0.18)  # m (dead-ahead half-width)
        self.declare_parameter("pcl_center_trip_dist", 1.20)        # m — any center point closer than this blocks
        self.declare_parameter("emergency_stop_dist", 0.35)         # m — full stop, no rotation, if anything is this close
        self.declare_parameter("health_log_period",   3.0)          # s between diagnostic logs
        # Camera link offset in base frame (matches the URDF:
        # camera_joint origin xyz="0.16 0.09 0.348" rpy="0 0 0").
        self.declare_parameter("cam_offset_x", 0.16)
        self.declare_parameter("cam_offset_y", 0.09)
        self.declare_parameter("cam_offset_z", 0.348)

        # ── Stair / drop-off detection ───────────────────────────────
        # Detects missing floor ahead (drop-off / stair descent) and
        # sudden rising surfaces at close range (stair ascent).
        # Works on the low-z band of the point cloud, independent of
        # the existing obstacle height filter (pcl_z_min / pcl_z_max).
        self.declare_parameter("stair_detect_enabled",   True)
        self.declare_parameter("stair_floor_bz_max",     0.15)  # floor band upper bound in base frame
        self.declare_parameter("stair_floor_bz_min",    -0.35)  # floor band lower bound
        self.declare_parameter("stair_look_ahead_min",   0.30)  # start looking at this bx (m)
        self.declare_parameter("stair_look_ahead_max",   0.90)  # stop looking at this bx (m)
        self.declare_parameter("stair_min_floor_pts",    6)     # < this → floor absent → drop-off
        self.declare_parameter("stair_ascent_bz_min",    0.08)  # rising surface band low (m)
        self.declare_parameter("stair_ascent_bz_max",    0.50)  # rising surface band high (m)
        self.declare_parameter("stair_ascent_pts",       25)    # min points for ascent detection
        self.declare_parameter("stair_ascent_dist",      0.65)  # bx < this triggers ascent check

        # Read parameters
        self.robot_half_x = self.get_parameter("robot_length").value / 2.0
        self.robot_half_y = self.get_parameter("robot_width").value  / 2.0
        self.safety_margin  = self.get_parameter("safety_margin").value
        self.warning_margin = self.get_parameter("warning_margin").value
        self.base_frame     = self.get_parameter("base_frame").value
        self.allow_reverse  = self.get_parameter("allow_reverse_escape").value
        self.escape_speed   = self.get_parameter("escape_speed").value

        # Derived half-extents for the safety and warning zones
        self.safe_half_x = self.robot_half_x + self.safety_margin
        self.safe_half_y = self.robot_half_y + self.safety_margin
        self.warn_half_x = self.robot_half_x + self.warning_margin
        self.warn_half_y = self.robot_half_y + self.warning_margin

        # ── State ───────────────────────────────────────────────────
        self.latest_scan: LaserScan | None = None
        self.latest_cmd = Twist()
        self.obstacle_in_safety  = False
        self.obstacle_in_warning = False
        self.nearest_dist = float("inf")
        self.nearest_angle = 0.0
        self.blocked_front = False
        self.blocked_rear  = False
        self.blocked_left  = False
        self.blocked_right = False
        self.veto_count = 0   # total vetoes since start

        # 3D obstacle flags (populated from point cloud; merged into
        # blocked_front / obstacle_in_safety when present).
        self.pcl_blocked_front = False
        self.pcl_blocked_front_warn = False
        self.pcl_nearest_dist = float("inf")
        self.pcl_point_count = 0
        self.cam_x = self.get_parameter("cam_offset_x").value
        self.cam_y = self.get_parameter("cam_offset_y").value
        self.cam_z = self.get_parameter("cam_offset_z").value
        self.pcl_enabled      = bool(self.get_parameter("pcl_enabled").value)
        self.pcl_z_min        = float(self.get_parameter("pcl_z_min").value)
        self.pcl_z_max        = float(self.get_parameter("pcl_z_max").value)
        self.pcl_min_points   = max(1, int(self.get_parameter("pcl_min_points").value))
        self.pcl_subsample    = max(1, int(self.get_parameter("pcl_subsample").value))
        self.pcl_stale_timeout = float(self.get_parameter("pcl_stale_timeout").value)
        self.brake_decel      = max(0.1, float(self.get_parameter("brake_decel").value))
        self.reaction_time    = max(0.0, float(self.get_parameter("reaction_time").value))
        self.pcl_center_half_y = max(0.05, float(self.get_parameter("pcl_center_corridor_half_y").value))
        self.pcl_center_trip  = max(0.2, float(self.get_parameter("pcl_center_trip_dist").value))
        self.emergency_stop_dist = max(0.1, float(self.get_parameter("emergency_stop_dist").value))
        self.require_sensor   = bool(self.get_parameter("require_sensor_before_motion").value)

        # Corridor distances from the latest PCL frame — populated in
        # _pcl_cb, consumed in _cmd_cb for dynamic (speed-aware) checks.
        self.pcl_corridor_bx: np.ndarray | None = None
        self.pcl_corridor_by: np.ndarray | None = None
        self.pcl_last_stamp = None          # rclpy Time, None = never received
        self.pcl_stale_warned = False
        self.pcl_center_nearest = float("inf")  # nearest point in tight dead-ahead corridor

        # Stair / hazard detection state
        self.stair_detect_enabled = bool(self.get_parameter("stair_detect_enabled").value)
        self._stair_floor_bz_max  = float(self.get_parameter("stair_floor_bz_max").value)
        self._stair_floor_bz_min  = float(self.get_parameter("stair_floor_bz_min").value)
        self._stair_look_min      = float(self.get_parameter("stair_look_ahead_min").value)
        self._stair_look_max      = float(self.get_parameter("stair_look_ahead_max").value)
        self._stair_min_floor_pts = max(1, int(self.get_parameter("stair_min_floor_pts").value))
        self._stair_ascent_bz_min = float(self.get_parameter("stair_ascent_bz_min").value)
        self._stair_ascent_bz_max = float(self.get_parameter("stair_ascent_bz_max").value)
        self._stair_ascent_pts    = max(1, int(self.get_parameter("stair_ascent_pts").value))
        self._stair_ascent_dist   = float(self.get_parameter("stair_ascent_dist").value)
        # Rolling confirmation: require N consecutive frames to avoid false positives
        self._stair_descent_frames = 0
        self._stair_ascent_frames  = 0
        self._STAIR_CONFIRM_FRAMES = 3
        self.stair_descent_warning = False
        self.stair_ascent_warning  = False

        # Startup / sensor-presence tracking
        self.node_start_time = self.get_clock().now()
        self.scan_received = False

        # ── Subscribers ─────────────────────────────────────────────
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5,
        )
        pcl_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=5,
        )
        self.create_subscription(
            LaserScan,
            self.get_parameter("scan_topic").value,
            self._scan_cb,
            scan_qos,
        )
        self.create_subscription(
            Twist,
            self.get_parameter("cmd_vel_in_topic").value,
            self._cmd_cb,
            10,
        )

        if self.pcl_enabled:
            self.create_subscription(
                PointCloud2,
                self.get_parameter("pcl_topic").value,
                self._pcl_cb,
                pcl_qos
            )

        # ── Publishers ──────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(
            Twist,
            self.get_parameter("cmd_vel_out_topic").value,
            10,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, "/safety_guard/markers", 10
        )
        self.footprint_pub = self.create_publisher(
            PolygonStamped, "/safety_guard/footprint", 10
        )
        self.safety_zone_pub = self.create_publisher(
            PolygonStamped, "/safety_guard/safety_zone", 10
        )
        self.warning_zone_pub = self.create_publisher(
            PolygonStamped, "/safety_guard/warning_zone", 10
        )

        # ── Timer for marker publishing ─────────────────────────────
        rate = self.get_parameter("rate_hz").value
        self.create_timer(1.0 / rate, self._publish_markers)

        # ── Periodic health log (PCL/lidar flow visibility) ─────────
        health_period = max(0.5, float(self.get_parameter("health_log_period").value))
        self.create_timer(health_period, self._health_log)

        self.get_logger().info(
            f"SafetyGuard active  |  footprint {self.robot_half_x*2:.2f}×{self.robot_half_y*2:.2f} m  "
            f"|  safety margin {self.safety_margin:.2f} m  |  warning margin {self.warning_margin:.2f} m  "
            f"|  PCL={'on' if self.pcl_enabled else 'off'}  "
            f"|  reverse_escape={'on' if self.allow_reverse else 'off'}  "
            f"|  listening on '{self.get_parameter('cmd_vel_in_topic').value}' → "
            f"publishing on '{self.get_parameter('cmd_vel_out_topic').value}'"
        )

    # ────────────────────────────────────────────────────────────────
    # Callbacks
    # ────────────────────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        """Process laser scan and update obstacle flags.

        All aggregation is done in locals and committed atomically at the
        end so that _cmd_cb can never observe a half-reset state (i.e. a
        window where everything has been cleared but nothing has been
        re-populated yet — a race that would silently let the robot move
        through an obstacle).
        """
        self.latest_scan = msg

        obstacle_in_safety  = False
        obstacle_in_warning = False
        blocked_front = False
        blocked_rear  = False
        blocked_left  = False
        blocked_right = False
        nearest_dist  = float("inf")
        nearest_angle = 0.0

        angle = msg.angle_min
        r_min = msg.range_min
        r_max = msg.range_max
        inc = msg.angle_increment
        for r in msg.ranges:
            if math.isnan(r) or math.isinf(r) or r < r_min or r > r_max:
                angle += inc
                continue

            px = r * math.cos(angle)
            py = r * math.sin(angle)

            if r < nearest_dist:
                nearest_dist = r
                nearest_angle = angle

            if _point_in_rect(px, py, self.warn_half_x, self.warn_half_y):
                obstacle_in_warning = True

            if _point_in_rect(px, py, self.safe_half_x, self.safe_half_y):
                obstacle_in_safety = True
                if px > 0:
                    blocked_front = True
                if px < 0:
                    blocked_rear = True
                if py > 0:
                    blocked_left = True
                if py < 0:
                    blocked_right = True

            angle += inc

        # Atomic commit
        self.obstacle_in_safety  = obstacle_in_safety
        self.obstacle_in_warning = obstacle_in_warning
        self.blocked_front = blocked_front
        self.blocked_rear  = blocked_rear
        self.blocked_left  = blocked_left
        self.blocked_right = blocked_right
        self.nearest_dist  = nearest_dist
        self.nearest_angle = nearest_angle
        self.scan_received = True

    # ────────────────────────────────────────────────────────────────
    # Point cloud obstacle check — catches 3D obstacles the 2D scan
    # can't see (low tables, overhangs, chair seats, etc.)
    # Vectorised with numpy so we can process the entire cloud each
    # frame without falling behind at 15 Hz.
    # ────────────────────────────────────────────────────────────────
    def _pcl_cb(self, msg: PointCloud2):
        offsets = {f.name: (f.offset, f.datatype) for f in msg.fields}
        if not {"x", "y", "z"}.issubset(offsets):
            return

        point_step = msg.point_step
        n_total = msg.width * msg.height
        if n_total == 0 or point_step == 0:
            return

        ox, dtx = offsets["x"]
        oy, dty = offsets["y"]
        oz, dtz = offsets["z"]
        if not (dtx == dty == dtz == PointField.FLOAT32):
            # Only FLOAT32 clouds supported (matches gz rgbd_camera / ros2 depth).
            return

        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8)
        except (TypeError, ValueError):
            return
        expected = n_total * point_step
        if raw.size < expected:
            return

        grid = raw[:expected].reshape(n_total, point_step)
        xs = grid[:, ox:ox + 4].copy().view(np.float32).ravel()
        ys = grid[:, oy:oy + 4].copy().view(np.float32).ravel()
        zs = grid[:, oz:oz + 4].copy().view(np.float32).ravel()

        stride = self.pcl_subsample
        if stride > 1:
            xs = xs[::stride]
            ys = ys[::stride]
            zs = zs[::stride]

        # Depth (optical z) must be finite and within sensor range.
        valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
        valid &= (zs > 0.0) & (zs < 8.0)
        if not valid.any():
            self.pcl_corridor_bx = None
            self.pcl_nearest_dist = float("inf")
            self.pcl_blocked_front = False
            self.pcl_blocked_front_warn = False
            self.pcl_point_count = 0
            self.pcl_last_stamp = self.get_clock().now()
            return

        xs = xs[valid]
        ys = ys[valid]
        zs = zs[valid]

        # Optical-frame -> base_link (camera rpy 0 0 0 with optical
        # rpy (-pi/2, 0, -pi/2)):
        #   bx = +z_optical + cam_x
        #   by = -x_optical + cam_y
        #   bz = -y_optical + cam_z
        bx = zs + self.cam_x
        by = -xs + self.cam_y
        bz = -ys + self.cam_z

        # ── Stair / drop-off hazard detection (before height filter) ──
        # Uses the full bz range to see the floor level and detect when
        # the floor disappears (descent/drop-off) or a rising surface
        # appears at close range (ascent).
        if self.stair_detect_enabled:
            self._check_stair_hazards(bx, by, bz)

        # Height + forward filter first — everything else is a width check.
        m_ahead = (
            (bz >= self.pcl_z_min) & (bz <= self.pcl_z_max) &
            (bx > 0.0)
        )
        bx_ahead = bx[m_ahead]
        by_ahead = by[m_ahead]

        # Lateral safety corridor ahead of the robot (robot-width + margin).
        m_corridor = np.abs(by_ahead) <= self.safe_half_y
        corridor_bx = bx_ahead[m_corridor]
        corridor_by = by_ahead[m_corridor]

        # Tight dead-ahead corridor — used for the "any single point close
        # in dead-ahead" override.  Catches thin obstacles (chair legs,
        # table legs, poles) that may not accumulate enough points to pass
        # pcl_min_points in the wider corridor.
        m_center = np.abs(by_ahead) <= self.pcl_center_half_y
        center_bx = bx_ahead[m_center]

        # Warning band: wider and further out.
        m_warn = (np.abs(by_ahead) <= self.warn_half_y) & (bx_ahead <= self.warn_half_x)

        if corridor_bx.size == 0:
            self.pcl_corridor_bx = None
            self.pcl_corridor_by = None
            self.pcl_nearest_dist = float("inf")
            self.pcl_blocked_front = False
            self.pcl_point_count = 0
        else:
            self.pcl_corridor_bx = corridor_bx
            self.pcl_corridor_by = corridor_by
            self.pcl_nearest_dist = float(corridor_bx.min())
            safe_count = int(np.count_nonzero(corridor_bx <= self.safe_half_x))
            self.pcl_point_count = safe_count
            self.pcl_blocked_front = safe_count >= self.pcl_min_points

        self.pcl_center_nearest = float(center_bx.min()) if center_bx.size > 0 else float("inf")

        self.pcl_blocked_front_warn = (
            int(np.count_nonzero(m_warn)) >= self.pcl_min_points
        )
        self.pcl_last_stamp = self.get_clock().now()

    def _check_stair_hazards(self, bx: np.ndarray, by: np.ndarray, bz: np.ndarray):
        """
        Analyse the full-range point cloud for stair descent / ascent hazards.
        Uses temporal confirmation (N consecutive frames) to suppress false positives.

        Stair DESCENT / drop-off: the floor (low bz band) disappears ahead
        of the robot, meaning the ground level dropped away.

        Stair ASCENT: a dense cluster of mid-height points at very close range
        forms a riser face — the ground rises steeply.
        """
        lateral = np.abs(by) <= (self.robot_half_y + 0.15)
        look_zone = (bx >= self._stair_look_min) & (bx <= self._stair_look_max)

        # ── Descent / drop-off ──────────────────────────────────────
        floor_band = (bz >= self._stair_floor_bz_min) & (bz <= self._stair_floor_bz_max)
        floor_pts  = int(np.count_nonzero(lateral & look_zone & floor_band))
        # Only trigger if sensor is returning data in the look zone at all
        any_in_zone = int(np.count_nonzero(lateral & look_zone))
        descent_candidate = (any_in_zone >= 3) and (floor_pts < self._stair_min_floor_pts)

        if descent_candidate:
            self._stair_descent_frames += 1
        else:
            self._stair_descent_frames = max(0, self._stair_descent_frames - 1)

        prev_descent = self.stair_descent_warning
        self.stair_descent_warning = self._stair_descent_frames >= self._STAIR_CONFIRM_FRAMES

        if self.stair_descent_warning and not prev_descent:
            self.get_logger().warn(
                f"STAIR/DROP-OFF DETECTED  |  floor_pts={floor_pts}  "
                f"any_in_zone={any_in_zone}  |  blocking forward motion"
            )
        elif not self.stair_descent_warning and prev_descent:
            self.get_logger().info("Stair/drop-off warning cleared")

        # ── Ascent ──────────────────────────────────────────────────
        close_zone = (bx >= 0.15) & (bx <= self._stair_ascent_dist)
        riser_band = (bz >= self._stair_ascent_bz_min) & (bz <= self._stair_ascent_bz_max)
        riser_pts  = int(np.count_nonzero(lateral & close_zone & riser_band))
        ascent_candidate = riser_pts >= self._stair_ascent_pts

        if ascent_candidate:
            self._stair_ascent_frames += 1
        else:
            self._stair_ascent_frames = max(0, self._stair_ascent_frames - 1)

        prev_ascent = self.stair_ascent_warning
        self.stair_ascent_warning = self._stair_ascent_frames >= self._STAIR_CONFIRM_FRAMES

        if self.stair_ascent_warning and not prev_ascent:
            self.get_logger().warn(
                f"STAIR ASCENT DETECTED  |  riser_pts={riser_pts}  |  blocking forward motion"
            )
        elif not self.stair_ascent_warning and prev_ascent:
            self.get_logger().info("Stair ascent warning cleared")

    def _cmd_cb(self, msg: Twist):
        """
        Receive a velocity command, filter it for safety, and republish.
        This is the VETO gate – if the robot would collide, we zero/reduce
        the command in the dangerous direction.
        """
        out = Twist()
        out.linear  = msg.linear
        out.angular = msg.angular

        vetoed = False

        # ── PCL freshness / fail-safe ───────────────────────────────
        # If the cloud stream has been seen but then stops updating,
        # treat the front as blocked so we don't coast into something
        # the lidar can't see (low tables, overhangs).
        pcl_stale = False
        if self.pcl_enabled and self.pcl_last_stamp is not None:
            age_s = (
                self.get_clock().now() - self.pcl_last_stamp
            ).nanoseconds * 1e-9
            pcl_stale = age_s > self.pcl_stale_timeout

        if pcl_stale and not self.pcl_stale_warned:
            self.pcl_stale_warned = True
            self.get_logger().warn(
                "PointCloud stale — blocking forward motion as fail-safe"
            )
        elif not pcl_stale and self.pcl_stale_warned:
            self.pcl_stale_warned = False
            self.get_logger().info("PointCloud stream restored")

        # ── Speed-aware forward safety distance ─────────────────────
        # We need room to decelerate.  Safe distance grows with the
        # commanded forward speed: d = v*t_react + v^2 / (2*a).
        v_cmd = max(0.0, msg.linear.x)
        brake_dist = (v_cmd * v_cmd) / (2.0 * self.brake_decel) + v_cmd * self.reaction_time
        dynamic_safe_x = self.safe_half_x + brake_dist

        # Re-check PCL against the dynamic zone using stored distances.
        pcl_dynamic_block = False
        if self.pcl_corridor_bx is not None and self.pcl_corridor_bx.size > 0:
            count_dyn = int(np.count_nonzero(self.pcl_corridor_bx <= dynamic_safe_x))
            pcl_dynamic_block = count_dyn >= self.pcl_min_points

        # Tight dead-ahead single-point override: a thin obstacle (chair
        # leg, table leg) may only produce a handful of depth pixels —
        # not enough for pcl_min_points in the wider corridor.  If ANY
        # point is close enough dead-ahead, block immediately.
        pcl_center_block = self.pcl_center_nearest <= max(
            self.pcl_center_trip, dynamic_safe_x
        )

        # Merge stair hazard warnings into the front-block flags.
        stair_hazard = self.stair_descent_warning or self.stair_ascent_warning
        if stair_hazard and msg.linear.x > 0.0:
            self.blocked_front = True
            self.obstacle_in_safety = True

        # Merge 3D point-cloud front-block into the standard flags.
        if self.pcl_blocked_front or pcl_dynamic_block or pcl_center_block or pcl_stale:
            self.blocked_front = True
            self.obstacle_in_safety = True
            cand = min(self.pcl_nearest_dist, self.pcl_center_nearest)
            if cand < self.nearest_dist:
                self.nearest_dist = cand
                self.nearest_angle = 0.0
        elif self.pcl_blocked_front_warn:
            self.obstacle_in_warning = True

        # Lidar-only speed-aware: if lidar sees something close in front
        # that is inside the braking zone, block forward even if it's
        # slightly outside the static safe rectangle.  Restrict to the
        # forward ±60° cone so rear/side hits don't trip this.
        if (not self.blocked_front) and self.nearest_dist <= dynamic_safe_x:
            if -math.pi / 3 < self.nearest_angle < math.pi / 3:
                self.blocked_front = True
                self.obstacle_in_safety = True

        # ── Startup sensor-presence fail-safe ───────────────────────
        # Refuse motion until at least one sensor has reported in.
        no_sensor_yet = (
            self.require_sensor
            and not self.scan_received
            and self.pcl_last_stamp is None
        )
        if no_sensor_yet:
            boot_age = (
                self.get_clock().now() - self.node_start_time
            ).nanoseconds * 1e-9
            if boot_age > 0.5:  # tiny grace for the first scan to land
                # Fail-safe: stop everything.
                out.linear.x = 0.0
                out.linear.y = 0.0
                out.angular.z = 0.0
                if int(boot_age) % 2 == 0:
                    self.get_logger().warn(
                        "No scan or point-cloud received yet — holding robot"
                    )
                self.cmd_pub.publish(out)
                return

        # ── Emergency close-range stop ──────────────────────────────
        # If anything is very close in front, we don't trust any forward
        # or rotational motion — the robot could swing into it.
        close_nearest = min(self.nearest_dist, self.pcl_center_nearest)
        if self.pcl_corridor_bx is not None and self.pcl_corridor_bx.size > 0:
            close_nearest = min(close_nearest, float(self.pcl_corridor_bx.min()))
        emergency = close_nearest <= self.emergency_stop_dist

        if emergency:
            out.linear.x = 0.0
            out.linear.y = 0.0
            # Allow in-place rotation only if it moves us AWAY from the
            # obstacle side. Simpler + safer: just freeze.
            out.angular.z = 0.0
            vetoed = True
            self.obstacle_in_safety = True
            self.blocked_front = True

        if self.obstacle_in_safety and not emergency:
            # ── Linear veto ──────────────────────────────────────────
            # Block forward motion if obstacle in front.  Only attempt
            # reverse escape when the rear is also clear, PCL is fresh,
            # and the upstream command already wanted forward.
            if self.blocked_front and msg.linear.x > 0.0:
                if self.allow_reverse and not self.blocked_rear and not pcl_stale:
                    out.linear.x = 0.0 #float(self.escape_speed)
                else:
                    out.linear.x = 0.0
                vetoed = True

            # Block reverse motion if obstacle behind
            if self.blocked_rear and msg.linear.x < 0.0:
                out.linear.x = 0.0
                vetoed = True

            # Block lateral motion (if the robot ever uses it)
            if self.blocked_left and msg.linear.y > 0.0:
                out.linear.y = 0.0
                vetoed = True
            if self.blocked_right and msg.linear.y < 0.0:
                out.linear.y = 0.0
                vetoed = True

            # ── Angular veto ─────────────────────────────────────────
            # If obstacle is on the left, block counter-clockwise rotation
            # If obstacle is on the right, block clockwise rotation
            # This prevents the robot from swinging its body into the obstacle
            if self.blocked_left and msg.angular.z > 0.0:
                out.angular.z = 0.0
                vetoed = True
            if self.blocked_right and msg.angular.z < 0.0:
                out.angular.z = 0.0
                vetoed = True

            # If blocked on both front sides, block ALL rotation to be safe
            if self.blocked_front and self.blocked_left and self.blocked_right:
                out.angular.z = 0.0
                vetoed = True

        if vetoed:
            self.veto_count += 1
            blocked_dirs = []
            if self.blocked_front: blocked_dirs.append("FRONT")
            if self.blocked_rear:  blocked_dirs.append("REAR")
            if self.blocked_left:  blocked_dirs.append("LEFT")
            if self.blocked_right: blocked_dirs.append("RIGHT")

            if self.stair_descent_warning:
                src = "STAIR-DESCENT"
            elif self.stair_ascent_warning:
                src = "STAIR-ASCENT"
            elif self.pcl_blocked_front:
                src = "PCL"
            else:
                src = "LIDAR"
            self.get_logger().warn(
                f"COLLISION VETO #{self.veto_count}  |  src={src}  "
                f"blocked: [{', '.join(blocked_dirs)}]  |  "
                f"nearest obstacle: {self.nearest_dist:.2f} m @ {math.degrees(self.nearest_angle):.0f} deg  |  "
                f"cmd_in: lin={msg.linear.x:.2f} ang={msg.angular.z:.2f}  →  "
                f"cmd_out: lin={out.linear.x:.2f} ang={out.angular.z:.2f}"
            )
        elif self.obstacle_in_warning:
            self.get_logger().info(
                f"WARNING ZONE  |  nearest obstacle: {self.nearest_dist:.2f} m @ "
                f"{math.degrees(self.nearest_angle):.0f} deg  |  passing command through"
            )

        self.cmd_pub.publish(out)

    # ────────────────────────────────────────────────────────────────
    # Health / diagnostics
    # ────────────────────────────────────────────────────────────────
    def _health_log(self):
        now = self.get_clock().now()
        if self.pcl_last_stamp is not None:
            pcl_age = (now - self.pcl_last_stamp).nanoseconds * 1e-9
            pcl_state = f"{pcl_age:.2f}s ago"
        elif self.pcl_enabled:
            pcl_state = "NEVER RECEIVED"
        else:
            pcl_state = "disabled"

        lidar_state = (
            f"nearest {self.nearest_dist:.2f}m" if self.scan_received
            else "NEVER RECEIVED"
        )

        stair_str = ""
        if self.stair_detect_enabled:
            stair_str = (
                f"  |  stair_descent={'WARN' if self.stair_descent_warning else 'ok'}"
                f"  stair_ascent={'WARN' if self.stair_ascent_warning else 'ok'}"
            )
        self.get_logger().info(
            f"[health] lidar: {lidar_state}  |  pcl: {pcl_state}  |  "
            f"pcl_nearest_corridor={self.pcl_nearest_dist:.2f}m  "
            f"pcl_nearest_center={self.pcl_center_nearest:.2f}m  "
            f"vetoes={self.veto_count}{stair_str}"
        )

        if self.pcl_enabled and self.pcl_last_stamp is None:
            boot_age = (now - self.node_start_time).nanoseconds * 1e-9
            if boot_age > 3.0:
                self.get_logger().error(
                    "Point-cloud enabled but NO data received — check the "
                    f"'{self.get_parameter('pcl_topic').value}' topic / QoS / bridge"
                )

    # ────────────────────────────────────────────────────────────────
    # RViz2 Visualization
    # ────────────────────────────────────────────────────────────────

    def _make_polygon_msg(self, half_x: float, half_y: float) -> PolygonStamped:
        ps = PolygonStamped()
        ps.header = Header()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.base_frame
        corners = _rect_polygon(half_x, half_y)
        for cx, cy in corners:
            pt = Point32()
            pt.x = float(cx)
            pt.y = float(cy)
            pt.z = 0.0
            ps.polygon.points.append(pt)
        return ps

    def _make_zone_marker(
        self,
        marker_id: int,
        half_x: float,
        half_y: float,
        color: ColorRGBA,
        z_offset: float = 0.01,
    ) -> Marker:
        """Create a LINE_STRIP marker tracing the rectangle in RViz2."""
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.base_frame
        m.ns = "safety_guard"
        m.id = marker_id
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.03  # line width
        m.color = color
        m.pose.orientation.w = 1.0
        m.lifetime.sec = 0
        m.lifetime.nanosec = 500_000_000  # 0.5 s

        corners = _rect_polygon(half_x, half_y)
        corners.append(corners[0])  # close the loop
        for cx, cy in corners:
            p = Point()
            p.x = float(cx)
            p.y = float(cy)
            p.z = z_offset
            m.points.append(p)

        return m

    def _make_filled_zone_marker(
        self,
        marker_id: int,
        half_x: float,
        half_y: float,
        color: ColorRGBA,
        z_offset: float = 0.005,
    ) -> Marker:
        """Create a semi-transparent filled rectangle using TRIANGLE_LIST."""
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.base_frame
        m.ns = "safety_guard_fill"
        m.id = marker_id
        m.type = Marker.TRIANGLE_LIST
        m.action = Marker.ADD
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0
        m.color = color
        m.pose.orientation.w = 1.0
        m.lifetime.sec = 0
        m.lifetime.nanosec = 500_000_000

        # Two triangles to fill the rectangle
        corners = _rect_polygon(half_x, half_y)
        tris = [
            (corners[0], corners[1], corners[2]),
            (corners[0], corners[2], corners[3]),
        ]
        for tri in tris:
            for cx, cy in tri:
                p = Point()
                p.x = float(cx)
                p.y = float(cy)
                p.z = z_offset
                m.points.append(p)

        return m

    def _make_nearest_arrow(self) -> Marker:
        """Arrow pointing from robot centre toward nearest obstacle."""
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.base_frame
        m.ns = "safety_guard"
        m.id = 100
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale.x = 0.04  # shaft diameter
        m.scale.y = 0.08  # head diameter
        m.scale.z = 0.06  # head length
        m.pose.orientation.w = 1.0
        m.lifetime.sec = 0
        m.lifetime.nanosec = 500_000_000

        if self.nearest_dist < float("inf"):
            start = Point(x=0.0, y=0.0, z=0.15)
            end = Point(
                x=self.nearest_dist * math.cos(self.nearest_angle),
                y=self.nearest_dist * math.sin(self.nearest_angle),
                z=0.15,
            )
            m.points = [start, end]

            if self.obstacle_in_safety:
                m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            elif self.obstacle_in_warning:
                m.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0)
            else:
                m.color = ColorRGBA(r=0.2, g=1.0, b=0.2, a=0.5)
        else:
            m.action = Marker.DELETE

        return m

    def _make_status_text(self) -> Marker:
        """Floating text above the robot showing safety status."""
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.base_frame
        m.ns = "safety_guard_text"
        m.id = 200
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.70
        m.pose.orientation.w = 1.0
        m.scale.z = 0.12  # text height
        m.lifetime.sec = 0
        m.lifetime.nanosec = 500_000_000

        if self.obstacle_in_safety:
            blocked = []
            if self.blocked_front: blocked.append("F")
            if self.blocked_rear:  blocked.append("B")
            if self.blocked_left:  blocked.append("L")
            if self.blocked_right: blocked.append("R")
            m.text = f"BLOCKED [{','.join(blocked)}] {self.nearest_dist:.2f}m"
            m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        elif self.obstacle_in_warning:
            m.text = f"CAUTION {self.nearest_dist:.2f}m"
            m.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0)
        else:
            m.text = "CLEAR"
            m.color = ColorRGBA(r=0.2, g=1.0, b=0.2, a=1.0)

        return m

    def _publish_markers(self):
        """Periodic timer callback to publish all RViz2 markers."""
        now = self.get_clock().now().to_msg()
        markers = MarkerArray()

        # Choose colours based on state
        if self.obstacle_in_safety:
            safety_line = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            safety_fill = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.15)
        elif self.obstacle_in_warning:
            safety_line = ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0)
            safety_fill = ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.08)
        else:
            safety_line = ColorRGBA(r=0.2, g=1.0, b=0.2, a=0.8)
            safety_fill = ColorRGBA(r=0.2, g=1.0, b=0.2, a=0.05)

        warning_line = ColorRGBA(r=1.0, g=0.65, b=0.0, a=0.5)
        footprint_line = ColorRGBA(r=0.3, g=0.6, b=1.0, a=1.0)
        footprint_fill = ColorRGBA(r=0.3, g=0.6, b=1.0, a=0.10)

        # 1. Robot footprint (blue)
        markers.markers.append(
            self._make_zone_marker(0, self.robot_half_x, self.robot_half_y, footprint_line, 0.02)
        )
        markers.markers.append(
            self._make_filled_zone_marker(0, self.robot_half_x, self.robot_half_y, footprint_fill, 0.005)
        )

        # 2. Safety zone (green/yellow/red depending on state)
        markers.markers.append(
            self._make_zone_marker(1, self.safe_half_x, self.safe_half_y, safety_line, 0.015)
        )
        markers.markers.append(
            self._make_filled_zone_marker(1, self.safe_half_x, self.safe_half_y, safety_fill, 0.004)
        )

        # 3. Warning zone (orange outline only)
        markers.markers.append(
            self._make_zone_marker(2, self.warn_half_x, self.warn_half_y, warning_line, 0.01)
        )

        # 4. Arrow to nearest obstacle
        markers.markers.append(self._make_nearest_arrow())

        # 5. Status text
        markers.markers.append(self._make_status_text())

        self.marker_pub.publish(markers)

        # Also publish the polygon topics (useful for Nav2 costmap footprint)
        self.footprint_pub.publish(
            self._make_polygon_msg(self.robot_half_x, self.robot_half_y)
        )
        self.safety_zone_pub.publish(
            self._make_polygon_msg(self.safe_half_x, self.safe_half_y)
        )
        self.warning_zone_pub.publish(
            self._make_polygon_msg(self.warn_half_x, self.warn_half_y)
        )


# ────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = SafetyGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"SafetyGuard shutting down  |  total vetoes: {node.veto_count}"
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
