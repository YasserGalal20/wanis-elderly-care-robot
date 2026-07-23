"""
RTAB-Map SLAM launch — Wanis heavy mode.

Bandwidth-aware setup
---------------------
The RPi's kinect2_bridge already publishes JPEG-compressed color
(/kinect2/sd/image_color_rect/compressed) and PNG-compressed depth
(/kinect2/sd/image_depth_rect/compressedDepth).  rtabmap_slam itself
cannot consume compressed images (it uses image_transport::SubscriberFilter
hard-wired to "raw"), so when compressed:=true (default) we run two tiny
`image_transport republish` *decoder* nodes on the SERVER:

      RPi (kinect2_bridge)           wifi              Server
      ──────────────────────         ────              ───────────────────
      .../compressed             ─────────────►   kinect2_rgb_decoder
      .../compressedDepth        ─────────────►   kinect2_depth_decoder
                                                         │  (loopback)
                                                         ▼
                                                  .../image_color_rect_decoded
                                                  .../image_depth_rect_decoded
                                                         │
                                                         ▼
                                                       rtabmap

Only the compressed bytes traverse wifi.  Decoding happens once on the
server's CPU and the raw stream lives entirely on loopback.

Set compressed:=false to skip the decoders and have rtabmap subscribe to
the raw kinect2 topics directly (much heavier wifi load).

Inputs
------
  /kinect2/sd/image_color_rect[/compressed]      sensor_msgs/(Compressed)Image
  /kinect2/sd/image_depth_rect[/compressedDepth] sensor_msgs/(Compressed)Image
  /kinect2/sd/camera_info                        sensor_msgs/CameraInfo
  /scan                                          sensor_msgs/LaserScan
  /odometry/filtered                             nav_msgs/Odometry
  /tf, /tf_static

Outputs (under /rtabmap)
------------------------
  /rtabmap/info                rtabmap_msgs/Info       loop closures, timing
  /rtabmap/cloud_map           sensor_msgs/PointCloud2 3D map (TRANSIENT_LOCAL)
  /rtabmap/grid_map            nav_msgs/OccupancyGrid  2D map (TRANSIENT_LOCAL)
  /rtabmap/mapPath             nav_msgs/Path           graph nodes
  /rtabmap/localization_pose   geometry_msgs/PoseWithCovarianceStamped

The viz_bridge subscribes to /rtabmap/cloud_map, /rtabmap/grid_map and
/rtabmap/info — see flask_robot_ui/viz_bridge.py.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


# Where the decoded raw images get republished when compressed:=true.
# rtabmap subscribes here instead of the raw kinect2 topics.
_RGB_DECODED_TOPIC   = '/kinect2/sd/image_color_rect_decoded'
_DEPTH_DECODED_TOPIC = '/kinect2/sd/image_depth_rect_decoded'

# Raw (uncompressed) topics — used when compressed:=false.
_RGB_RAW_TOPIC   = '/kinect2/sd/image_color_rect'
_DEPTH_RAW_TOPIC = '/kinect2/sd/image_depth_rect'


def launch_setup(context, *args, **kwargs):
    use_sim_time       = LaunchConfiguration('use_sim_time').perform(context)
    delete_db_on_start = LaunchConfiguration('delete_db_on_start').perform(context)
    rtabmap_viz        = LaunchConfiguration('rtabmap_viz')
    localization       = LaunchConfiguration('localization').perform(context).lower() == 'true'
    use_compressed     = LaunchConfiguration('compressed').perform(context).lower() == 'false'

    rgb_topic   = _RGB_DECODED_TOPIC   if use_compressed else _RGB_RAW_TOPIC
    depth_topic = _DEPTH_DECODED_TOPIC if use_compressed else _DEPTH_RAW_TOPIC

    parameters = [{
        'use_sim_time':            use_sim_time.lower() == 'true',

        # ── Frames ───────────────────────────────────────────────
        'frame_id':                'base_footprint',
        'odom_frame_id':           'odom',
        'map_frame_id':            'map',
        'publish_tf':              True,           # map → odom

        # ── Subscriptions ────────────────────────────────────────
        'subscribe_rgb':           True,
        'subscribe_depth':         True,
        'subscribe_rgbd':          False,
        'subscribe_scan':          True,           # /scan from RPLiDAR
        'subscribe_scan_cloud':    False,
        'subscribe_odom_info':     False,          # external EKF, no rtabmap visual odom

        # ── Sync ─────────────────────────────────────────────────
        'approx_sync':             True,
        'approx_sync_max_interval': 0.1,
        'queue_size':              10,
        'topic_queue_size':        10,
        'sync_queue_size':         10,

        # ── QoS (1 = RELIABLE — matches kinect2_bridge & robot_localization) ─
        'qos_image':               1,
        'qos_camera_info':         1,
        'qos_scan':                1,
        'qos_odom':                1,
        'qos_user_data':           1,
        'qos_imu':                 1,
        'qos_gps':                 1,

        # ── Map output (occupancy grid + 3D cloud) ──────────────
        'Grid/FromDepth':              'true',
        'Grid/CellSize':               '0.05',
        'Grid/RangeMax':               '4.0',
        'Grid/3D':                     'true',
        'Grid/RayTracing':             'true',
        'Grid/NormalsSegmentation':    'false',
        'Grid/MaxObstacleHeight':      '1.5',
        'Grid/MaxGroundHeight':        '0.05',
        'cloud_max_depth':             4.0,
        'cloud_voxel_size':            0.05,

        # ── Add nodes to the graph more aggressively so cloud_map fills
        #    in even when the robot moves slowly. rtabmap_slam only
        #    publishes /rtabmap/cloud_map when the graph updates, which is
        #    gated by these thresholds (defaults 0.1 m / 0.1 rad).
        'RGBD/LinearUpdate':       '0.03',
        'RGBD/AngularUpdate':      '0.03',
        'Rtabmap/DetectionRate':   '2.0',

        # ── 2D wheeled-robot regime ─────────────────────────────
        'Reg/Force3DoF':           'true',
        'Optimizer/Slam2D':        'true',

        # ── Memory / mapping mode ───────────────────────────────
        'Mem/IncrementalMemory':   'false' if localization else 'true',
        'Mem/InitWMWithAllNodes':  'true'  if localization else 'false',

        # ── Publish rate of the cloud_map / grid_map ────────────
        'map_always_update':       True,
        'map_empty_ray_tracing':   True,

        # ── DB ──────────────────────────────────────────────────
        'database_path':           '~/.ros/rtabmap.db',
    }]

    remappings = [
        ('rgb/image',       rgb_topic),
        ('depth/image',     depth_topic),
        ('rgb/camera_info', '/kinect2/sd/camera_info'),
        ('scan',            '/scan'),
        ('odom',            '/odometry/filtered'),
    ]

    rtabmap_args = ['-d'] if delete_db_on_start.lower() == 'true' else []

    nodes = []

    # ── Compressed → raw decoders (server-side, only when compressed:=true) ──
    # `image_transport republish` takes (in_transport, out_transport).  RGB
    # comes in as JPEG ('compressed').  Depth comes in as a ROS-specific
    # PNG-with-encoding-header ('compressedDepth').  Both decoder libs ship
    # with image_transport_plugins (already a dependency of rtabmap).
    if use_compressed:
        nodes.append(Node(
            package='image_transport', executable='republish',
            name='kinect2_rgb_decoder',
            output='screen', emulate_tty=True,
            arguments=['compressed', 'raw'],
            remappings=[
                ('in/compressed', '/kinect2/sd/image_color_rect/compressed'),
                ('out',           _RGB_DECODED_TOPIC),
            ],
        ))
        nodes.append(Node(
            package='image_transport', executable='republish',
            name='kinect2_depth_decoder',
            output='screen', emulate_tty=True,
            arguments=['compressedDepth', 'raw'],
            remappings=[
                ('in/compressedDepth', '/kinect2/sd/image_depth_rect/compressedDepth'),
                ('out',                _DEPTH_DECODED_TOPIC),
            ],
        ))

    nodes.append(Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        namespace='rtabmap',
        output='screen',
        emulate_tty=True,
        parameters=parameters,
        remappings=remappings,
        arguments=rtabmap_args,
    ))

    # ── map_assembler: rebuild the 3D point cloud from /rtabmap/mapData
    #    at a steady rate. rtabmap_slam itself only republishes
    #    /rtabmap/cloud_map when the graph updates (after ~3 cm of motion
    #    with the threshold above), which is fine for an actively driven
    #    robot but leaves the web 3D viewer empty when the robot is idle.
    #    map_assembler re-derives cloud_map every time mapData arrives and
    #    publishes it with TRANSIENT_LOCAL durability so a late subscriber
    #    (viz_bridge restart) gets the latest cached cloud immediately.
    nodes.append(Node(
        package='rtabmap_util',
        executable='map_assembler',
        name='map_assembler',
        namespace='rtabmap',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'use_sim_time':       use_sim_time.lower() == 'true',
            'cloud_voxel_size':   0.05,
            'cloud_decimation':   4,
            'cloud_max_depth':    4.0,
            'cloud_min_depth':    0.0,
            'regenerate_cloud':   True,
            'qos':                1,
        }],
    ))

    nodes.append(Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        name='rtabmap_viz',
        namespace='rtabmap',
        output='screen',
        emulate_tty=True,
        parameters=parameters,
        remappings=remappings,
        condition=IfCondition(rtabmap_viz),
    ))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time',       default_value='false'),
        DeclareLaunchArgument('rtabmap_viz',        default_value='true',
                              description='Open the standalone rtabmap_viz GUI.'),
        DeclareLaunchArgument('delete_db_on_start', default_value='false',
                              description='Pass -d to rtabmap to wipe ~/.ros/rtabmap.db on launch.'),
        DeclareLaunchArgument('localization',       default_value='false',
                              description='Localization-only mode (no new map nodes).'),
        DeclareLaunchArgument('compressed',         default_value='true',
                              description='When true, rtabmap consumes the compressed Kinect '
                                          'topics via small server-side decoder nodes (saves '
                                          'wifi bandwidth).  Set false to subscribe to raw.'),
        OpaqueFunction(function=launch_setup),
    ])
