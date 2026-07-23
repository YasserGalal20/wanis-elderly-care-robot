"""
kinect2_pointcloud.launch.py

Runs server-side depth_image_proc to generate a PointCloud2 from Kinect v2
depth + RGB images published by kinect2_bridge on the RPi.

The RPi runs `ros2 run kinect2_bridge kinect2_bridge_node` (lite, no pointcloud).
This launch file runs on the SERVER and subscribes to:
  /kinect2/qhd/image_depth_rect        (16-bit depth, registered to color)
  /kinect2/qhd/image_color_rect        (rectified RGB)
  /kinect2/qhd/camera_info             (depth camera intrinsics)

Publishes:
  /kinect2/points                      (sensor_msgs/PointCloud2, XYZRGB)

This keeps Kinect v2 heavy processing off the RPi while safety_guard,
Nav2 costmap, and RTAB-Map on the server can all consume /kinect2/points.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")

    container = ComposableNodeContainer(
        name="kinect2_pointcloud_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="depth_image_proc",
                plugin="depth_image_proc::PointCloudXyzrgbNode",
                name="kinect2_pointcloud_node",
                remappings=[
                    # depth_image_proc::PointCloudXyzrgbNode default topic names →
                    # kinect2_bridge qhd topics
                    ("rgb/image_rect_color",       "/kinect2/qhd/image_color_rect"),
                    ("rgb/camera_info",            "/kinect2/qhd/camera_info"),
                    ("depth_registered/image_rect","/kinect2/qhd/image_depth_rect"),
                    ("points",                     "/kinect2/points"),
                ],
                parameters=[{
                    "use_sim_time": use_sim_time,
                    # ── QoS override: publish /kinect2/points BEST_EFFORT ──
                    # Default is RELIABLE which forces the publisher to wait
                    # for ACKs from every subscriber.  Multi-MB PointCloud2
                    # frames over Wi-Fi will lose packets often, and each
                    # loss stalls the whole topic until retransmission —
                    # which is why safety_guard on the RPi sees "NO data
                    # received" even though the topic exists.  BEST_EFFORT
                    # lets dropped frames be dropped and keeps the stream
                    # flowing.  Subscribers (safety_guard, Nav2 costmap)
                    # must also use BEST_EFFORT to match.
                    # "qos_overrides./kinect2/points.publisher.reliability": "best_effort",
                    # "qos_overrides./kinect2/points.publisher.history":     "keep_last",
                    # "qos_overrides./kinect2/points.publisher.depth":       5,
                    # "qos_overrides./kinect2/points.publisher.durability":  "volatile",
                }],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
        output="screen",
    )
    # 1. Define the command to set the parameter
    set_safety_guard_param = ExecuteProcess(
        cmd=['ros2', 'param', 'set', '/safety_guard', 'pcl_topic', '/kinect2/points'],
        output='screen'
    )

    # 2. Wrap it in a timer to give the system 2 seconds to ensure /safety_guard is running
    delayed_param_set = TimerAction(
        period=2.0,
        actions=[set_safety_guard_param]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation clock",
        ),
        container,
        #delayed_param_set, # 3. Add the delayed action to the launch description
    ])
