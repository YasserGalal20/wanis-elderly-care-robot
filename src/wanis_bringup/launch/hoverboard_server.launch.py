
import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():

    # ── Launch arguments ──
    declare_rviz = DeclareLaunchArgument(
        "rviz", default_value="false", description="Launch RViz2"
    )
    declare_world = DeclareLaunchArgument(
        "world",
        default_value=PathJoinSubstitution([
            FindPackageShare("hoverboard_demo_bringup"), "worlds", "obstacles.world"
        ]),
        description="Path to Gazebo world SDF file",
    )
    declare_slam = DeclareLaunchArgument(
        "slam", default_value="false", description="Launch SLAM Toolbox"
    )
    declare_nav2 = DeclareLaunchArgument(
        "nav2", default_value="false", description="Launch Nav2 stack"
    )

    use_rviz = LaunchConfiguration("rviz")
    world_path = LaunchConfiguration("world")
    use_slam = LaunchConfiguration("slam")
    use_nav2 = LaunchConfiguration("nav2")

    bringup_share = get_package_share_directory("hoverboard_demo_bringup")
    description_share = get_package_share_directory("hoverboard_demo_description")

    # ── Robot description (sim variant) ──
    # robot_description_content = Command([
    #     PathJoinSubstitution([FindExecutable(name="xacro")]),
    #     " ",
    #     PathJoinSubstitution([
    #         FindPackageShare("hoverboard_demo_description"),
    #         "urdf", "hoverboard_description_sim.xacro",
    #     ]),
    # ])
    # robot_description = {"robot_description": robot_description_content}

    # ── Robot description (sim variant) ──
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("hoverboard_demo_description"),
            "urdf", "hoverboard_description_sim.xacro",
        ]),
    ])

    # WRAP IT HERE:
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # ── Gazebo Harmonic ──
    # The native wanis_4x4 SDF model embeds a gz_ros2_control plugin whose
    # <parameters> tag must point at an absolute YAML path — rcl does NOT
    # resolve package:// or $(find pkg) when gz_ros2_control forwards the
    # string through --params-file.  Workaround: write a processed copy
    # of the model to a temp dir with the placeholder substituted, and
    # prepend that dir to GZ_SIM_RESOURCE_PATH so Gazebo finds our copy
    # before the one installed under description/models.
    import shutil, tempfile
    _source_model_dir = os.path.join(description_share, "models", "wanis_4x4")
    _temp_models_root = os.path.join(tempfile.gettempdir(), "wanis_gazebo_models")
    _temp_model_dir = os.path.join(_temp_models_root, "wanis_4x4")
    os.makedirs(_temp_model_dir, exist_ok=True)

    _controllers_yaml = os.path.join(
        bringup_share, "config", "hoverboard_controllers_sim.yaml"
    )
    with open(os.path.join(_source_model_dir, "model.sdf"), "r") as _f:
        _sdf_text = _f.read()
    _sdf_text = _sdf_text.replace("@BRINGUP_CONTROLLERS_YAML@", _controllers_yaml)
    with open(os.path.join(_temp_model_dir, "model.sdf"), "w") as _f:
        _f.write(_sdf_text)
    shutil.copy(
        os.path.join(_source_model_dir, "model.config"),
        os.path.join(_temp_model_dir, "model.config"),
    )

    # Search order: temp (processed) → installed models dir → legacy share.
    # Keep whatever was in the user's shell env at the end so Fuel caches
    # and system model dirs still work.
    _existing_gz_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    _gz_path_parts = [
        _temp_models_root,
        os.path.join(description_share, "models"),
        os.path.join(description_share, ".."),
    ]
    if _existing_gz_path:
        _gz_path_parts.append(_existing_gz_path)
    gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=":".join(_gz_path_parts),
    )

    # GUI config adds the VisualizePointCloud plugin so /kinect/points
    # can be rendered inside Gazebo's 3D view.
    gui_config_path = os.path.join(bringup_share, "config", "gui.config")

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"
            ])
        ]),
        launch_arguments={
            "gz_args": ["-r -v 4 --gui-config ", gui_config_path, " ", world_path],
            "on_exit_shutdown": "true",
        }.items(),
    )

    # NOTE: URDF-spawn disabled — robot is now a native <model> inside
    # obstacles.world (see <include> block there).  URDF-spawn broke
    # gz-gui scene-graph lookups for sensors so visualisations snapped
    # to the world origin; native SDF fixes that.  Kept here (commented)
    # for easy rollback if the SDF path ever needs debugging.
    # gz_spawn = Node(
    #     package="ros_gz_sim",
    #     executable="create",
    #     arguments=[
    #         "-name", "wanis_4x4",
    #         "-topic", "robot_description",
    #         "-x", "0.0",
    #         "-y", "1.0",
    #         "-z", "0.15",
    #     ],
    #     output="screen",
    # )

    # ── Point-cloud colorizer ──
    # gz-gui's PointCloud panel renders nothing when no Float_V topic matches
    # the cloud size (see scripts/pointcloud_colorizer.py for the gory
    # detail).  This helper republishes /kinect/points as a constant Float_V
    # on /kinect/points_colors so the plugin actually draws the points.
    pointcloud_colorizer = Node(
        package="hoverboard_demo_bringup",
        executable="pointcloud_colorizer.py",
        name="pointcloud_colorizer",
        output="screen",
    )

    # ── ros_gz_bridge — bridges Gazebo topics to ROS2 ──
    # Remap rgbd_camera outputs so sim exposes the same topics as the
    # real Kinect v1 driver: /kinect/rgb/image_raw, /kinect/depth/image_raw,
    # /kinect/points, /kinect/depth/camera_info.
    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{
            "config_file": os.path.join(bringup_share, "config", "gz_bridge.yaml"),
            "use_sim_time": True,
        }],
        remappings=[
            # Kinect V1 → lite-mode topics
            ("kinect/image",        "/kinect/rgb/image_raw"),
            ("kinect/depth_image",  "/kinect/depth/image_raw"),
            ("kinect/points",       "/kinect/points"),
            ("kinect/camera_info",  "/kinect/depth/camera_info"),
            # Kinect V2 → heavy-mode topics (matches real kinect2_bridge QHD output)
            ("kinect2/image",       "/kinect2/qhd/image_color_rect"),
            ("kinect2/depth_image", "/kinect2/qhd/image_depth_rect"),
            ("kinect2/points",      "/kinect2/points"),
            ("kinect2/camera_info", "/kinect2/qhd/camera_info"),
        ],
        output="screen",
    )

    # ── Robot state publisher ──
    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    # ── Controller spawners (spawned after Gazebo loads the robot) ──
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "-c", "/controller_manager",
            "--controller-manager-timeout", "60",
        ],
        parameters=[{"use_sim_time": True}],
    )

    robot_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "hoverboard_base_controller",
            "-c", "/controller_manager",
            "--controller-manager-timeout", "60",
        ],
        parameters=[{"use_sim_time": True}],
    )

    # ── Twist stamper (cmd_vel → stamped for diff_drive_controller) ──
    twist_stamper = Node(
        package="twist_stamper",
        executable="twist_stamper",
        parameters=[{"use_sim_time": True}],
        remappings=[
            ("cmd_vel_in", "cmd_vel"),
            ("cmd_vel_out", "hoverboard_base_controller/cmd_vel"),
        ],
    )

    # ── EKF (sim version with use_sim_time) ──
    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[
            os.path.join(bringup_share, "config", "ekf_sim.yaml"),
            {"use_sim_time": True},
        ],
    )

    # ── SLAM Toolbox ──
    slam_node = Node(
        package="slam_toolbox",
        executable="sync_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            os.path.join(bringup_share, "config", "mapper_params_online_sync.yaml"),
            {"use_sim_time": True},
        ],
        condition=IfCondition(use_slam),
    )

    # ── Nav2 ──
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("nav2_bringup"), "launch", "navigation_launch.py",
            ])
        ]),
        launch_arguments={
            "use_sim_time": "true",
            "params_file": os.path.join(bringup_share, "config", "nav2_params_changed.yaml"),
        }.items(),
        condition=IfCondition(use_nav2),
    )

    # ── Safety Guard ──
    safety_guard_node = Node(
        package="person_follower",
        executable="safety",
        name="safety_guard",
        output="screen",
    )

    # ── RViz2 ──
    rviz_config = PathJoinSubstitution([
        FindPackageShare("hoverboard_demo_description"), "rviz", "hoverboard.rviz",
    ])
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(use_rviz),
    )

    # ── Delayed spawns (wait for Gazebo to be ready) ──
    # Spawn controllers after joint_state_broadcaster is up
    delayed_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[robot_controller_spawner],
        )
    )

    # Delay EKF, SLAM, and twist_stamper to let controllers initialize
    delayed_nodes = TimerAction(
        period=5.0,
        actions=[
            twist_stamper,
            ekf_node,
            slam_node,
        ],
    )

    return LaunchDescription([
        # Arguments
        declare_rviz,
        declare_world,
        declare_slam,
        declare_nav2,

        # Environment
        gz_resource_path,

        # Gazebo
        gz_sim,

        # Robot description (drives TF only; physics/sensors come from SDF)
        robot_state_pub,

        # Spawn into Gazebo — disabled (robot is native SDF in the world).
        # Uncomment gz_spawn above and this line to revert to URDF-spawn.
        # gz_spawn,

        # Point-cloud colorizer (makes gz-gui's PointCloud panel actually render)
        # pointcloud_colorizer,

        # Bridge
        gz_bridge,

        # Controllers (joint_state_broadcaster first, then diff_drive after it's up)
        joint_state_broadcaster_spawner,
        delayed_controller_spawner,

        # Delayed nodes (EKF, SLAM, twist_stamper)
        delayed_nodes,

        # Nav2 (optional)
        nav2_launch,

        # Safety Guard
        safety_guard_node,

        # RViz (optional)
        rviz_node,
    ])