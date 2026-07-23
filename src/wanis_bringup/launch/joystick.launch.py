
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    teleop_twist_joy_config_file = PathJoinSubstitution(
        [
            FindPackageShare("hoverboard_demo_bringup"), "config", "ps4.config.yaml",
        ]
    )

    joy_node = Node(
        package='joy', executable='joy_node', name='joy_node',
        parameters=[{
            'deadzone': 0.05,
            'autorepeat_rate': 30.0
        }],
    )

    teleop_twist_joy_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy_node',
        parameters=[teleop_twist_joy_config_file],
    )




    return LaunchDescription([
        joy_node,
        teleop_twist_joy_node,
    ])