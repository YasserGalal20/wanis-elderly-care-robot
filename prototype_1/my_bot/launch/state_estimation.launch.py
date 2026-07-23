import launch
from launch import LaunchDescription
from launch_ros.actions import Node
import rclpy
from rclpy import logging
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import python_launch_description_source
from ament_index_python import get_package_share_directory 
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import FindExecutable
from launch_ros.actions import Node 
import os
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    logging.get_logger("Y").info("Yasser IS A GENIUS ENGINE")


    ld = LaunchDescription()
    dev=False
    dev2=False

    robot_localization_dir = get_package_share_directory('robot_localization')
    parameters_file_dir = os.path.join(robot_localization_dir, 'params')
    parameters_file_path = os.path.join(parameters_file_dir, 'dual_ekf_navsat_example.yaml')
    
    
    ekf_fusion= Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[os.path.join(get_package_share_directory("robot_localization"), 'params', 'ekf.yaml')],
           )

    
    ukf_fusion= Node(
            package='robot_localization',
            executable='ukf_node',
            name='ukf_filter_node',
            output='screen',
            parameters=[os.path.join(get_package_share_directory("robot_localization"), 'params', 'ukf.yaml')],
    )

    ekf_gps_odom  = Node(
          package='robot_localization', 
            executable='ekf_node', 
            name='ekf_filter_node_odom',
	        output='screen',
            parameters=[parameters_file_path],
            remappings=[('odometry/filtered', 'odometry/local')]   ,
           )


    ekf_gps_map  = Node(
              package='robot_localization', 
            executable='ekf_node', 
            name='ekf_filter_node_map',
	        output='screen',
            parameters=[parameters_file_path],
            remappings=[('odometry/filtered', 'odometry/global')],
           )



    ekf_gps_navsat  = Node(
            package='robot_localization', 
            executable='navsat_transform_node', 
            name='navsat_transform',
	        output='screen',
            parameters=[parameters_file_path],
            remappings=[('imu/data', 'imu/data'),
                        ('gps/fix', 'gps/fix'), 
                        ('gps/filtered', 'gps/filtered'),
                        ('odometry/gps', 'odometry/gps'),
                        ('odometry/filtered', 'odometry/global')]  ,
           )




    print("\n\n-----For EKF enter ( E )----- (RELIABLE)\n-----For UKF enter ( U )----- (RELIABLE)")
    print("-----For EKF GPS with multiple sensors enter ( GE )----- (RELIABLE)\n")



    print("\nEnter your value: ")

    v = input()


    while True:

     if(v=="E"):
        ld.add_action(ekf_fusion)
        print("\n\nStarting....\n\n")

        
        break

     elif(v=="U"):
        ld.add_action(ukf_fusion)
        print("\n\nStarting....\n\n")


        break
     


     
     elif(v=="GE"):
         ld.add_action(ekf_gps_odom)
         ld.add_action(ekf_gps_map)
         ld.add_action(ekf_gps_navsat)
         print("\n\nStarting....\n\n")
         break
     


            


    return ld


