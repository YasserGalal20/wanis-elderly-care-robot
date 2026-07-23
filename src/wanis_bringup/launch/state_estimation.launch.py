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

    logging.get_logger("Y").info("Yasser IS A GENIUS ENGINE)")


    ld = LaunchDescription()
    dev=False
    dev2=False

    

    navsat= Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform_node',
            output='screen',
            parameters=[os.path.join(get_package_share_directory("hoverboard_demo_bringup"), 'config', 'navsat_transform.yaml')],
            remappings=[('imu', 'imu/data'),
                 ('gps/fix', 'gps/fix'), 
                 ('gps/filtered', 'gps/filtered'),
                 ('odometry/gps', 'odometry/gps'),
                 ('odometry/filtered', 'odometry/global')]
            )
    
    ekf_navsat = Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[os.path.join(get_package_share_directory("hoverboard_demo_bringup"), 'config', 'ekf_gps_nav.yaml')],
            remappings=[('odometry/filtered', 'odometry/global'),
                 ('/set_pose', '/initialpose')])

           
    
    ukf_navsat = Node(
            package='robot_localization',
            executable='ukf_node',
            name='ukf_filter_node',
            output='screen',
            parameters=[os.path.join(get_package_share_directory("hoverboard_demo_bringup"), 'config', 'ukf_gps_nav.yaml')],
            remappings=[('odometry/filtered', 'odometry/global'),
                 ('/set_pose', '/initialpose')])

           
    ekf_se  = Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[os.path.join(get_package_share_directory("hoverboard_demo_bringup"), 'config', 'ekf.yaml')],
           )



    ukf_se = Node(
            package='robot_localization',
            executable='ukf_node',
            name='ukf_filter_node',
            output='screen',
            parameters=[os.path.join(get_package_share_directory("hoverboard_demo_bringup"), 'config', 'ukf.yaml')],
           )





    
    note_about_gps_nodes="topic: /odometry/gps is the odometry with unfiltered gps only transformed from longitude and latitude to cartesian (not reliable)\ntopic: /odometry/global is the odometry with fused and filtered data with gps,imu,encoders\n\n"

    print("\n\n-----For EKF enter ( E )----- (RELIABLE)\n-----For UKF enter ( U )----- (RELIABLE)\n-----For Ekf with GPS and multiple sensores enter ( G )----- (UNRELIABLE --Testing purposes--)\n-----For EKf With GPS and one sensor enter ( T )----- (UNRELIABLE --Testing purposes--)")
    print("-----For EKF GPS with multiple sensors enter ( GE )----- (RELIABLE)\n-----For UKF GPS with multiple sensors enter ( GU )----- (RELIABLE)")



    print("\nEnter your value: ")

    v = input()


    while True:

     if(v=="E"):

        ld.add_action(ekf_se)
        print("\n\nStarting....\n\n")

        
        break

     elif(v=="U"):

        ld.add_action(ukf_se)

        print("\n\nStarting....\n\n")


        break
     


     
     elif(v=="GE"):
         ld.add_action(navsat)
         ld.add_action(ekf_navsat)

         print("\n\nStarting....\n\n")
         print(note_about_gps_nodes)

         break
     


     
     elif(v=="GU"):

         ld.add_action(navsat)
         ld.add_action(ukf_navsat)

         print("\n\nStarting....\n\n")
         print(note_about_gps_nodes)

         break


     else:
        if(v=="G" or v =="T"):     
            if(dev2==True):
              break
            print("GPS is unreliable yet for development purposes enter ( YN )\n")
            print("Enter your value: ")
            v = input()

            if(v=="YN"):   #for development by yasser 
               dev = True
               while dev :   
                print("\n--In development mode--\n")
                print("\n\n-----For Ekf with GPS and multiple sensores enter ( G )-----\n-----For EKf With GPS and one sensor enter ( T )-----\n")

                print("Enter your value: ")
                v = input()

                if (v=="Q"):
                   dev = False
                   print("Exited development mode\n")


                if(v == "T" and dev ==True):

                
                 start_navsat_transform_cmd = Node(
                 package='robot_localization',
                 executable='navsat_transform_node',
                 name='navsat_transform',
                 output='screen',
                 parameters=[os.path.join(get_package_share_directory("hoverboard_demo_bringup"), 'config', 'ekf_with_gps.yaml')],
                 remappings=[('imu', 'imu/data'),
                 ('gps', 'gps'), 
                 ('gps/filtered', 'gps/filtered'),
                 ('odometry/gps', 'odometry/gps'),
                 ('odometry/filtered', 'odometry/global')])
    


    
                 start_robot_localization_global_cmd = Node(
                 package='robot_localization',
                 executable='ekf_node',
                 name='ekf_filter_node_map',
                 output='screen',
                 parameters=[os.path.join(get_package_share_directory("hoverboard_demo_bringup"), 'config', 'ekf_with_gps.yaml')],
                 remappings=[('odometry/filtered', 'odometry/global'),
                 ('/set_pose', '/initialpose')])



                 start_robot_localization_local_cmd = Node(
                 package='robot_localization',
                 executable='ekf_node',
                 name='ekf_filter_node_odom',
                 output='screen',
                 parameters=[os.path.join(get_package_share_directory("hoverboard_demo_bringup"), 'config', 'ekf_with_gps.yaml')],
                 remappings=[('odometry/filtered', 'odometry/local'),
                 ('/set_pose', '/initialpose')])

                 ld.add_action(start_navsat_transform_cmd)
                 ld.add_action(start_robot_localization_global_cmd)
                 ld.add_action(start_robot_localization_local_cmd)
                 dev2 = True
                 print("\n\nStarting....\n\n")
                 break


                




                elif(v == "G" and dev ==True):      

                 
                 start_navsat_transform_cmd = Node(
                 package='robot_localization',
                 executable='navsat_transform_node',
                 name='navsat_transform',
                 output='screen',
                 parameters=[os.path.join(get_package_share_directory("hoverboard_demo_bringup"), 'config', 'ekf_with_multiple_sensors_and_gps.yaml')],
                 remappings=[('imu', 'imu/data'),
                 ('gps', 'gps'), 
                 ('gps/filtered', 'gps/filtered'),
                 ('odometry/gps', 'odometry/gps'),
                 ('odometry/filtered', 'odometry/global')])
    



                 start_robot_localization_global_cmd = Node(
                 package='robot_localization',
                 executable='ekf_node',
                 name='ekf_filter_node_map',
                 output='screen',
                 parameters=[os.path.join(get_package_share_directory("hoverboard_demo_bringup"), 'config', 'ekf_with_multiple_sensors_and_gps.yaml')],
                 remappings=[('odometry/filtered', 'odometry/global'),
                 ('/set_pose', '/initialpose')])



                 start_robot_localization_local_cmd = Node(
                 package='robot_localization',
                 executable='ekf_node',
                 name='ekf_filter_node_odom',
                 output='screen',
                 parameters=[os.path.join(get_package_share_directory("hoverboard_demo_bringup"), 'config', 'ekf_with_multiple_sensors_and_gps.yaml')],
                 remappings=[('odometry/filtered', 'odometry/local'),
                 ('/set_pose', '/initialpose')])

                 ld.add_action(start_navsat_transform_cmd)
                 ld.add_action(start_robot_localization_global_cmd)
                 ld.add_action(start_robot_localization_local_cmd)
                 dev2 = True
                 print("\n\nStarting....\n\n")

                 break


                else:
                 print("Invalid entry (If you want to exit development press Q ) \n")
                 

    




        else:
            print("Invalid entry\n")
            print("Enter your value: ")
            v = input()

            


    return ld


