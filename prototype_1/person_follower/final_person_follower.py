import os
#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
from sensor_msgs.msg import LaserScan
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import numpy as np
import rclpy
from rclpy.node import Node
import time
from geometry_msgs.msg import PoseWithCovarianceStamped
from slam_toolbox.srv import Reset
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy



class ResetClient(Node):
    def __init__(self):
        super().__init__('slam_toolbox_reset_client')
        self.client = self.create_client(Reset, '/slam_toolbox/reset')

    def send_request(self):
        # Wait for the service to become available
        while not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('Service not available, waiting...')
        
        request = Reset.Request()
        self.future = self.client.call_async(request)
        self.get_logger().info('Calling /slam_toolbox/reset service...')
        return self.future



class PersonFollower(Node):
    def __init__(self):
        super().__init__("person_follower")
        self.bridge = CvBridge()

        # Subscribe to the RGB image topic and depth image topic
        self.image_sub = self.create_subscription(Image, "/kinect/rgb/image_raw", self.callback, 10) #/kinect/topic for sim --yasser
        self.depth_sub = self.create_subscription(Image, "/kinect/depth/image_raw", self.depth_callback, 10)
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_pose_callback,
            10)
        # Publisher for robot velocity commands
        self.velocity_publisher = self.create_publisher(Twist, "/cmd_vel_smoothed", 10)
        self.last_time_detected = 0.0
        self.program_time = time.time()
        self.integral_error_yaw = 0.0
        self.integral_error_linear = 0.0
        self.angular_zeed = False
        self.angular_zeed_last = 0
        # Initialize variables
        self.last_error = 0.0
        self.last_depth = 0.0
        self.depth_image = None
        self.depth_mm = 0
        self.velocity_msg = Twist()
        self.velocity_msg = Twist()
        self.velocity_msg.linear.y = 0.0
        self.velocity_msg.linear.z = 0.0
        self.velocity_msg.angular.x = 0.0
        self.velocity_msg.angular.y = 0.0
        self.safety_margin = 3.0
        self.critical_margin = 2.5
        self.angular_seen = 0.0
        self.detected_while_search = False
        self.search_wait = 30
        # Initialize MediaPipe Pose module
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.initiated_search = False
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.kp_angular=0.005
        self.kp_linear=0.7
        self.x_center=None
        self.image_center=None
        self.buffer=10
        self.pose = 0
        # Laser scan
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.laser_callback, 10)
        self.goal = 0
        self.goal_accepted = False
        # Variable to track obstacle distance behind robot
        self.obstacle_behind = False
        self.min_safe_distance_back = 0.6  # meters

        # Create the options that will be used for ImageSegmenter
        base_options = python.BaseOptions(model_asset_path=os.path.expanduser('~/autonomus_bot/src/person_follower/person_follower/deeplabv3.tflite'))
        options = vision.ImageSegmenterOptions(base_options=base_options,output_category_mask=True)
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

        self.BG_COLOR = (192, 192, 192) # gray
        self.MASK_COLOR = (0, 255, 0) # white
        self.person_detected_before = False
        self.goal_reached = False
        self.logged_before_1 = False
        self.logged_before_2 =  False
        # Subscriber to the map topic
        map_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, 
            '/map', 
            self.map_callback, 
            map_qos_profile
        )

        self.person_detected = False
        self.final_navigation = False
        # Action client for navigation
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        # Visited frontiers set
        self.visited_frontiers = set()
        self.checked = 0

        # Map and position data
        self.map_data = None
        self.robot_position = (0, 0)  # Placeholder, update from localization

        # Timer for periodic exploration
        # self.timer = self.create_timer(5.0, self.explore)

    def map_callback(self, msg):
        self.map_data = msg
        self.get_logger().info("Map received")




    def amcl_pose_callback(self, msg):
        self.pose = msg.pose.pose
        self.get_logger().info(
            f"Current location: ({self.pose.position.x:.2f}, {self.pose.position.y:.2f})")


    def navigate_to(self, x, y):
        """
        Send navigation goal to Nav2.
        """
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.orientation.w = 1.0  # Facing forward
        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal_msg
        self.goal = nav_goal

        self.get_logger().info(f"Navigating to goal: x={x}, y={y}")

        # Wait for the action server
        self.nav_to_pose_client.wait_for_server()

        # Send the goal and register a callback for the result
        send_goal_future = self.nav_to_pose_client.send_goal_async(nav_goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Handle the goal response and attach a callback to the result.
        """
        self.goal = future.result()
        # print(type(self.goal),"AAAAAAAAAAAAAAAAAAAAAAAAAAA")
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warning("Goal rejected!")
            return

        self.get_logger().info("Goal accepted")
        self.goal_accepted = True
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_complete_callback)


    def cancel_done_callback(self, future):
        self.get_logger().info('Goal cancelled.')


    def navigation_complete_callback(self, future):
        """
        Callback to handle the result of the navigation action.
        """
        try:
            result = future.result().result
            self.get_logger().info(f"Navigation completed with result: {result}")
            self.goal_accepted = False
        except Exception as e:
            self.get_logger().error(f"Navigation failed: {e}")

    def find_frontiers(self, map_array):
        """
        Detect frontiers in the occupancy grid map.
        """
        frontiers = []
        rows, cols = map_array.shape

        # Iterate through each cell in the map
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if map_array[r, c] == 0:  # Free cell
                    # Check if any neighbors are unknown
                    neighbors = map_array[r-1:r+2, c-1:c+2].flatten()
                    if -1 in neighbors:
                        frontiers.append((r, c))

        self.get_logger().info(f"Found {len(frontiers)} frontiers")
        return frontiers

    def choose_frontier(self, frontiers):
        """
        Choose the closest frontier to the robot.
        """
        robot_row, robot_col = self.robot_position
        min_distance = float('inf')
        chosen_frontier = None
        

        for frontier in frontiers:
            if frontier in self.visited_frontiers:
                continue

            distance = np.sqrt((robot_row - frontier[0])**2 + (robot_col - frontier[1])**2)
            if distance < min_distance:
                min_distance = distance
                chosen_frontier = frontier

        if chosen_frontier:
            self.visited_frontiers.add(chosen_frontier)
            self.get_logger().info(f"Chosen frontier: {chosen_frontier}")
        else:
            self.get_logger().warning("No valid frontier found")
            self.checked +=1
            if self.checked >= 5:
                self.get_logger().warning("DID NOT FIND ANYONE")
                self.get_logger().info("Please call the police!")




        return chosen_frontier

    def explore(self):
        if self.map_data is None:
            self.get_logger().warning("No map data available")
            return

        # Convert map to numpy array
        map_array = np.array(self.map_data.data).reshape(
            (self.map_data.info.height, self.map_data.info.width))

        # Detect frontiers
        frontiers = self.find_frontiers(map_array)

        if not frontiers:
                self.get_logger().warning("DID NOT FIND ANYONE")
                self.get_logger().info("Please call the police!")




        #     # self.shutdown_robot()
        #     return

        # Choose the closest frontier
        chosen_frontier = self.choose_frontier(frontiers)

        if not chosen_frontier:
            self.get_logger().warning("No frontiers to explore")
            return

        # Convert the chosen frontier to world coordinates
        # if not self.origin_set:
        #     self.origin_x = self.map_data.info.origin.position.x
        #     self.origin_y = self.map_data.info.origin.position.y
        #     print ("SET ORIGIN")
        #     self.origin_set = True
        goal_x = chosen_frontier[1] * self.map_data.info.resolution + self.map_data.info.origin.position.x
        goal_y = chosen_frontier[0] * self.map_data.info.resolution + self.map_data.info.origin.position.y

        if self.person_detected:
            # print("PERSON WAS DETECTED")
            # send_goal_future = self.nav_to_pose_client._cancel_goal_async(self.goal)
            # send_goal_future.add_done_callback(self.goal_response_callback)
            cancel_future = self.nav_to_pose_client._cancel_goal_async(self.goal)
            cancel_future.add_done_callback(self.cancel_done_callback)


            
        else:
            self.navigate_to(goal_x, goal_y)
            



    def depth_callback(self, data):
        try:
            # Convert depth image to OpenCV format
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")


    def laser_callback(self, msg: LaserScan):
        try:
            # Default safety distance (meters)
            threshold = self.min_safe_distance_back


            # Convert to numpy for easier processing
            ranges = np.array(msg.ranges)
            valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max)
            ranges = np.where(valid_mask, ranges, np.inf)

            n = len(ranges)

            # Helper function to convert angle to array index
            def angle_to_index(angle_rad):
                idx = int((angle_rad - msg.angle_min) / msg.angle_increment)
                return max(0, min(idx, n - 1))

            # Define the angular sectors for back and sides (in radians)
            # These cover ~120° behind the robot (-150° to +150°)
            left_back_start  = angle_to_index(1.57)      # +90°
            left_back_end    = angle_to_index(2.62)      # +150°
            right_back_start = angle_to_index(-2.62)     # -150°
            right_back_end   = angle_to_index(-1.57)     # -90°

            # Combine readings from both side-back arcs
            left_back_ranges = ranges[left_back_start:left_back_end]
            right_back_ranges = ranges[right_back_start:right_back_end]

            # Merge both sides
            combined = np.concatenate((left_back_ranges, right_back_ranges))

            # Check if anything is closer than the threshold
            min_dist = np.min(combined) if len(combined) > 0 else np.inf
            self.obstacle_behind = min_dist < threshold

            if self.obstacle_behind:
                self.get_logger().warn(f"⚠️ Obstacle detected behind/sides at {min_dist:.2f} m")
            else:
                self.get_logger().warn(f"✅ Rear and sides are clear {min_dist:.2f} m")

        except Exception as e:
            self.get_logger().error(f"LaserScan error: {e}")


    def callback(self, data):

        # Convert RGB image to OpenCV format
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        rgb_cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)

        self.segmentation_frame=self.cv_image
        self.results = self.mp_pose.process(rgb_cv_image)

        if self.results.pose_landmarks is not None:
            # Person detected
            self.person_detected_before = True
            self.person_detected = True
            self.initiated_search = False                                   
            self.angular_zeed = False
            self.last_time_detected = time.time() - self.program_time                
            landmarks = self.results.pose_landmarks.landmark
            if self.goal_accepted:
                cancel_future = self.nav_to_pose_client._cancel_goal_async(self.goal)
                cancel_future.add_done_callback(self.cancel_done_callback)
            # Calculate centroid
            x_centroid = sum([landmark.x for landmark in landmarks]) / len(landmarks)
            y_centroid = sum([landmark.y for landmark in landmarks]) / len(landmarks)
            self.x_center = x_centroid * self.cv_image.shape[1]
            self.y_center = y_centroid * self.cv_image.shape[0]
            self.image_center = self.cv_image.shape[1] / 2

            cv2.circle(self.cv_image, (int(x_centroid * self.cv_image.shape[1]), int(y_centroid * self.cv_image.shape[0])), 5, (0, 0, 255), -1)

            x_min = min([landmark.x for landmark in landmarks])
            x_max = max([landmark.x for landmark in landmarks])
            y_min = min([landmark.y for landmark in landmarks])
            y_max = max([landmark.y for landmark in landmarks])

            cv2.rectangle(self.cv_image, (int(x_min * self.cv_image.shape[1]), int(y_min * self.cv_image.shape[0])),
                          (int(x_max * self.cv_image.shape[1]), int(y_max * self.cv_image.shape[0])), (0, 255, 0), 2)
            self.segmentation_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.segmentation_frame)
            # mask for the segmented image
            segmentation_result = self.segmenter.segment(self.segmentation_frame)
            category_mask = segmentation_result.category_mask

            image_data = self.segmentation_frame.numpy_view()
            fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            fg_image[:] = self.MASK_COLOR
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = self.BG_COLOR
            
            condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2

            self.segmentation_frame = np.where(condition, fg_image, bg_image)
            self.mp_drawing.draw_landmarks(self.segmentation_frame, self.results.pose_landmarks,self.mp_holistic.POSE_CONNECTIONS)

            if self.results.pose_landmarks is not None:
                cv2.line(self.cv_image, (int(self.x_center-15), int(self.y_center)), (int(self.x_center+15), int(self.y_center)), (255, 0, 0), 3) 
                cv2.line(self.cv_image, (int(self.x_center), int(self.y_center-15)), (int(self.x_center), int(self.y_center+15)), (255, 0, 0), 3)
                cv2.line(self.segmentation_frame, (int(self.x_center-15), int(self.y_center)), (int(self.x_center+15), int(self.y_center)), (255, 0, 0), 3) 
                cv2.line(self.segmentation_frame, (int(self.x_center), int(self.y_center-15)), (int(self.x_center), int(self.y_center+15)), (255, 0, 0), 3)
                cv2.line(self.segmentation_frame, (int(350), int(0)), (int(350), int(500)), (0, 0, 255), 2) 
                cv2.line(self.segmentation_frame, (int(0), int(self.y_center)), (int(700), int(self.y_center)), (0, 0, 255), 2)
            
            # Setting the limit for co-ordinates
            self.limiting_loop()
            try:
                # Check if depth information is available
                if self.depth_image is not None:
                    depth_value = self.depth_image[int(self.y_center), int(self.x_center)]
                    if isinstance(depth_value, np.ndarray):
                        depth_value = depth_value[0] 

                    self.depth_mm = float(depth_value)

                    print("Depth is:", self.depth_mm)
                else:
                    self.vel_control(0.0, 0.0)
                # Draw landmarks and bounding box
                self.draw_landmarks_and_box()
                # Move the robot based on the detected person
                self.move_robot()
            except:
                pass
        else: # No Person Detected
            self.person_detected = False
            searching = False

            curr_time = time.time() - self.program_time
            if curr_time - self.last_time_detected >= self.search_wait and not self.person_detected:
                searching = True
                self.logged_before_2 = False
                if not self.initiated_search:
                    self.initiate_search()
                    self.get_logger().info("Initiation complete")
                if not self.goal_accepted:
                    self.explore()


            if self.person_detected_before == True:  # Person was detected atleast once before
                Kp_yaw = 0.00075   # Angular P
                Ki_yaw = 0.0001  # Angular I
                Kd_yaw = 0.00007 # Angular D
                if not self.angular_zeed:
                    x_error = self.x_center - self.image_center  # Horizontal offset
                    self.integral_error_yaw += x_error
                    derivative_error_yaw = (x_error - self.last_error) / 0.6

                    angular_z = -(Kp_yaw * x_error + Ki_yaw * self.integral_error_yaw + Kd_yaw * derivative_error_yaw)

                    self.angular_zeed_last = 0.3 if angular_z >=0 else -0.3
                    # if angular_z == 0.0:
                    #     angular_z = 0.3
                    self.angular_zeed = True
                if not searching:
                    if not self.logged_before_2:
                        self.get_logger().info("The Robot is Rotating to look for a person")
                        self.logged_before_2 = True
                    self.vel_control(0.0 , self.angular_zeed_last)

                # Put text on the screen
                top = "Searching Person"
                bottom = "Stop"
                self.display_text_on_image(bottom,top)


            
            else:  # Person was never detected
                curr_time = time.time() - self.program_time
                if curr_time - self.last_time_detected >= self.search_wait and not self.person_detected:
                    searching = True
                    self.logged_before_1 = False
                    # print("duration:",curr_time - self.last_time_detected)
                    if not self.initiated_search:
                        self.initiate_search()
                        self.get_logger().info("Initiation complete")
                    if not self.goal_accepted:
                        self.explore()

                if not searching:
                    if not self.logged_before_1:
                        self.get_logger().info("The Robot is stopped")
                        self.logged_before_1 = True
                    self.vel_control(0.0, 0.0)



                                    

    def initiate_search(self):                                

        self.get_logger().info("Initiated searching in area.")
        while not self.initiated_search:
            self.get_logger().info(f"Resetting Map And Initiating Search")
            self.visited_frontiers = set()
            self.checked = 0
            # Map and position data
            self.map_data = None
            self.robot_position = (0, 0)  # Placeholder, update from localization
            client_node = ResetClient()
            
            future = client_node.send_request()
            
            rclpy.spin_until_future_complete(client_node, future)
            
            if future.result() is not None:
                client_node.get_logger().info('Slam Toolbox has been reset successfully.')
                self.initiated_search = True
            else:
                client_node.get_logger().error('Service call failed. Trying again')

                


    def draw_landmarks_and_box(self):
        cv2.circle(self.cv_image, (int(self.x_center), int(self.y_center)), 5, (0, 0, 255), -1)

        x_min = min([landmark.x for landmark in self.results.pose_landmarks.landmark])
        x_max = max([landmark.x for landmark in self.results.pose_landmarks.landmark])
        y_min = min([landmark.y for landmark in self.results.pose_landmarks.landmark])
        y_max = max([landmark.y for landmark in self.results.pose_landmarks.landmark])

        cv2.rectangle(self.cv_image, (int(x_min * self.cv_image.shape[1]), int(y_min * self.cv_image.shape[0])),
                      (int(x_max * self.cv_image.shape[1]), int(y_max * self.cv_image.shape[0])), (0, 255, 0), 2)
        self.mp_drawing.draw_landmarks(self.cv_image, self.results.pose_landmarks)



    # def move_robot(self):
    #     # Constants for PD controller
    #     Kp_l = 0.4
    #     Kp_yaw = 0.00065
    #     Kd_yaw = 0.00007
    #     Kd_l = 0.37

    #     # Calculating the error
    #     x_error = self.x_center - self.image_center -3.0
    #     try:

    #         if self.depth_mm > 3:
    #             # Determine the direction to move based on the person's position
    #             if x_error > 10:
    #                 top = "Right==>"
    #                 bottom = "Go Forward"
    #             elif x_error < -10:
    #                 top = "<==Left"
    #                 bottom = "Go Forward"
    #             else:
    #                 top = "Centre"
    #                 bottom = "Go Forward"
    #         else:
    #             # Stop the robot if depth information is insufficient
    #             self.vel_control(0.0, 0.0)
    #             top = "Centre"
    #             bottom = "Stopped"

    #         # Proportional and Derivative Drive
    #         P_x = Kp_l * self.depth_mm
    #         P_yaw = -(Kp_yaw * x_error)
    #         D_yaw = ((x_error - self.last_error) / 0.6) * Kd_yaw
    #         D_l = ((self.depth_mm - self.last_depth) / 0.6) * Kd_l

    #         self.last_depth = self.depth_mm
    #         self.last_error = x_error

    #         print(P_x,"1P_x")
    #         print(D_l,"2D_l")
    #         print(D_yaw,"3D_yaw")
    #         print(P_yaw,"4P_yaw")
    #         # Publish the Twist message to move the robot
    #         print((P_x + D_l),"px+dl",(P_yaw + D_yaw),"pyaw+dyaw")
    #         self.vel_control((P_x + D_l), (P_yaw + D_yaw))

    #         # Display text on the image
    #         self.display_text_on_image(bottom, top)
            
    #     except:
    #         pass
    
    def move_robot(self):
        # PID Constants
        Kp_l = 0.4     # Linear P
        Ki_l = 0.1   # Linear I
        Kd_l = 0.3     # Linear D

        Kp_yaw = 0.00075   # Angular P
        Ki_yaw = 0.0001  # Angular I
        Kd_yaw = 0.00007 # Angular D

        x_error = self.x_center - self.image_center  # Horizontal offset
        distance_error = self.depth_mm - self.safety_margin  # Forward/backward

        # Safety check
        if self.depth_mm <= self.safety_margin and self.depth_mm > self.critical_margin :
            self.get_logger().warn("Too close! Stopping the robot.")
            self.display_text_on_image("Too Close!", " Stopping the robot")
            x_error = self.x_center - self.image_center  # Horizontal offset
            self.integral_error_yaw += x_error
            derivative_error_yaw = (x_error - self.last_error) / 0.6

            angular_z = -(Kp_yaw * x_error + Ki_yaw * self.integral_error_yaw + Kd_yaw * derivative_error_yaw)
            self.angular_seen = angular_z

            self.vel_control(0.0 , angular_z)
            return

        elif self.depth_mm <= self.critical_margin:

            x_error = self.x_center - self.image_center  # Horizontal offset
            self.integral_error_yaw += x_error
            derivative_error_yaw = (x_error - self.last_error) / 0.6

            angular_z = -(Kp_yaw * x_error + Ki_yaw * self.integral_error_yaw + Kd_yaw * derivative_error_yaw)
            self.angular_seen = angular_z

            if self.obstacle_behind:

                self.vel_control(0.0 , angular_z)
                self.get_logger().warn("Warning!! an obstacle is behind the robot.")
                self.display_text_on_image("Warning!!", "Stopping")
            else:  


                self.vel_control(-0.3 , angular_z)
                self.get_logger().warn("Way Too Close!! Moving away the robot.")
                self.display_text_on_image("Way Too Close!", "Moving Away")
            return

        # Directional text
        if x_error > 10:
            top = "Right ==>"
        elif x_error < -10:
            top = "<== Left"
        else:
            top = "Centre"

        bottom = "Following Person"

        # PID: Angular velocity (yaw)
        self.integral_error_yaw += x_error
        derivative_error_yaw = (x_error - self.last_error) / 0.6

        angular_z = -(Kp_yaw * x_error + Ki_yaw * self.integral_error_yaw + Kd_yaw * derivative_error_yaw)

        # PID: Linear velocity (forward/backward)
        self.integral_error_linear += distance_error
        derivative_error_linear = (distance_error - self.last_depth) / 0.6

        linear_x = Kp_l * distance_error + Ki_l * self.integral_error_linear + Kd_l * derivative_error_linear

        # Clamp linear velocity (optional safety)
        linear_x = max(min(linear_x, 0.5), 0.0)  # Only forward motion up to 0.5 m/s

        self.last_error = x_error
        self.last_depth = distance_error

        self.get_logger().info(f"[PID] linear_x: {linear_x:.3f}, angular_z: {angular_z:.3f}, depth_mm: {self.depth_mm}")

        self.vel_control(linear_x, angular_z)
        self.display_text_on_image(bottom, top)



    def limiting_loop(self):
        if self.x_center > 700:
                self.x_center=699.0
        if self.y_center> 500:
                self.y_center=499.0
    def vel_control(self, vel_x, vel_spin):
        twist_msg = Twist()
        twist_msg.linear.x = float(vel_x)
        twist_msg.angular.z = float(vel_spin)
        self.velocity_publisher.publish(twist_msg)

    def display_text_on_image(self, bottom, top):
        # Display text on the image
        img = self.cv_image
        text1 =bottom
        text2 =top
        txt1_location = (300, 450)
        txt2_location = (300, 30)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        fontScale = 1
        fontColor_lt = (255, 255, 255)
        fontColor_at = (0, 100, 0)
        thickness = 1
        lineType = cv2.LINE_AA

        cv2.putText(img, text1, txt1_location, font, fontScale, fontColor_lt, thickness, lineType)
        cv2.putText(img, text2, txt2_location, font, fontScale, fontColor_at, thickness, lineType)
        cv2.imshow('Person Detection', self.cv_image)
        cv2.imshow('Person Segmentation', self.segmentation_frame)
        cv2.waitKey(3)
        
def main():
    rclpy.init()
    Mynode = PersonFollower()
    rclpy.spin(Mynode)
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
