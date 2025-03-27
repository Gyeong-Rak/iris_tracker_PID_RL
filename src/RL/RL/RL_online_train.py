#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math, os, ast
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import RLAgent_DQNModel

"""msgs for subscription"""
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleGlobalPosition
from px4_msgs.msg import VehicleAttitude
from std_msgs.msg import String, Int32

"""msgs for publishing"""
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import ActuatorMotors

class RLtraining(Node):
    def __init__(self):
        super().__init__('RL_model_training')

        """
        0. Configure QoS profile for publishing and subscribing
        """
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_yolo = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        """
        1. Create Subscribers
        """
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/px4_1/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile
        )
        self.vehicle_pitch_subscriber = self.create_subscription(
            VehicleAttitude, '/px4_1/fmu/out/vehicle_attitude', self.roll_pitch_callback, qos_profile
        )
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/px4_1/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile
        )
        self.vehicle_global_position_subscriber = self.create_subscription(
            VehicleGlobalPosition, '/px4_1/fmu/out/vehicle_global_position', self.vehicle_global_position_callback, qos_profile
        )
        self.yolo_detection_subscriber = self.create_subscription(
            String, '/YOLOv8/bounding_boxes', self.bbox_callback, qos_profile_yolo
        )
        self.fastsam_detection_subscriber = self.create_subscription(
            Int32, '/FastSAM/object_pixel_count', self.pixel_count_callback, qos_profile_yolo
        )

        """
        2. Create Publishers
        """
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/px4_1/fmu/in/vehicle_command', qos_profile
        )
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/px4_1/fmu/in/offboard_control_mode', qos_profile
        )
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/px4_1/fmu/in/trajectory_setpoint', qos_profile
        )
        self.motor_command_publisher = self.create_publisher(
            ActuatorMotors, '/px4_1/fmu/in/actuator_motors', qos_profile
        )

        """
        3. Status variables
        """
        # Vehicle attitude
        self.vehicle_status = VehicleStatus()
        self.vehicle_attitude = VehicleAttitude()
        self.pos = np.array([0.0, 0.0, 0.0])
        self.pos_gps = np.array([0.0, 0.0, 0.0])
        self.roll = 0.0
        self.pitch = 0.0
        self.home_position = np.array([0.0, 0.0, 0.0])
        self.home_position_gps = np.array([0.0, 0.0, 0.0])
        self.get_position_flag = False

        # Vehicle state
        self.state = 'ready2flight' # ready2flight -> takeoff -> Tagging

        """
        4. Learning variables
        """
        self.desired_distance = 3  # m
        self.desired_bbox_area = self.Distance2BBoxArea(self.desired_distance)
        self.desired_pixel_count = self.Distance2PixelCount(self.desired_distance)
        self.image_center = np.array([320.0, 200.0])  # 640x480 이미지의 중심

        self.latest_bbox = None
        self.bbox_size_window = []  # 최근 10개의 bbox area를 저장하는 리스트
        self.pixel_count_window = []

        """
        4. Timer setup
        """
        self.offboard_heartbeat = self.create_timer(0.02, self.offboard_heartbeat_callback) # 50Hz 
        self.main_timer = self.create_timer(0.1, self.main_callback) # 10Hz

        """
        7. Node parameter
        """
        self.declare_parameter('mode', 'pixel')
        self.control_mode = self.get_parameter('mode').get_parameter_value().string_value

        self.declare_parameter('episode', 0)
        self.episode_number = self.get_parameter('episode').get_parameter_value().integer_value

        self.declare_parameter('max_episodes', 50)
        self.max_episodes = self.get_parameter('max_episodes').get_parameter_value().integer_value

    """
    Helper Functions
    """ 
    def Distance2BBoxArea(self, distance):
        return 79162.49 * np.exp(-0.94 * distance) + 2034.21 # fitting result
    
    def BBoxArea2Distance(self, bbox_area):
        bbox_area = max(2034.30, bbox_area)
        return min(-1/0.94 * math.log((bbox_area-2034.21)/79162.49), 20)

    def Distance2PixelCount(self, distance):
        return 6941.21 / (distance**2)
    
    def PixelCount2Distance(self, pixel_count):
        return min(math.sqrt(6941.21 / pixel_count), 20)

    def set_home_position(self):
        """Convert global GPS coordinates to local coordinates relative to home position"""     
        R = 6371000.0  # Earth radius in meters
        try:
            lat1 = float(os.environ.get('PX4_HOME_LAT', 0.0))
            lon1 = float(os.environ.get('PX4_HOME_LON', 0.0))
            alt1 = float(os.environ.get('PX4_HOME_ALT', 0.0))
        except (ValueError, TypeError) as e:
            self.get_logger().error(f"Error converting environment variables: {e}")
            lat1, lon1, alt1 = 47.397742, 8.545594, 488.0
            
        lat2, lon2, alt2 = self.pos_gps
        
        if lat1 == 0.0 and lon1 == 0.0:
            self.get_logger().warn("No home position in environment variables, using current position")
            self.home_position = np.array([0.0, 0.0, 0.0])
            self.home_position_gps = self.pos_gps.copy()
            return self.home_position

        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)
        
        x_ned = R * (lat2 - lat1)  # North
        y_ned = R * (lon2 - lon1) * np.cos(lat1)  # East
        z_ned = -(alt2 - alt1)  # Down
        
        self.home_position = np.array([x_ned, y_ned, z_ned])
        self.home_position_gps = np.array([lat1, lon1, alt1])
        return self.home_position    

    def get_filtered_bbox_area(self):
        if len(self.bbox_size_window) == 0:
            return 0
        arr = np.array(self.bbox_size_window)
        mean_area = np.mean(arr)
        std_area = np.std(arr)
        inliers = arr[(arr >= mean_area - 2 * std_area) & (arr <= mean_area + 2 * std_area)] # remove outliers
        if len(inliers) == 0:
            return mean_area  # 모두 outlier면 그냥 전체 평균 사용
        else:
            return np.mean(inliers)
        
    def get_filtered_pixel_count(self):
        if len(self.pixel_count_window) == 0:
            return 0
        arr = np.array(self.pixel_count_window)
        mean_count = np.mean(arr)
        std_count = np.std(arr)
        inliers = arr[(arr >= mean_count - 2 * std_count) & (arr <= mean_count + 2 * std_count)] # remove outliers
        if len(inliers) == 0:
            return mean_count  # 모두 outlier면 그냥 전체 평균 사용
        else:
            return np.mean(inliers)

    def VerticalError2RelativeAltitude(self, error_vertical, distance, vertical_fov=1.123):
        angle_per_pixel = vertical_fov / 480
        angle_offset = error_vertical * angle_per_pixel
        total_angle = self.pitch + angle_offset
        return distance * math.sin(total_angle)

    """
    Callback functions for subscribers.
    """
    def vehicle_status_callback(self, msg):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = msg

    def roll_pitch_callback(self, msg):
        self.vehicle_attitude = msg
        w, x, y, z = msg.q[0], msg.q[1], msg.q[2], msg.q[3]

        self.pitch = math.atan2(2.0 * (w * y - z * x), math.sqrt(1 - (2.0 * (w * y - z * x))**2))
        self.roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))

    def vehicle_local_position_callback(self, msg): # NED
        self.vehicle_local_position = msg
        self.pos = np.array([msg.x, msg.y, msg.z])
        self.vel = np.array([msg.vx, msg.vy, msg.vz])
        self.yaw = msg.heading

    def vehicle_global_position_callback(self, msg):
        self.get_position_flag = True
        self.vehicle_global_position = msg
        self.pos_gps = np.array([msg.lat, msg.lon, msg.alt])

    def bbox_callback(self, msg):
        try:
            bbox_list = ast.literal_eval(msg.data)
            if isinstance(bbox_list, list) and len(bbox_list) > 0:
                bbox = bbox_list[0]
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
                    area = (x2 - x1) * (y2 - y1)
                    self.latest_bbox = {'center': center, 'area': area, 'timestamp': self.get_clock().now().nanoseconds * 1e-9}

                    self.bbox_size_window.append(area)
                    if len(self.bbox_size_window) > 10:
                        self.bbox_size_window.pop(0)
                    if self.state == 'Tagging':
                        print("YOLO detected")
            else:
                self.latest_bbox = None
                if self.state == 'Tagging':
                    print("YOLO detection lost")
        except Exception as e:
            print("Failed to parse bounding box: " + str(e))

    def pixel_count_callback(self, msg):
        if msg.data == 0:
            self.pixel_count = None
        else:
            self.pixel_count = msg.data

            self.pixel_count_window.append(self.pixel_count)
            if len(self.pixel_count_window) > 10:
                self.pixel_count_window.pop(0)

            if self.state == 'Tagging':
                print(f"FastSAM pixel count received: {self.pixel_count}")

    """
    Callback functions for timers
    """
    def offboard_heartbeat_callback(self):
        """offboard heartbeat signal"""
        # Log the current time delta from the last call to monitor timing consistency
        current_time = self.get_clock().now()
        if hasattr(self, 'last_heartbeat_time'):
            delta = (current_time - self.last_heartbeat_time).nanoseconds / 1e9
            if delta > 0.06:  # More than 60ms between heartbeats
                self.get_logger().warn(f"Heartbeat delay detected: {delta:.3f}s")
        self.last_heartbeat_time = current_time
        
        self.publish_offboard_control_mode(position=True)

    def main_callback(self):
        if self.control_mode not in ['bbox', 'pixel']:
            self.get_logger().warn("ROS parameter error: mode should be 'bbox' or 'pixel'")
            return

        if self.state == 'ready2flight':
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                    print("Irisstart_training.py를 실행하면 모델 학습이 잘 될 지, 꽁꽁히판단해줘. Camera Arming...")
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0, target_system=2)
                else:
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, param7=10.0, target_system=2)
            elif self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                if not self.get_position_flag:
                    print("Waiting for position data")
                    return
                self.home_position = self.set_home_position()
                self.get_logger().info(f"Home position set to: [{self.home_position[0]:.2f}, {self.home_position[1]:.2f}, {self.home_position[2]:.2f}]")
                print("Iris Camera Taking off...")
                self.state = 'takeoff'
        
        if self.state == 'takeoff':
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                print("Learning start...")
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0, target_system=2)
                self.state = 'Tagging'
                self.tagging_start_time = self.get_clock().now().nanoseconds * 1e-9

        if self.state == 'Tagging':
            error_lateral = 0.0
            error_vertical = 0.0
            error_forward = 0.0
            
            if not hasattr(self, 'rl_initialized'):
                self.agent = RLAgent_DQNModel.RLAgent(learning_rate=0.001, gamma=0.99)
                self.state_size = self.agent.state_size
                self.action_size = self.agent.action_size
                self.episode_steps = 0
                self.episode_rewards = []
                self.last_step_state = None
                self.last_step_action_idx = None
                
                if os.path.exists("RL_model_temp.pth"):
                    self.agent.load_model("RL_model_temp.pth")
                    self.get_logger().info("Loaded existing RL model")
                
                self.rl_initialized = True
                self.get_logger().info("RL training initialized")
                return
            
            if self.latest_bbox is not None:
                bbox_center = self.latest_bbox['center']
                filtered_bbox_area = self.get_filtered_bbox_area()
                filtered_pixel_count = self.get_filtered_pixel_count()
                
                if self.control_mode == 'bbox':
                    error_lateral = (self.image_center[0] - bbox_center[0])
                    error_vertical = self.VerticalError2RelativeAltitude(self.image_center[1] - bbox_center[1], self.BBoxArea2Distance(filtered_bbox_area))
                    error_forward = self.BBoxArea2Distance(filtered_bbox_area) - self.desired_distance
                elif self.control_mode == 'pixel':
                    if 10 < filtered_pixel_count <15000:
                        error_lateral = (self.image_center[0] - bbox_center[0])
                        error_vertical = self.VerticalError2RelativeAltitude(self.image_center[1] - bbox_center[1], self.PixelCount2Distance(filtered_pixel_count))
                        error_forward = self.PixelCount2Distance(filtered_pixel_count) - self.desired_distance
                    else:
                        target_ned = np.array([0.0, 0.0, -10.0])
                        self.publish_local2global_setpoint(local_setpoint=target_ned, yaw_sp=self.yaw + 0.5)
                        print(f"Go to origin")
                        return
                
                current_step_state = np.array([
                    error_lateral,
                    error_vertical,
                    error_forward,
                    self.roll,
                    self.pitch
                ])
                
                current_time = self.get_clock().now().nanoseconds * 1e-9
                done_episode = current_time - self.latest_bbox['timestamp'] > 3.0
                reward = self.agent.compute_reward(error_lateral, error_vertical, error_forward, self.roll, self.pitch)
                
                if self.last_step_state is not None and self.last_step_action_idx is not None:
                    self.agent.remember(self.last_step_state, self.last_step_action_idx, reward, current_step_state, done_episode)
                    self.episode_rewards.append(reward)
                
                action_dict = self.agent.get_action(current_step_state)
                self.last_step_action = [
                    action_dict['motor0'],
                    action_dict['motor1'],
                    action_dict['motor2'],
                    action_dict['motor3']
                ]
                self.last_step_action_idx = action_dict['action_idx']
                self.last_step_state = current_step_state
                
                self.publish_motor_command(self.last_step_action)
                
                self.agent.train()
                
                self.episode_steps += 1

                if self.episode_steps % 100 == 0:
                    avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0
                    self.get_logger().info(f"Step: {self.episode_steps}, Avg Reward: {avg_reward:.2f}")
                    
                if done_episode:
                    model_name = "RL_model_temp.pth"
                    self.agent.save_model(model_name)

                    if self.episode_number == self.max_episodes:
                        model_name = f"RL_model_final_episode{self.episode_number}.pth"
                        self.agent.save_model(model_name)
                        self.get_logger().info(f"Training complete. Model saved. Total episodes: {self.episode_number}")
                        rclpy.shutdown()
                    elif self.episode_number % 10 == 0:
                        model_name = f"RL_model_episode{self.episode_number}.pth"
                        self.agent.save_model(model_name)
                        self.get_logger().info(f"Model saved. Current episodes: {self.episode_number}")
                        return
            else:
                target_ned = np.array([0.0, 0.0, -10.0])
                self.publish_local2global_setpoint(local_setpoint=target_ned, yaw_sp=self.yaw + 0.5)
                print(f"Go to origin")

    """
    Functions for publishing topics.
    """
    def publish_vehicle_command(self, command, **kwargs):
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = kwargs.get("param1", float('nan'))
        msg.param2 = kwargs.get("param2", float('nan'))
        msg.param3 = kwargs.get("param3", float('nan'))
        msg.param4 = kwargs.get("param4", float('nan'))
        msg.param5 = kwargs.get("param5", float('nan'))
        msg.param6 = kwargs.get("param6", float('nan'))
        msg.param7 = kwargs.get("param7", float('nan'))
        msg.target_system = kwargs.get("target_system", 1)
        msg.target_component = kwargs.get("target_component", 1)
        msg.source_system = kwargs.get("source_system", 1)
        msg.source_component = kwargs.get("source_component", 1)
        msg.from_external = kwargs.get("from_external", True)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def publish_offboard_control_mode(self, **kwargs):
        msg = OffboardControlMode()
        msg.position = kwargs.get("position", False)
        msg.velocity = kwargs.get("velocity", False)
        msg.acceleration = kwargs.get("acceleration", False)
        msg.attitude = kwargs.get("attitude", False)
        msg.body_rate = kwargs.get("body_rate", False)
        msg.thrust_and_torque = kwargs.get("thrust_and_torque", False)
        msg.direct_actuator = kwargs.get("direct_actuator", False)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_offboard_control_mode(self, **kwargs):
        msg = OffboardControlMode()
        msg.position = kwargs.get("position", False)
        msg.velocity = kwargs.get("velocity", False)
        msg.acceleration = kwargs.get("acceleration", False)
        msg.attitude = kwargs.get("attitude", False)
        msg.body_rate = kwargs.get("body_rate", False)
        msg.thrust_and_torque = kwargs.get("thrust_and_torque", False)
        msg.direct_actuator = kwargs.get("direct_actuator", False)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_local2global_setpoint(self, **kwargs):
        msg = TrajectorySetpoint()
        local_setpoint = kwargs.get("local_setpoint", np.nan * np.zeros(3))
        global_setpoint = local_setpoint - self.home_position
        msg.position = list(global_setpoint)
        msg.velocity = list(kwargs.get("velocity_sp", np.nan * np.zeros(3)))
        msg.yaw = kwargs.get("yaw_sp", float('nan'))
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.get_logger().debug(f"Global setpoint: {global_setpoint}")
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_motor_command(self, motor_values):
        msg = ActuatorMotors()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.timestamp_sample = msg.timestamp
        msg.reversible_flags = 0  # 일반적인 쿼드로터에서는 0

        # 12개 값 중 첫 4개 값만 사용, 나머지는 NaN
        full_motor_values = np.full(12, np.nan, dtype=np.float32)
        full_motor_values[:4] = motor_values  # 모터 0~3에 값 설정

        msg.control = full_motor_values.tolist()
        self.motor_command_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RLtraining()
    rclpy.spin(node)

if __name__ == '__main__':
    main()