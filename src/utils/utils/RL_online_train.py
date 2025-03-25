#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math, os, ast
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import RLAgent_DQNModel

"""msgs for subscription"""
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleAttitude
from std_msgs.msg import String

"""msgs for publishing"""
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import ActuatorMotors

class RLtraining(Node):
    def __init__(self):
        super().__init__('RL model training')

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
        self.yolo_detection_subscriber = self.create_subscription(
            String, '/yolov8/bounding_boxes', self.bbox_callback, qos_profile_yolo
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
        self.motor_command_publisher = self.create_publisher(
            ActuatorMotors, '/px4_1/fmu/in/actuator_motors', qos_profile
        )

        """
        3. Status variables
        """
        # Vehicle attitude
        self.vehicle_attitude = VehicleAttitude()
        self.roll = 0.0
        self.pitch = 0.0
        # Vehicle state
        self.state = 'ready2flight' # ready2flight -> takeoff -> Tagging

        """
        4. Learning variables
        """
        self.desired_distance = 3  # m
        self.desired_bbox_area = self.distance2area(self.desired_distance)
        self.image_center = np.array([320.0, 200.0])  # 640x480 이미지의 중심
        self.latest_bbox = None
        self.bbox_size_window = []  # 최근 10개의 bbox area를 저장하는 리스트

        """
        4. Timer setup
        """
        self.offboard_heartbeat = self.create_timer(0.05, self.offboard_heartbeat_callback) # 20Hz 
        self.main_timer = self.create_timer(0.1, self.main_callback) # 20Hz

    """
    Helper Functions
    """ 
    def distance2area(self, distance):
        return 79162.49 * np.exp(-0.94 * distance) + 2034.21 # fitting result

    def area2distance(self, area):
        return min(-1/0.94 * math.log((area-2034.21)/79162.49), 7)

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
            else:
                # 감지가 없을 경우 기존 데이터를 유지
                self.get_logger().warn("YOLO detection lost, using last detected bounding box.")
        except Exception as e:
            self.get_logger().error("Failed to parse bounding box: " + str(e))

    """
    Callback functions for therosgraph timers
    """
    def offboard_heartbeat_callback(self):
        """offboard heartbeat signal"""
        self.publish_offboard_control_mode(direct_actuator=True)

    def main_callback(self):
        if self.state == 'ready2flight':
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                    print("Iris Camera Arming...")
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0, target_system=2)
                else:
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, param7=10.0, target_system=2)
            elif self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                print("Iris Camera Taking off...")
                self.state = 'takeoff'
        
        if self.state == 'takeoff':
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                print("Learning start...")
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0, target_system=2)
                self.state = 'Tagging'
                self.tagging_start_time = self.get_clock().now().nanoseconds * 1e-9

        if self.state == 'Tagging':
            current_time = self.get_clock().now().nanoseconds * 1e-9
            
            if not hasattr(self, 'rl_initialized'):
                self.state_size = 5  # [lateral_error, vertical_error, forward_error, roll, pitch]
                self.action_size = 15 * 15 * 15 * 15  # 모터 제어값 (15개 값 * 4개 모터)
                
                self.agent = RLAgent_DQNModel.RLAgent(self.state_size, self.action_size, learning_rate=0.001, gamma=0.99)
                
                if os.path.exists("drone_rl_model.pth"):
                    self.agent.load_model("drone_rl_model.pth")
                    self.get_logger().info("Loaded existing RL model")
                
                self.last_state = None
                self.last_action = None
                self.last_action_idx = None
                self.episode_rewards = []
                self.episode_steps = 0
                self.max_episode_steps = 1000  # 최대 에피소드 길이
                self.training_episodes = 0
                self.max_training_episodes = 50  # 최대 학습 에피소드 수
                
                self.rl_initialized = True
                self.get_logger().info("RL training initialized")
                return
            
            if self.training_episodes >= self.max_training_episodes:
                self.agent.save_model("drone_rl_model.pth")
                self.get_logger().info(f"Training complete. Model saved. Total episodes: {self.training_episodes}")
                return
            
            bbox_detected = (self.latest_bbox is not None)
            
            if bbox_detected:
                bbox_center = self.latest_bbox['center']
                filtered_bbox_area = self.get_filtered_bbox_area()
                
                lateral_error = (self.image_center[0] - bbox_center[0])
                vertical_error = (self.image_center[1] - bbox_center[1]) - self.pitch/0.5615 * 240
                forward_error = self.area2distance(filtered_bbox_area) - self.desired_distance
                
                current_state = np.array([
                    lateral_error,
                    vertical_error,
                    forward_error,
                    self.roll,
                    self.pitch
                ])
                
                done = (self.episode_steps >= self.max_episode_steps or (current_time - self.latest_bbox['timestamp']) > 3.0)
                reward = RLAgent_DQNModel.compute_reward(lateral_error, vertical_error, forward_error, self.roll, self.pitch)
                
                if self.last_state is not None and self.last_action_idx is not None:
                    self.agent.remember(self.last_state, self.last_action_idx, reward, current_state, done)
                    self.episode_rewards.append(reward)
                
                action_dict = self.agent.get_action(current_state)
                self.last_action = [
                    action_dict['motor0'],
                    action_dict['motor1'],
                    action_dict['motor2'],
                    action_dict['motor3']
                ]
                self.last_action_idx = action_dict['action_idx']
                self.last_state = current_state
                
                self.publish_motor_command(self.last_action)
                
                self.agent.train()
                
                self.episode_steps += 1
                
                if done:
                    avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0
                    self.get_logger().info(f"Episode {self.training_episodes} completed. Steps: {self.episode_steps}, Avg Reward: {avg_reward:.2f}")
                    
                    # Preparing for next episode
                    self.episode_rewards = []
                    self.episode_steps = 0
                    self.training_episodes += 1
                    self.last_state = None
                    self.last_action = None
                    self.last_action_idx = None
                    
                    if self.training_episodes < self.max_training_episodes:
                        # 랜덤 위치로 이동 (오프보드 모드 유지)
                        self.publish_vehicle_command(
                            VehicleCommand.VEHICLE_CMD_DO_SET_POSITION_YAW, 
                            param5=np.random.uniform(-5.0, 5.0),  # X 좌표
                            param6=np.random.uniform(-5.0, 5.0),  # Y 좌표
                            param7=10.0,  # 고도
                            target_system=2
                        )
            else:
                search_action = [0.8, 0.8, 0.8, 0.8]  # 0.8 for hovering
                self.publish_motor_command(search_action)
                
                if self.episode_steps % 50 == 0:
                    self.get_logger().info("Searching for target...")

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