#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math, os, ast, csv
from pathlib import Path
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import Decoupled_DDPGAgent as DDPGAgent
import pickle
from collections import deque
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from threading import Lock

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

def enu_to_ned(enu_vec):
    return np.array([enu_vec[1], enu_vec[0], -enu_vec[2]])

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
        self.vehicle_pose_subscriber = self.create_subscription(
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

        """
        3. Status variables
        """
        # Vehicle attitude
        self.vehicle_status = VehicleStatus()
        self.pos = np.array([0.0, 0.0, 0.0])
        self.pos_gps = np.array([0.0, 0.0, 0.0])
        self.pitch = 0.0
        self.yaw = 0.0
        self.home_position = np.array([0.0, 0.0, 0.0])
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
        self.last_bbox_time = 0.0
        self.bbox_size_window = []  # 최근 10개의 bbox area를 저장하는 리스트
        self.pixel_count_window = []
        
        self.done_episode = False

        """
        5. Model initialization
        """
        self.results_dir = os.environ.get("RESULTS_DIR")

        self.ver_agent = DDPGAgent.DDPGAgent(actor_lr = 1.5e-4, critic_lr = 1.5e-3, mode='vertical')
        self.fwd_agent  = DDPGAgent.DDPGAgent(actor_lr = 1e-4, critic_lr = 1e-3, mode='forward')
        self.lat_agent  = DDPGAgent.DDPGAgent(actor_lr = 2e-4, critic_lr = 2e-3, mode='lateral')

        self.env_step          = 0
        self.train_step        = 0
        self.episode_rewards   = []
        self.last_ver_state, self.last_fwd_state, self.last_lat_state = None, None, None
        self.last_ver_error_norm, self.last_fwd_error_norm, self.last_lat_error_norm = None, None, None
        self.last_ver_action, self.last_fwd_action, self.last_lat_action = None, None, None
        self.high_reward_count = 0
        
        if os.path.exists(path=os.path.join(self.results_dir, "RL_model_temp_vertical.pth")):
            self.ver_agent.load_model(path=os.path.join(self.results_dir, "RL_model_temp_vertical.pth"))
            self.fwd_agent.load_model(path=os.path.join(self.results_dir, "RL_model_temp_forward.pth"))
            self.lat_agent.load_model(path=os.path.join(self.results_dir, "RL_model_temp_lateral.pth"))
            print(f"Loaded :     temporary models")
        if os.path.exists(os.path.join(self.results_dir, "RL_buffer_temp_vertical.pkl")):
            with open(os.path.join(self.results_dir, "RL_buffer_temp_vertical.pkl"), "rb") as ver_f: ver_buf_list = pickle.load(ver_f)
            with open(os.path.join(self.results_dir, "RL_buffer_temp_forward.pkl"), "rb") as fwd_f: fwd_buf_list = pickle.load(fwd_f)
            with open(os.path.join(self.results_dir, "RL_buffer_temp_lateral.pkl"), "rb") as lat_f: lat_buf_list = pickle.load(lat_f)
            self.ver_agent.buffer.buffer = deque(ver_buf_list, maxlen=self.ver_agent.buffer.buffer.maxlen)
            self.fwd_agent.buffer.buffer = deque(fwd_buf_list, maxlen=self.fwd_agent.buffer.buffer.maxlen)
            self.lat_agent.buffer.buffer = deque(lat_buf_list, maxlen=self.lat_agent.buffer.buffer.maxlen)
            print(f"Loaded :     temporary buffers     /     ({len(ver_buf_list)} transitions)")

        """
        4. Timer setup
        """
        self.offboard_heartbeat = self.create_timer(0.02, self.offboard_heartbeat_callback) # 50Hz 
        self.control_timer = self.create_timer(0.05, self.control_callback) # 20Hz
        
        train_cb_group = ReentrantCallbackGroup()
        self.train_lock = Lock()
        self.training_timer = self.create_timer(0.2, self.train_callback, callback_group=train_cb_group) # 5Hz

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
    @staticmethod
    def Distance2BBoxArea(distance):
        return 79162.49 * np.exp(-0.94 * distance) + 2034.21 # fitting result
    
    @staticmethod
    def BBoxArea2Distance(bbox_area):
        bbox_area = max(2034.30, bbox_area)
        return min(-1/0.94 * math.log((bbox_area-2034.21)/79162.49), 20)

    @staticmethod
    def Distance2PixelCount(distance):
        return 6941.21 / (distance**2)
    
    @staticmethod
    def PixelCount2Distance(pixel_count):
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
            return self.home_position

        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)
        
        x_ned = R * (lat2 - lat1)  # North
        y_ned = R * (lon2 - lon1) * np.cos(lat1)  # East
        z_ned = -(alt2 - alt1)  # Down
        
        self.home_position = np.array([x_ned, y_ned, z_ned])
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
        w, x, y, z = msg.q[0], msg.q[1], msg.q[2], msg.q[3]
        self.pitch = math.atan2(2.0 * (w * y - z * x), math.sqrt(1 - (2.0 * (w * y - z * x))**2))

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
                    self.last_bbox_time = self.get_clock().now().nanoseconds * 1e-9

                    self.bbox_size_window.append(area)
                    if len(self.bbox_size_window) > 10:
                        self.bbox_size_window.pop(0)
            else:
                self.latest_bbox = None
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

    """
    Callback functions for timers
    """
    def offboard_heartbeat_callback(self):
        """offboard heartbeat signal"""
        # Log the current time delta from the last call to monitor timing consistency
        current_time = self.get_clock().now()
        if hasattr(self, 'last_heartbeat_time'):
            delta = (current_time - self.last_heartbeat_time).nanoseconds / 1e9
            if delta > 1.0:  # More than 60ms between heartbeats
                self.get_logger().warn(f"Heartbeat delay detected: {delta:.3f}s")
        self.last_heartbeat_time = current_time
        
        self.publish_offboard_control_mode(position = True)

    def control_callback(self):
        try:
            if self.control_mode not in ['bbox', 'pixel']:
                self.get_logger().warn("ROS parameter error: mode should be 'bbox' or 'pixel'")
                return

            if self.state == 'ready2flight':
                if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                    if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                        print("Camera Arming...")
                        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0, target_system=2)
                    else:
                        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, param7=10.0, target_system=2)
                elif self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                    if not self.get_position_flag:
                        print("Waiting for position data")
                        return
                    self.home_position = self.set_home_position()
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

                current_time = self.get_clock().now().nanoseconds * 1e-9
                lost_condition = (current_time - self.last_bbox_time) > 10.0
                success_condition = self.high_reward_count >= 100
                
                if lost_condition: print("Done episode: YOLO detection lost")
                if success_condition: print("Done episode: Success condition met")

                self.done_episode = lost_condition or success_condition

                if self.done_episode:
                    if self.episode_number == self.max_episodes:
                        self.ver_agent.save_model(path=os.path.join(self.results_dir, f"RL_model_final_episode{self.episode_number}_vertical.pth"))
                        self.fwd_agent.save_model(path=os.path.join(self.results_dir, f"RL_model_final_episode{self.episode_number}_forward.pth"))
                        self.lat_agent.save_model(path=os.path.join(self.results_dir, f"RL_model_final_episode{self.episode_number}_lateral.pth"))
                    elif self.episode_number % 5 == 0:
                        self.ver_agent.save_model(path=os.path.join(self.results_dir, f"RL_model_episode{self.episode_number}_vertical.pth"))
                        self.fwd_agent.save_model(path=os.path.join(self.results_dir, f"RL_model_episode{self.episode_number}_forward.pth"))
                        self.lat_agent.save_model(path=os.path.join(self.results_dir, f"RL_model_episode{self.episode_number}_lateral.pth"))

                    self.ver_agent.save_model(path=os.path.join(self.results_dir, "RL_model_temp_vertical.pth"))
                    self.fwd_agent.save_model(path=os.path.join(self.results_dir, "RL_model_temp_forward.pth"))
                    self.lat_agent.save_model(path=os.path.join(self.results_dir, "RL_model_temp_lateral.pth"))
                    print(f"Saved :      temporary models")
                
                    with open(os.path.join(self.results_dir, "RL_buffer_temp_vertical.pkl"), "wb") as ver_f:
                        pickle.dump(list(self.ver_agent.buffer.buffer), ver_f)
                    with open(os.path.join(self.results_dir, "RL_buffer_temp_forward.pkl"), "wb") as fwd_f:
                        pickle.dump(list(self.fwd_agent.buffer.buffer), fwd_f)
                    with open(os.path.join(self.results_dir, "RL_buffer_temp_lateral.pkl"), "wb") as lat_f:
                        pickle.dump(list(self.lat_agent.buffer.buffer), lat_f)
                    print(f"Saved :      temporary buffers     /     ({len(self.ver_agent.buffer)} transitions)")
            
                    Path("/tmp/rl_episode_done.flag").touch()
                    rclpy.shutdown()
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
                            self.publish_local2global_setpoint(local_setpoint=target_ned, yaw_sp=self.yaw + 0.3)
                            return
                        
                    # --- 에러 정규화 ---
                    error_ver_norm = error_vertical   / 1.0
                    error_fwd_norm = error_forward    / 2.0
                    error_lat_norm = error_lateral    / 300.0

                    # --- 상태 벡터 생성 ---
                    if self.last_ver_error_norm is None and self.last_ver_action is None:
                        ver_state_curr = np.array([error_ver_norm, 0.0, 0.0], dtype=float)
                        fwd_state_curr = np.array([error_fwd_norm, 0.0, 0.0], dtype=float)
                        lat_state_curr = np.array([error_lat_norm, 0.0], dtype=float)
                    else:
                        ver_state_curr = np.array([error_ver_norm, self.last_ver_error_norm, self.last_fwd_action], dtype=float)
                        fwd_state_curr = np.array([error_fwd_norm, self.last_fwd_error_norm, self.last_ver_action], dtype=float)
                        lat_state_curr = np.array([error_lat_norm, self.last_lat_error_norm], dtype=float)

                    # --- 보상 계산 ---
                    ver_reward = self.ver_agent.compute_reward(error_ver_norm)
                    fwd_reward = self.fwd_agent.compute_reward(error_fwd_norm)
                    lat_reward = self.lat_agent.compute_reward(error_lat_norm)

                    # --- 경험(transition) 저장 ---
                    if self.last_ver_state is not None:
                        self.ver_agent.remember(self.last_ver_state, self.last_ver_action, ver_reward, ver_state_curr, self.done_episode)
                        self.fwd_agent.remember(self.last_fwd_state, self.last_fwd_action, fwd_reward, fwd_state_curr, self.done_episode)
                        self.lat_agent.remember(self.last_lat_state, self.last_lat_action, lat_reward, lat_state_curr, self.done_episode)
                        self.episode_rewards.append([ver_reward, fwd_reward, lat_reward])

                    # --- 액션 선택 ---
                    correction_vertical  = float(self.ver_agent.get_action(ver_state_curr, add_noise=True))
                    correction_forward   = float(self.fwd_agent.get_action(fwd_state_curr, add_noise=True))
                    correction_yaw       = float(self.lat_agent.get_action(lat_state_curr, add_noise=True))

                    # --- 마지막 상태/액션 업데이트 ---
                    self.last_ver_state         = ver_state_curr
                    self.last_ver_error_norm    = error_ver_norm
                    self.last_ver_action        = correction_vertical
                    self.last_fwd_state         = fwd_state_curr
                    self.last_fwd_error_norm    = error_fwd_norm
                    self.last_fwd_action        = correction_forward
                    self.last_lat_state         = lat_state_curr
                    self.last_lat_error_norm    = error_lat_norm
                    self.last_lat_action        = correction_yaw

                    if len(self.ver_agent.buffer) < self.ver_agent.batch_size:
                        self.publish_setpoint(local_setpoint=[0.0, 0.0, 0.0])
                    else:
                        new_yaw = float(self.yaw - correction_yaw)
                        east_error = np.sin(self.yaw) * correction_forward
                        north_error = np.cos(self.yaw) * correction_forward

                        local_enu_setpoint = np.array([east_error, north_error, correction_vertical]).flatten()
                        local_ned_setpoint = self.pos + enu_to_ned(local_enu_setpoint)
                        self.publish_setpoint(setpoint=local_ned_setpoint, yaw_sp=new_yaw)
                    
                    self.env_step += 1

                    # 에피소드 전체 평균 보상을 에이전트별로 계산
                    if self.episode_rewards:
                        ver_rewards = [r[0] for r in self.episode_rewards]
                        fwd_rewards = [r[1] for r in self.episode_rewards]
                        lat_rewards = [r[2] for r in self.episode_rewards]

                        avg_ver_reward_episode = sum(ver_rewards) / len(ver_rewards)
                        avg_fwd_reward_episode = sum(fwd_rewards) / len(fwd_rewards)
                        avg_lat_reward_episode = sum(lat_rewards) / len(lat_rewards)
                    else:
                        avg_ver_reward_episode = 0.0
                        avg_fwd_reward_episode = 0.0
                        avg_lat_reward_episode = 0.0

                    # 100-step 평균 보상을 에이전트별로 계산
                    recent_rewards = self.episode_rewards[-100:]
                    if recent_rewards:
                        ver_rewards = [r[0] for r in recent_rewards]
                        fwd_rewards = [r[1] for r in recent_rewards]
                        lat_rewards = [r[2] for r in recent_rewards]

                        avg_ver_reward_recent = sum(ver_rewards) / len(ver_rewards)
                        avg_fwd_reward_recent = sum(fwd_rewards) / len(fwd_rewards)
                        avg_lat_reward_recent = sum(lat_rewards) / len(lat_rewards)
                    else:
                        avg_ver_reward_recent = 0.0
                        avg_fwd_reward_recent = 0.0
                        avg_lat_reward_recent = 0.0

                    if self.env_step % 100 == 0:
                        csv_path = os.path.join(self.results_dir, "step_log.csv")
                        file_exists = os.path.exists(csv_path)
                        
                        with open(csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow([
                                    "episode", "env_step", "train_step", 
                                    "ver_reward(avg100env)", "fwd_reward(avg100env)", "lat_reward(avg100env)",
                                    "ver_reward(episode)", "fwd_reward(episode)", "lat_reward(episode)"
                                ])
                            writer.writerow([
                                self.episode_number, self.env_step, self.train_step, 
                                f"{avg_ver_reward_recent:.4f}", f"{avg_fwd_reward_recent:.4f}", f"{avg_lat_reward_recent:.4f}",
                                f"{avg_ver_reward_episode:.4f}", f"{avg_fwd_reward_episode:.4f}", f"{avg_lat_reward_episode:.4f}"
                            ])

                        # 성공 조건 업데이트
                        if abs(avg_ver_reward_episode) < 0.2 and abs(avg_fwd_reward_episode) < 0.3 and abs(avg_lat_reward_episode) < 0.2:
                            self.high_reward_count += 1
                        else:
                            self.high_reward_count = 0

                else:
                    target_ned = np.array([0.0, 0.0, -10.0])
                    self.publish_local2global_setpoint(local_setpoint=target_ned, yaw_sp=self.yaw + 0.3)
        except Exception as e:
            print(f"control callback error: ({e})")
            raise

    def train_callback(self):
        try:
            if len(self.ver_agent.buffer) > self.ver_agent.batch_size:
                with self.train_lock:
                    self.ver_agent.train()
                    self.fwd_agent.train()
                    self.lat_agent.train()
                    self.train_step += 1
        except Exception as e:
            print(f"train callback error: ({e})")
            raise

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

    def publish_setpoint(self, **kwargs):
        msg = TrajectorySetpoint()
        setpoint = kwargs.get("setpoint", np.nan * np.zeros(3))
        msg.position = list(setpoint)
        msg.velocity = list(kwargs.get("velocity_sp", np.nan * np.zeros(3)))
        msg.yaw = kwargs.get("yaw_sp", float('nan'))
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

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

def main(args=None):
    rclpy.init(args=args)
    node = RLtraining()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()

if __name__ == '__main__':
    main()