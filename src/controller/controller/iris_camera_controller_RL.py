#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import os, math, csv
import numpy as np
import ast
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

# import px4_msgs
"""msgs for subscription"""
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleGlobalPosition
from px4_msgs.msg import VehicleAttitude
"""msgs for publishing"""
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint

from std_msgs.msg import String

# Define neural network for Q-learning
class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class RLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.learning_rate = learning_rate
        
        # Define action space (discretized)
        self.forward_actions = np.linspace(-3.0, 3.0, 7)  # [-3, -2, -1, 0, 1, 2, 3]
        self.lateral_actions = np.linspace(-0.5, 0.5, 5)  # [-0.5, -0.25, 0, 0.25, 0.5]
        self.vertical_actions = np.linspace(-0.5, 0.5, 5)  # [-0.5, -0.25, 0, 0.25, 0.5]
        
        # Initialize networks
        self.model = DQNModel(state_size, action_size)
        self.target_model = DQNModel(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(10000)
        
        # Training parameters
        self.update_target_every = 100
        self.train_counter = 0
        
    def get_action(self, state):
        """Select action according to epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            # Random action
            action_idx = random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action_idx = torch.argmax(q_values).item()
        
        # Convert action index to actual control values
        action_space = self.action_size // 3  # Assuming equal divisions for each control
        forward_idx = action_idx % 7
        lateral_idx = (action_idx // 7) % 5
        vertical_idx = (action_idx // 35) % 5
        
        return {
            'forward': self.forward_actions[forward_idx],
            'lateral': self.lateral_actions[lateral_idx],
            'vertical': self.vertical_actions[vertical_idx],
            'action_idx': action_idx
        }
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.push(state, action, reward, next_state, done)
        
    def train(self):
        """Train the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch from memory
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and backpropagate
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network if needed
        self.train_counter += 1
        if self.train_counter % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_model(self, filepath):
        """Save model weights"""
        torch.save(self.model.state_dict(), filepath)
        
    def load_model(self, filepath):
        """Load model weights"""
        self.model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(self.model.state_dict())

def enu_to_ned(enu_vec):
    return np.array([enu_vec[1], enu_vec[0], -enu_vec[2]])

class IrisCameraControllerRL(Node):
    def __init__(self):
        super().__init__('iris_camera_controller_rl')

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
            VehicleAttitude, '/px4_1/fmu/out/vehicle_attitude', self.vehicle_pitch_callback, qos_profile
        )
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/px4_1/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile
        )
        self.vehicle_global_position_subscriber = self.create_subscription(
            VehicleGlobalPosition, '/px4_1/fmu/out/vehicle_global_position', self.vehicle_global_position_callback, qos_profile
        )
        self.yolo_detection_subscriber = self.create_subscription(String, '/yolov8/bounding_boxes', self.bbox_callback, qos_profile_yolo
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
        # Vehicle status, attitude
        self.vehicle_status = VehicleStatus()
        self.vehicle_attitude = VehicleAttitude()
        self.pos = np.array([0.0, 0.0, 0.0])
        self.pos_gps = np.array([0.0, 0.0, 0.0]) 
        self.vel = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        self.pitch = 0.0
        self.home_position = np.array([0.0, 0.0, 0.0])
        self.home_position_gps = np.array([0.0, 0.0, 0.0])  # Store initial GPS position
        self.get_position_flag = False
        # Vehicle state
        self.state = 'ready2flight' # ready2flight -> takeoff -> Tagging

        """
        4. RL variables
        """
        self.desired_distance = 3  # m
        self.desired_bbox_area = self.distance2area(self.desired_distance)
        self.image_center = np.array([320.0, 200.0])  # 640x480 image center
        
        # Define state and action spaces
        state_size = 6  # [lateral_error, vertical_error, forward_error, vel_x, vel_y, vel_z]
        action_size = 7 * 5 * 5  # Discretized actions for forward, lateral, vertical
        self.rl_agent = RLAgent(state_size, action_size)
        
        # RL state variables
        self.prev_state = None
        self.prev_action = None
        self.prev_action_idx = None
        self.total_reward = 0
        self.episode_step = 0
        self.load_model_if_exists()

        # Latest bounding box data (None if not available)
        self.latest_bbox = None
        self.bbox_size_window = []  # Store recent 10 bbox areas
        self.dt = 0.05  # Timer period

        """
        5. Error plot variables
        """
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(4, 1, figsize=(8, 8))
        self.ax[0].set_title('Forward Error')
        self.ax[1].set_title('Lateral Error')
        self.ax[2].set_title('Vertical Error')
        self.ax[3].set_title('Cumulative Reward')
        self.ax[3].set_xlabel('Time (s)')
        self.time_data = []
        self.forward_error_data = []
        self.lateral_error_data = []
        self.vertical_error_data = []
        self.reward_data = []
        self.tagging_start_time = None  # Record tagging start time
        fig_manager = plt.get_current_fig_manager()
        if hasattr(fig_manager, 'window'):
            fig_manager.window.setGeometry(3400, 1000, 700, 600)

        """
        6. Timer setup
        """
        self.offboard_heartbeat = self.create_timer(0.05, self.offboard_heartbeat_callback) # 20Hz 
        self.main_timer = self.create_timer(0.05, self.main_callback) # 20Hz
        self.train_timer = self.create_timer(0.5, self.train_callback) # 2Hz

        self.get_logger().info("Iris Camera Controller with RL initialized")

    def load_model_if_exists(self):
        model_path = "drone_rl_model.pth"
        if os.path.exists(model_path):
            try:
                self.rl_agent.load_model(model_path)
                self.get_logger().info(f"Loaded RL model from {model_path}")
                # Reduce epsilon to use more exploitation than exploration
                self.rl_agent.epsilon = 0.1
            except Exception as e:
                self.get_logger().error(f"Failed to load model: {e}")
        else:
            self.get_logger().info("No pretrained model found - starting with new model")

    """
    Helper Functions
    """ 
    def distance2area(self, distance):
        """Exponential relationship between distance and bounding box area"""
        return 79162.49 * np.exp(-0.94 * distance) + 2034.21 # fitting result
    
    def area2distance(self, area):
        """Convert bounding box area to distance"""
        return min(-1/0.94 * math.log((area-2034.21)/79162.49), 7)

    def normalized_forward_error(self, distance, forward_error):
        """Convert area error to distance error"""
        dA_dd = -0.94 * 79162.49 * np.exp(-0.94 * distance) # derivative of distance2area function
        normalized_error = -forward_error / dA_dd # convert area error to distance error
        return normalized_error

    def compute_reward(self, errors, actions):
        """Compute reward based on tracking errors and control actions"""
        lateral_error, vertical_error, forward_error = errors
        
        # Target region - higher reward for getting close to target
        target_reward = 0
        if abs(lateral_error) < 30 and abs(vertical_error) < 30 and abs(forward_error) < 0.5:
            target_reward = 10.0
        elif abs(lateral_error) < 60 and abs(vertical_error) < 60 and abs(forward_error) < 1.0:
            target_reward = 5.0
            
        # Error penalties - penalize being far from target
        lateral_penalty = -0.01 * abs(lateral_error)
        vertical_penalty = -0.01 * abs(vertical_error)
        forward_penalty = -1.0 * abs(forward_error)
        
        # Action penalties - penalize large control inputs
        action_penalty = -0.1 * (abs(actions['forward']) + abs(actions['lateral']) + abs(actions['vertical']))
        
        # Smoothness penalties - penalize jerky movements
        reward = target_reward + lateral_penalty + vertical_penalty + forward_penalty + action_penalty
        
        return reward

    def set_home_position(self):
        """Convert global GPS coordinates to local coordinates relative to home position"""     
        R = 6371000.0  # Earth radius in meters
        try:
            lat1 = float(os.environ.get('PX4_HOME_LAT', 0.0))
            lon1 = float(os.environ.get('PX4_HOME_LON', 0.0))
            alt1 = float(os.environ.get('PX4_HOME_ALT', 0.0))
        except (ValueError, TypeError) as e:
            self.get_logger().error(f"Error converting environment variables: {e}")
            lat1, lon1, alt1 = 0.0, 0.0, 0.0
            
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

    def update_error_plot(self, time, forward_error, lateral_error, vertical_error, reward):
        self.time_data.append(time)
        self.forward_error_data.append(forward_error)
        self.lateral_error_data.append(lateral_error)
        self.vertical_error_data.append(vertical_error)
        self.reward_data.append(self.total_reward)

        self.ax[0].clear()
        self.ax[1].clear()
        self.ax[2].clear()
        self.ax[3].clear()
        
        self.ax[0].plot(self.time_data, self.forward_error_data, label='Forward Error', color='red')
        self.ax[1].plot(self.time_data, self.lateral_error_data, label='Lateral Error', color='green')
        self.ax[2].plot(self.time_data, self.vertical_error_data, label='Vertical Error', color='blue')
        self.ax[3].plot(self.time_data, self.reward_data, label='Cumulative Reward', color='purple')

        self.ax[0].legend()
        self.ax[1].legend()
        self.ax[2].legend()
        self.ax[3].legend()
        self.ax[3].set_xlabel('Time (s)')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def get_filtered_bbox_area(self):
        if len(self.bbox_size_window) == 0:
            return 0
        arr = np.array(self.bbox_size_window)
        mean_area = np.mean(arr)
        std_area = np.std(arr)
        inliers = arr[(arr >= mean_area - 2 * std_area) & (arr <= mean_area + 2 * std_area)] # remove outliers
        if len(inliers) == 0:
            return mean_area  # If all are outliers, use the overall mean
        else:
            return np.mean(inliers)

    """
    Callback functions for subscribers.
    """
    def vehicle_status_callback(self, msg):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = msg

    def vehicle_pitch_callback(self, msg):
        self.vehicle_attitude = msg
        w, x, y, z = msg.q[0], msg.q[1], msg.q[2], msg.q[3]
        t = +2.0 * (w * y - z * x)
        t = +1.0 if t > +1.0 else t
        t = -1.0 if t < -1.0 else t
        self.pitch = math.asin(t)

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
            else:
                # Keep existing data if no detection
                self.get_logger().warn("YOLO detection lost, using last detected bounding box.")
        except Exception as e:
            self.get_logger().error("Failed to parse bounding box: " + str(e))

    """
    Callback functions for timers
    """
    def offboard_heartbeat_callback(self):
        """offboard heartbeat signal"""
        self.publish_offboard_control_mode(position=True)
    
    def train_callback(self):
        """Train the RL model periodically"""
        if self.prev_state is not None and self.prev_action_idx is not None:
            self.rl_agent.train()
            
    def main_callback(self):
        if self.state == 'ready2flight':
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                    print("Iris Camera Arming...")
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
                print("Seeking...")
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0, target_system=2)
                self.state = 'Tagging'
                self.tagging_start_time = self.get_clock().now().nanoseconds * 1e-9
                # Reset episode variables
                self.total_reward = 0
                self.episode_step = 0

        if self.state == 'Tagging':
            error_lateral = 0.0
            error_vertical = 0.0
            error_forward = 0.0

            if self.latest_bbox is not None:
                bbox_center = self.latest_bbox['center']
                filtered_bbox_area = self.get_filtered_bbox_area()
                
                error_lateral = (self.image_center[0] - bbox_center[0])
                error_vertical = (self.image_center[1] - bbox_center[1]) - self.pitch/0.5615 * 240
                error_forward = self.normalized_forward_error(self.area2distance(filtered_bbox_area), self.desired_bbox_area - filtered_bbox_area)

                # Create state vector for RL
                current_state = np.array([
                    error_lateral, 
                    error_vertical, 
                    error_forward,
                    self.vel[0], 
                    self.vel[1], 
                    self.vel[2]
                ])
                
                # Get action from RL agent
                action_dict = self.rl_agent.get_action(current_state)
                forward_action = action_dict['forward']
                lateral_action = action_dict['lateral']
                vertical_action = action_dict['vertical']
                action_idx = action_dict['action_idx']
                
                # Calculate reward for previous action if we have a previous state
                if self.prev_state is not None and self.prev_action is not None:
                    reward = self.compute_reward([error_lateral, error_vertical, error_forward], self.prev_action)
                    self.total_reward += reward
                    self.rl_agent.remember(self.prev_state, self.prev_action_idx, reward, current_state, False)
                    
                # Save current state and action for next iteration
                self.prev_state = current_state
                self.prev_action = action_dict
                self.prev_action_idx = action_idx
                self.episode_step += 1
                
                # Log info
                current_time = self.get_clock().now().nanoseconds * 1e-9 - self.tagging_start_time
                self.update_error_plot(current_time, error_forward, error_lateral, error_vertical, self.total_reward)
                
                self.get_logger().info(f"Action: forward={forward_action:.2f}, lateral={lateral_action:.2f}, vertical={vertical_action:.2f}")
                self.get_logger().info(f"Reward: {self.total_reward:.2f}, Epsilon: {self.rl_agent.epsilon:.2f}")
                
                # Apply action
                new_yaw = self.yaw - lateral_action
                
                yaw_rad = np.radians(self.yaw)
                cos_yaw = np.cos(yaw_rad)
                sin_yaw = np.sin(yaw_rad)
                
                x_enu = cos_yaw * forward_action - sin_yaw * 0
                y_enu = sin_yaw * forward_action + cos_yaw * 0
                
                local_enu_setpoint = np.array([x_enu, y_enu, vertical_action])
                local_ned_setpoint = self.pos + enu_to_ned(local_enu_setpoint)
                self.publish_setpoint(setpoint=local_ned_setpoint, yaw_sp=new_yaw)
                
                # Save model periodically
                if self.episode_step % 500 == 0:
                    self.rl_agent.save_model("drone_rl_model.pth")
                    self.get_logger().info("Model saved")
            else:
                self.get_logger().info("No detection received.")
                enu_setpoint = np.array([0, 0, 0])
                ned_setpoint = self.pos + enu_to_ned(enu_setpoint)
                new_yaw = self.yaw - 0
                self.publish_setpoint(setpoint=ned_setpoint, yaw_sp=new_yaw)

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
    node = IrisCameraControllerRL()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting and keeping plot window open...")
        # Save the model before exiting
        node.rl_agent.save_model("drone_rl_model.pth")
        print("Model saved to drone_rl_model.pth")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.show(block=True)  # Keep plot window open after program exit

if __name__ == '__main__':
    main()