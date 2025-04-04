#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import os, math, pickle, csv
import numpy as np
import ast
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
# import RLAgent_DQNModel

"""msgs for subscription"""
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleGlobalPosition
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import ActuatorOutputs
from std_msgs.msg import String, Int32

"""msgs for publishing"""
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint

class PID:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

def enu_to_ned(enu_vec):
    return np.array([enu_vec[1], enu_vec[0], -enu_vec[2]])

class IrisCameraController(Node):
    def __init__(self):
        super().__init__('iris_camera_controller')

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
        self.motor_output_subscriber = self.create_subscription(
            ActuatorOutputs, '/px4_1/fmu/out/actuator_outputs', self.motor_output_callback, qos_profile
        )
        self.yolo_detection_subscriber = self.create_subscription(
            String, '/YOLOv8/bounding_boxes', self.bbox_callback, qos_profile_yolo
        )
        self.fastsam_detection_subscriber = self.create_subscription(
            Int32, '/FastSAM/object_pixel_count', self.pixel_count_callback, qos_profile_yolo
        )
        # self.iris_local_pos_subscriber = self.create_subscription(
        #     VehicleLocalPosition, '/px4_2/fmu/out/vehicle_local_position', self.vehicle_local_position_callback2, qos_profile
        # )

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
        self.yaw = 0.0
        self.pitch = 0.0
        self.home_position = np.array([0.0, 0.0, 0.0])
        self.home_position_gps = np.array([0.0, 0.0, 0.0])  # Store initial GPS position
        self.get_position_flag = False
        # Vehicle state
        self.state = 'ready2flight' # ready2flight -> takeoff -> Tagging

        """
        4. PID variables
        """
        self.desired_distance = 3  # m
        self.desired_bbox_area = self.Distance2BBoxArea(self.desired_distance)
        self.desired_pixel_count = self.Distance2PixelCount(self.desired_distance)
        self.image_center = np.array([320.0, 240.0])  # 640x480 이미지의 중심

        # PID controller initialization
        self.dt = 0.05
        self.pid_forward = PID(Kp=1.5, Ki=0.0, Kd=0.0, dt=self.dt)
        self.pid_lateral = PID(Kp=0.007, Ki=0.01, Kd=0.01, dt=self.dt)
        self.pid_vertical = PID(Kp=2, Ki=0.0, Kd=0.1, dt=self.dt)

        self.latest_bbox = None
        self.bbox_size_window = []  # 최근 10개의 bbox area를 저장하는 리스트
        self.pixel_count_window = []  # 

        # """
        # 5. variables for saving expert data
        # """
        # self.expert_data_path = "expert_data.pkl"
        # if os.path.exists(self.expert_data_path):
        #     with open(self.expert_data_path, "rb") as f:
        #         self.expert_data = pickle.load(f)
        #     print(f"Loaded existing expert data: {len(self.expert_data)} samples")
        # else:
        #     self.expert_data = []
        # self.last_state = None
        # self.last_action_idx = None

        # self.last_bbox_time = 0.0

        # """
        # 5.1. Error plot variables
        # """
        # plt.ion()  # 인터랙티브 모드 활성화
        # self.fig, self.ax = plt.subplots(3, 1, figsize=(6, 5))
        # self.ax[0].set_title('Forward Error')
        # self.ax[1].set_title('Lateral Error')
        # self.ax[2].set_title('Vertical Error')
        # self.ax[2].set_xlabel('Time (s)')
        # self.time_data = []
        # self.forward_error_data = []
        # self.lateral_error_data = []
        # self.vertical_error_data = []
        # self.tagging_start_time = None  # Tagging 상태 시작 시각 기록용
        # fig_manager = plt.get_current_fig_manager()
        # fig_manager.window.setGeometry(3400, 1000, 700, 450)  # (X, Y, Width, Height)

        # """
        # 5.2. Distance-Area plot variables
        # """
        # self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 5))
        # self.ax.set_title('Pixel Count')
        # self.csv_filename = "pixel_count_log.csv"
        # self.csv_file = open(self.csv_filename, mode='w', newline='')
        # self.csv_writer = csv.writer(self.csv_file)
        # self.csv_writer.writerow(["Time (s)", "Pixel count"])

        """
        6. Timer setup
        """
        self.offboard_heartbeat = self.create_timer(0.02, self.offboard_heartbeat_callback) # 50Hz 
        self.main_timer = self.create_timer(0.1, self.main_callback) # 10Hz

        """
        7. Node parameter
        """
        self.declare_parameter('mode', 'bbox')
        self.control_mode = self.get_parameter('mode').get_parameter_value().string_value

        # """
        # 8. Save errors
        # """
        # self.error_csv_filename = f"error_log_{self.control_mode}_{self.desired_distance}m.csv"
        # self.error_csv_file = open(self.error_csv_filename, mode='w', newline='')
        # self.error_csv_writer = csv.writer(self.error_csv_file)

        # self.error_csv_writer.writerow(["Time (s)", "error_lateral", "error_vertical", "error_forward", "real_forward_error"])

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

    def update_error_plot(self, time, forward_error, lateral_error, vertical_error):
        self.time_data.append(time)
        self.forward_error_data.append(forward_error)
        self.lateral_error_data.append(lateral_error)
        self.vertical_error_data.append(vertical_error)

        self.ax[0].clear()
        self.ax[1].clear()
        self.ax[2].clear()
        
        self.ax[0].plot(self.time_data, self.forward_error_data, label='Forward Error', color='red')
        self.ax[1].plot(self.time_data, self.lateral_error_data, label='Lateral Error', color='green')
        self.ax[2].plot(self.time_data, self.vertical_error_data, label='Vertical Error', color='blue')

        self.ax[0].legend()
        self.ax[1].legend()
        self.ax[2].legend()
        self.ax[2].set_xlabel('Time (s)')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_area_plot(self, time, area):
        self.time_data.append(time)
        self.area_data.append(area)

        self.csv_writer.writerow([time, area])
        self.csv_file.flush()

        if hasattr(self, 'area_line'):
            self.area_line.set_xdata(self.time_data)
            self.area_line.set_ydata(self.area_data)
        else:
            self.area_line, = self.ax.plot(self.time_data, self.area_data, label='Pixel count', color='red')

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.legend()
        self.ax.set_xlabel('Time (s)')

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

    def get_action_idx_from_corrections(self, correction_vertical, correction_forward, correction_yaw, 
                                        vertical_action_space, forward_action_space, yaw_action_space):
        vertical_idx = int(np.abs(vertical_action_space - correction_vertical).argmin())
        forward_idx  = int(np.abs(forward_action_space - correction_forward).argmin())
        yaw_idx      = int(np.abs(yaw_action_space - correction_yaw).argmin())
        action_idx = vertical_idx + forward_idx * len(vertical_action_space) + yaw_idx * (len(vertical_action_space) * len(forward_action_space))
        
        return action_idx

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
        self.yaw = msg.heading

    def vehicle_local_position_callback2(self, msg2): # NED
        self.vehicle_local_position2= msg2
        self.pos2 = np.array([msg2.x, msg2.y, msg2.z])
        self.yaw2 = msg2.heading

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
                    # if self.state == 'Tagging':
                    #     print("YOLO detected")
            else:
                self.latest_bbox = None
                # if self.state == 'Tagging':
                #     print("YOLO detection lost")
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

            # if self.state == 'Tagging':
            #     print(f"FastSAM pixel count received: {self.pixel_count}")

    def motor_output_callback(self, msg):
        self.motor_values = msg.output[:4]  # 쿼드로터 기준 motor 0~3
        print(f"[M0: {self.motor_values[0]:.2f}, M1: {self.motor_values[1]:.2f}, M2: {self.motor_values[2]:.2f}, M3: {self.motor_values[3]:.2f}")

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
        
        self.publish_offboard_control_mode(position=True, direct_actuator=True)

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
                # print(f"Home position set to: [{self.home_position[0]:.2f}, {self.home_position[1]:.2f}, {self.home_position[2]:.2f}]")
                print("Iris Camera Taking off...")
                self.state = 'takeoff'
        
        if self.state == 'takeoff':
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                print("Seeking...")
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0, target_system=2)
                self.state = 'Tagging'
                self.tagging_start_time = self.get_clock().now().nanoseconds * 1e-9

        if self.state == 'Tagging':
            error_lateral = 0.0
            error_vertical = 0.0
            error_forward = 0.0

            # ### for expert data
            # current_time = self.get_clock().now().nanoseconds * 1e-9
            # done_episode = current_time - self.last_bbox_time > 10.0

            # if done_episode:
            #     if self.last_state is not None and self.last_action_idx is not None:
            #         final_sample = {
            #             "state": self.last_state,
            #             "action_idx": self.last_action_idx,
            #             "reward": 0.0,
            #             "next_state": self.last_state,
            #             "done": True
            #         }
            #         self.expert_data.append(final_sample)

            #     with open("expert_data.pkl", "wb") as f:
            #         pickle.dump(self.expert_data, f)
            #     self.get_logger().info(f"Episode done. Expert data saved: {len(self.expert_data)} samples")

            #     os.makedirs("/tmp", exist_ok=True)
            #     with open("/tmp/rl_episode_done.flag", "w") as f:
            #         f.write("done")

            #     rclpy.shutdown()
            #     return
            # ###

            if self.latest_bbox is not None:
                bbox_center = self.latest_bbox['center']

                if self.control_mode == 'bbox':
                    filtered_bbox_area = self.get_filtered_bbox_area()
                    error_lateral = (self.image_center[0] - bbox_center[0])
                    error_vertical = self.VerticalError2RelativeAltitude(self.image_center[1] - bbox_center[1], self.BBoxArea2Distance(filtered_bbox_area))
                    error_forward = self.BBoxArea2Distance(filtered_bbox_area) - self.desired_distance
                    
                    # current_time = self.get_clock().now().nanoseconds * 1e-9 - self.tagging_start_time
                    # real_forward_error = np.linalg.norm(self.pos - self.pos2) - self.desired_distance
                    # self.error_csv_writer.writerow([
                    #     current_time,
                    #     error_lateral,
                    #     error_vertical,
                    #     error_forward,
                    #     real_forward_error
                    # ])
                    # self.error_csv_file.flush()

                elif self.control_mode == 'pixel' and self.get_filtered_pixel_count() > 10:
                    filtered_pixel_count = self.get_filtered_pixel_count()
                    error_lateral = (self.image_center[0] - bbox_center[0])
                    error_vertical = self.VerticalError2RelativeAltitude(self.image_center[1] - bbox_center[1], self.PixelCount2Distance(filtered_pixel_count))
                    error_forward = self.PixelCount2Distance(filtered_pixel_count) - self.desired_distance

                    # current_time = self.get_clock().now().nanoseconds * 1e-9 - self.tagging_start_time
                    # real_forward_error = np.linalg.norm(self.pos - self.pos2) - self.desired_distance
                    # self.error_csv_writer.writerow([
                    #     current_time,
                    #     error_lateral,
                    #     error_vertical,
                    #     error_forward,
                    #     real_forward_error
                    # ])
                    # self.error_csv_file.flush()

                else:
                    # target_ned = np.array([0.0, 0.0, -10.0])
                    # self.publish_local2global_setpoint(local_setpoint=target_ned, yaw_sp=self.yaw + 0.5)
                    # print(f"Go to origin")
                    self.publish_setpoint(setpoint=self.pos)
                    return

                # print(f"  ver_E: {error_vertical:.2f},   lat_E: {error_lateral:.2f},   for_E: {error_forward:.2f}")

                correction_vertical = self.pid_vertical.update(error_vertical)
                correction_forward = np.clip(self.pid_forward.update(error_forward), -3, 20)
                correction_yaw = np.clip(self.pid_lateral.update(error_lateral), -0.5, 0.5)
                # if correction_forward == 20 and correction_vertical > 0: correction_vertical = 5
                
                # print(f"ver_cor: {correction_vertical:.2f}, lat_cor: {correction_yaw:.2f}, for_cor: {correction_forward:.2f}")

                new_yaw = self.yaw - correction_yaw

                east_error = np.sin(self.yaw) * correction_forward
                north_error = np.cos(self.yaw) * correction_forward

                local_enu_setpoint = np.array([east_error, north_error, correction_vertical])
                local_ned_setpoint = self.pos + enu_to_ned(local_enu_setpoint)
                self.publish_setpoint(setpoint=local_ned_setpoint, yaw_sp=new_yaw)
                # print(f"N: {local_ned_setpoint[0]:.4f}, E: {local_ned_setpoint[1]:.4f}, D: {local_ned_setpoint[2]:.4f}, cor_yaw: {correction_yaw:.4f}")
                # self.publish_local2global_setpoint(local_setpoint=np.array([0, 0, -5]))

                # ### for expert data
                # current_state = [
                #     error_lateral,
                #     error_vertical,
                #     error_forward
                # ]

                # agent = RLAgent_DQNModel.RLAgent()
                # current_action_idx = self.get_action_idx_from_corrections(correction_vertical, correction_forward, correction_yaw, 
                #                                                   agent.vertical_action_space, agent.forward_action_space, agent.yaw_action_space)
                # reward = agent.compute_reward(error_lateral, error_vertical, error_forward)

                # if self.last_state is not None and self.last_action_idx is not None:
                #     sample = {
                #         "state": self.last_state,
                #         "action_idx": self.last_action_idx,
                #         "reward": reward,
                #         "next_state": current_state,
                #         "done": False
                #     }
                #     self.expert_data.append(sample)

                #     if len(self.expert_data) % 1000 == 0:
                #         print(f"Expert data collected: {len(self.expert_data)} samples")

                # self.last_state = current_state
                # self.last_action_idx = current_action_idx
                # ###

            else:
                """
                Missing state policy
                """
                # ### go to view point
                # target_offset_ned = np.array([0.0, 0.0, -10.0])  # home_position 기준 상대 좌표
                # target_ned = self.home_position + target_offset_ned  # home_position을 기준으로 변환된 목표 위치

                # delta_x = -self.home_position[0]
                # delta_y = -self.home_position[1]

                # yaw_to_target_ned = np.degrees(np.arctan2(delta_y, delta_x))

                # self.publish_setpoint(setpoint=target_ned, yaw_sp=yaw_to_target_ned)
                # print(f"Setpoint: {target_ned}")

                ### go to origin and turn
                target_ned = np.array([0.0, 0.0, -10.0])
                self.publish_local2global_setpoint(local_setpoint=target_ned, yaw_sp=self.yaw + 0.5)
                print(f"Go to origin")

                # ### just turn
                # self.publish_setpoint(setpoint = [0.0, 0.0, 0.0], yaw_sp = self.yaw)

                # ### stop
                # self.publish_setpoint(setpoint=self.pos)

            # current_time = self.get_clock().now().nanoseconds * 1e-9 - self.tagging_start_time
            # self.update_error_plot(current_time, error_forward, error_lateral, error_vertical)
            # self.publish_local2global_setpoint(local_setpoint=np.array([0, 0, -5]))
            # self.update_area_plot(current_time, self.pixel_count)

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
    node = IrisCameraController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received.")
    finally:
        # with open("expert_data.pkl", "wb") as f:
        #     pickle.dump(node.expert_data, f)
        # print(f"Expert data saved: {len(node.expert_data)} samples")

        # node.error_csv_file.close()

        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()