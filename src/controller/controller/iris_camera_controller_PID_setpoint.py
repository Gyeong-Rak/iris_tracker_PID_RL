#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import os, math, csv
import numpy as np
import ast
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import matplotlib.pyplot as plt

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
        4. PID variables
        """
        self.desired_distance = 3  # m
        self.desired_bbox_area = self.distance2area(self.desired_distance)
        self.image_center = np.array([320.0, 200.0])  # 640x480 이미지의 중심

        # PID controller initialization
        self.dt = 0.05  # 타이머 주기
        # self.pid_forward = PID(Kp=2, Ki=0.01, Kd=0.01, dt=self.dt)
        # self.pid_lateral = PID(Kp=0.008, Ki=0.00, Kd=0.001, dt=self.dt)
        # self.pid_vertical = PID(Kp=0.03, Ki=0.002, Kd=0.001, dt=self.dt)
        self.pid_forward = PID(Kp=2, Ki=0.0, Kd=0.0, dt=self.dt)
        self.pid_lateral = PID(Kp=0.008, Ki=0.0, Kd=0.0, dt=self.dt)
        self.pid_vertical = PID(Kp=0.03, Ki=0.0, Kd=0.0, dt=self.dt)

        # 최신 바운딩 박스 데이터 저장 (없으면 None)
        self.latest_bbox = None
        self.bbox_size_window = []  # 최근 10개의 bbox area를 저장하는 리스트

        """
        5. Error plot variables
        """
        plt.ion()  # 인터랙티브 모드 활성화
        self.fig, self.ax = plt.subplots(3, 1, figsize=(6, 5))
        self.ax[0].set_title('Forward Error')
        self.ax[1].set_title('Lateral Error')
        self.ax[2].set_title('Vertical Error')
        self.ax[2].set_xlabel('Time (s)')
        self.time_data = []
        self.forward_error_data = []
        self.lateral_error_data = []
        self.vertical_error_data = []
        self.tagging_start_time = None  # Tagging 상태 시작 시각 기록용
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.setGeometry(3400, 1000, 700, 450)  # (X, Y, Width, Height)

        # """
        # 5.1. Distance-Area plot variables
        # """
        # self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 5))  # 하나의 plot만 생성
        # self.ax.set_title('Bounding Box Area')
        # self.area_data = []
        # self.csv_filename = "bbox_area_log.csv"
        # self.csv_file = open(self.csv_filename, mode='w', newline='')
        # self.csv_writer = csv.writer(self.csv_file)
        # self.csv_writer.writerow(["Time (s)", "Filtered BBox Area"])  # 헤더 작성

        """
        6. Timer setup
        """
        self.offboard_heartbeat = self.create_timer(0.05, self.offboard_heartbeat_callback) # 20Hz 
        self.main_timer = self.create_timer(0.05, self.main_callback) # 20Hz

    """
    Helper Functions
    """ 
    def distance2area(self, distance):
        return 79162.49 * np.exp(-0.94 * distance) + 2034.21 # fitting result
    
    def area2distance(self, area):
        return min(-1/0.94 * math.log((area-2034.21)/79162.49), 7)

    def normalized_forward_error(self, distance, forward_error):
        dA_dd = -0.94 * 79162.49 * np.exp(-0.94 * distance) # distance2area 함수의 도함수
        normalized_error = -forward_error / dA_dd # 면적 오차를 도함수로 나눠서 거리 오차로 환산 (부호 보정)
        return normalized_error

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
            self.area_line, = self.ax.plot(self.time_data, self.area_data, label='Area', color='red')

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
        return math.asin(t)

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
                # 감지가 없을 경우 기존 데이터를 유지
                self.get_logger().warn("YOLO detection lost, using last detected bounding box.")
        except Exception as e:
            self.get_logger().error("Failed to parse bounding box: " + str(e))


    """
    Callback functions for therosgraph timers
    """
    def offboard_heartbeat_callback(self):
        """offboard heartbeat signal"""
        self.publish_offboard_control_mode(position=True)

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

        if self.state == 'Tagging':
            error_lateral = 0.0
            error_vertical = 0.0
            error_forward = 0.0

            if self.latest_bbox is not None:
                bbox_center = self.latest_bbox['center']
                filtered_bbox_area = self.get_filtered_bbox_area()
                # self.get_logger().info(f"center={bbox_center}, filtered area={filtered_bbox_area:.0f}")
                error_lateral = (self.image_center[0] - bbox_center[0])
                error_vertical = (self.image_center[1] - bbox_center[1]) - self.pitch/0.5615 * 240
                error_forward = self.area2distance(filtered_bbox_area) - self.desired_distance

                self.get_logger().info(f"E_lat: {error_lateral:0f}, E_ver: {error_vertical:.0f}, E_for: {error_forward:.0f}")
            else:
                self.get_logger().info("No detection received.")
                filtered_bbox_area = 0
                enu_setpoint = np.array([0, 0, 0])
                ned_setpoint = self.pos + enu_to_ned(enu_setpoint)
                new_yaw = self.yaw - 0
                self.publish_setpoint(setpoint=ned_setpoint, yaw_sp=new_yaw)

            current_time = self.get_clock().now().nanoseconds * 1e-9 - self.tagging_start_time
            self.update_error_plot(current_time, error_forward, error_lateral, error_vertical)
            # self.update_area_plot(current_time, filtered_bbox_area)

            correction_vertical = self.pid_vertical.update(error_vertical)
            correction_forward = np.clip(self.pid_forward.update(error_forward), -3, 3)
            correction_yaw = np.clip(self.pid_lateral.update(error_lateral), -0.5, 0.5)
            if correction_forward == 3 and correction_vertical > 1: correction_vertical = 5

            print(f"ver_cor:{correction_vertical:0f}, for_cor:{correction_forward:0f}, lat_cor:{math.degrees(correction_yaw):0f}")

            new_yaw = self.yaw - correction_yaw
            # new_yaw = self.yaw

            yaw_rad = np.radians(self.yaw)
            cos_yaw = np.cos(yaw_rad)
            sin_yaw = np.sin(yaw_rad)

            x_enu = cos_yaw * correction_forward - sin_yaw * 0
            y_enu = sin_yaw * correction_forward + cos_yaw * 0

            local_enu_setpoint = np.array([x_enu, y_enu, correction_vertical])
            # local_enu_setpoint = np.array([x_enu, y_enu, 0])
            local_ned_setpoint = self.pos + enu_to_ned(local_enu_setpoint)
            self.publish_setpoint(setpoint=local_ned_setpoint, yaw_sp=new_yaw)
            # self.publish_local2global_setpoint(local_setpoint=np.array([0, 0, -5]))

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
        print("KeyboardInterrupt received. Exiting and keeping plot window open...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.show(block=True)  # 프로그램 종료 후에도 plot 창 유지

if __name__ == '__main__':
    main()