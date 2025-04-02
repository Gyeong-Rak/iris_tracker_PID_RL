#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import numpy as np
import os, random
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# import px4_msgs
"""msgs for subscription"""
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleGlobalPosition
"""msgs for publishing"""
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint

class iris_controller(Node):
    def __init__(self):
        super().__init__('iris_controller')

        """
        0. Configure QoS profile for publishing and subscribing
        """
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        """
        1. Create Subscribers
        """
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/px4_2/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile
        )
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/px4_2/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile
        )
        self.vehicle_global_position_subscriber = self.create_subscription(
            VehicleGlobalPosition, '/px4_2/fmu/out/vehicle_global_position', self.vehicle_global_position_callback, qos_profile
        )

        """
        2. Create Publishers
        """
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/px4_2/fmu/in/vehicle_command', qos_profile
        )
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/px4_2/fmu/in/offboard_control_mode', qos_profile
        )
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/px4_2/fmu/in/trajectory_setpoint', qos_profile
        )

        """
        3. State variables
        """
        # Vehicle status
        self.vehicle_status = VehicleStatus()

        # Vehicle position, velocity, and yaw
        self.pos = np.array([0.0, 0.0, 0.0])
        self.pos_gps = np.array([0.0, 0.0, 0.0]) 
        self.vel = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        self.home_position = np.array([0.0, 0.0, 0.0])
        self.home_position_gps = np.array([0.0, 0.0, 0.0])  # Store initial GPS position
        self.get_position_flag = False

        self.state = 'ready2flight' # ready2flight -> takeoff -> Running

        """
        5. timer setup
        """
        self.offboard_heartbeat = self.create_timer(0.05, self.offboard_heartbeat_callback) # 20Hz 
        self.main_timer = self.create_timer(0.05, self.main_callback) # 20Hz

    """
    Helper Functions
    """
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
    
    """
    Callback functions for the timers
    """
    def offboard_heartbeat_callback(self):
        """offboard heartbeat signal"""
        self.publish_offboard_control_mode(position=True)

    def main_callback(self):
        if self.state == 'ready2flight':
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                    print("Iris Arming...")
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0, target_system=3)
                else:
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, param7=10.0, target_system=3)
            elif self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                if not self.get_position_flag:
                    print("Waiting for position data")
                    return
                self.home_position = self.set_home_position()
                self.get_logger().info(f"Home position set to: [{self.home_position[0]:.2f}, {self.home_position[1]:.2f}, {self.home_position[2]:.2f}]")
                print("Iris Taking off...")
                self.state = 'takeoff'


        if self.state == 'takeoff':
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                print("Run~~ ^_^")
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0, target_system=3)
                self.start_time = self.get_clock().now().nanoseconds * 1e-9
                self.state = 'Running'

        if self.state == 'Running':
            # self.circle_trajectory(5, 0.4) # radius, omega
            # self.vertical_oscillation_trajectory(1.5, 0.4) # amplitude, vertical_speed
            # self.xy_helical_trajectory(5, 0.4, 1.5, 0.6) # radius, omega, amplitude, vertical_speed
            # self.forward_trajectory(3, 0.4) # amplitude, speed
            # self.distance_measure_trajectory()
            # self.xyz_helical_trajectory(3, 0.2, 1) # radius, omega, speed
            self.random_waypoint_trajectory()

    """
    Callback functions for subscribers.
    """
    def vehicle_status_callback(self, msg):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = msg

    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position = msg
        self.pos = np.array([msg.x, msg.y, msg.z])
        self.vel = np.array([msg.vx, msg.vy, msg.vz])
        self.yaw = msg.heading

    def vehicle_global_position_callback(self, msg):
        self.get_position_flag = True
        self.vehicle_global_position = msg
        self.pos_gps = np.array([msg.lat, msg.lon, msg.alt])

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

    """
    Trejectory functions
    """
    def circle_trajectory(self, radius, omega): # validate lateral control
        now = self.get_clock().now().nanoseconds * 1e-9
        t = now - self.start_time
        center_x, center_y, center_z = 0.0, 0.0, 5.0
        theta = omega * t

        x_ned = center_y + radius * math.sin(theta)
        y_ned = center_x + radius * math.cos(theta)
        z_ned = -center_z

        self.publish_local2global_setpoint(local_setpoint=np.array([x_ned, y_ned, z_ned]))

    def vertical_oscillation_trajectory(self, amplitude, vertical_speed): # validate vertical control
        now = self.get_clock().now().nanoseconds * 1e-9
        t = now - self.start_time
        center_altitude = 5.0
        z_offset = amplitude * math.sin(vertical_speed * t)

        x_ned = 0.0
        y_ned = 5.0
        z_ned = -(center_altitude + z_offset)
        
        self.publish_local2global_setpoint(local_setpoint=np.array([x_ned, y_ned, z_ned]))

    def xy_helical_trajectory(self, radius, omega, amplitude, vertical_speed): # validate lateral + vertical control
        now = self.get_clock().now().nanoseconds * 1e-9
        t = now - self.start_time
        center_x, center_y, center_z = 0.0, 0.0, 5.0
        theta = omega * t
        z_offset = amplitude * math.sin(vertical_speed * t)
        
        x_ned = center_y + radius * math.sin(theta)
        y_ned = center_x + radius * math.cos(theta)
        z_ned = -(center_z + z_offset)

        self.publish_local2global_setpoint(local_setpoint=np.array([x_ned, y_ned, z_ned]))

    def forward_trajectory(self, amplitude, speed):
        now = self.get_clock().now().nanoseconds * 1e-9
        t = now - self.start_time
        x_offset = amplitude * math.sin(speed * t)

        x_ned = 0.0
        y_ned = 5 + x_offset
        z_ned = -5.0

        self.publish_local2global_setpoint(local_setpoint=np.array([x_ned, y_ned, z_ned]))

    def distance_measure_trajectory(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        t = now - self.start_time - 20
        if t<10:
            print("1")
            self.publish_local2global_setpoint(local_setpoint=np.array([0, 1, -5]))
        elif t<20:
            print("2")
            self.publish_local2global_setpoint(local_setpoint=np.array([0, 2, -5]))
        elif t<30:
            print("3")
            self.publish_local2global_setpoint(local_setpoint=np.array([0, 3, -5]))
        elif t<40:
            print("4")
            self.publish_local2global_setpoint(local_setpoint=np.array([0, 4, -5]))
        elif t<50:
            print("5")
            self.publish_local2global_setpoint(local_setpoint=np.array([0, 5, -5]))
        elif t<60:
            print("6")
            self.publish_local2global_setpoint(local_setpoint=np.array([0, 6, -5]))
        elif t<70:
            print("7")
            self.publish_local2global_setpoint(local_setpoint=np.array([0, 7, -5]))
        elif t<80:
            print("8")
            self.publish_local2global_setpoint(local_setpoint=np.array([0, 8, -5]))
        elif t<90:
            print("9")
            self.publish_local2global_setpoint(local_setpoint=np.array([0, 9, -5]))
        elif t<100:
            print("10")
            self.publish_local2global_setpoint(local_setpoint=np.array([0, 10, -5]))

    def xyz_helical_trajectory(self, radius, omega, speed):
        now = self.get_clock().now().nanoseconds * 1e-9
        t = now - self.start_time
        center_x, center_y, center_z = 0.0, 5.0, 10.0
        theta = omega * t

        x_ned = center_x + radius * math.cos(theta)
        y_ned = center_y + t * speed
        z_ned = -(center_z + radius * math.sin(theta))

        self.publish_local2global_setpoint(local_setpoint=np.array([x_ned, y_ned, z_ned]))

    def random_waypoint_trajectory(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        t = now - self.start_time

        if t < 10: return # wait for iris_camera to be ready
        
        if not hasattr(self, 'next_waypoint_time') or t >= self.next_waypoint_time:
            self.next_waypoint_time = t + 0.01
            speed = 0.2
            if hasattr(self, 'current_waypoint'):
                delta = np.array([
                    random.uniform(-speed, speed),  # dx
                    random.uniform(-speed, speed),  # dy
                    random.uniform(-speed, speed)   # dz
                ])

                new_waypoint = self.current_waypoint + delta
                self.current_waypoint = np.clip(
                    new_waypoint,
                    [-5.0, -5.0, -10.5],  # 최소값 (x, y, z)
                    [5.0, 5.0, -9.5]      # 최대값 (x, y, z)
                )
            else:
                self.current_waypoint = self.pos
            
            print(f"setpoint: [{self.current_waypoint[0]:.2f}, {self.current_waypoint[1]:.2f}, {self.current_waypoint[2]:.2f}]")
        
        # 현재 웨이포인트로 이동
        if hasattr(self, 'current_waypoint'):
            self.publish_local2global_setpoint(local_setpoint=self.current_waypoint)

def main(args=None):
    rclpy.init(args=args)
    node = iris_controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()