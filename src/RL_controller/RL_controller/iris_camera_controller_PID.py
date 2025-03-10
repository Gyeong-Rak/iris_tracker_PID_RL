#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import numpy as np
import ast
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# 메시지 타입
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import OffboardControlMode
from std_msgs.msg import String

# 간단한 PID 제어기 클래스
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
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        """
        1. Create Subscribers
        """
        self.create_subscription(String, '/yolov8/bounding_boxes', self.bbox_callback, qos_profile)

        """
        2. Create Publishers
        """
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/px4_1/fmu/in/trajectory_setpoint', qos_profile
        )
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/px4_1/fmu/in/offboard_control_mode', qos_profile
        )

        """
        3. Timer setup
        """
        self.control_timer = self.create_timer(0.1, self.control_callback)

        # 기준(홈) 위치 설정 (ENU 좌표계로 설정; 예를 들어 카메라가 10m 위에 있다고 가정)
        self.home_position = np.array([0.0, 0.0, 10.0])
        self.current_setpoint = self.home_position.copy()

        # 원하는 바운딩 박스 조건
        self.desired_bbox_area = 1500.0
        self.image_center = np.array([320.0, 240.0])  # 640x480 이미지의 중심

        # PID 컨트롤러 초기화 (게인 값은 필요에 따라 조정)
        self.dt = 0.1  # 타이머 주기
        self.pid_forward = PID(Kp=0.0005, Ki=0.0, Kd=0.0001, dt=self.dt)   # 전진/후진 제어 (면적 오차)
        self.pid_lateral = PID(Kp=0.005, Ki=0.0, Kd=0.0005, dt=self.dt)    # 좌우 제어 (수평 오차)

        # 최신 바운딩 박스 데이터 저장 (없으면 None)
        self.latest_bbox = None

    def bbox_callback(self, msg):
        """
        /yolov8/bounding_boxes 메시지 (문자열)를 파싱하여,
        검출된 바운딩 박스의 중심 좌표와 면적을 저장합니다.
        형식: '[[x1, y1, x2, y2]]' 또는 '[]'
        """
        try:
            bbox_list = ast.literal_eval(msg.data)
            if isinstance(bbox_list, list) and len(bbox_list) > 0:
                # 첫 번째 박스를 사용
                bbox = bbox_list[0]
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
                    area = (x2 - x1) * (y2 - y1)
                    self.latest_bbox = {'center': center, 'area': area}
                    self.get_logger().info(f"Received bbox: center={center}, area={area:.2f}")
                else:
                    self.latest_bbox = None
            else:
                self.latest_bbox = None
        except Exception as e:
            self.get_logger().error("Failed to parse bounding box: " + str(e))
            self.latest_bbox = None

    def control_callback(self):
        # 기본 오차 (검출 없으면 0)
        error_lateral = 0.0
        error_forward = 0.0

        if self.latest_bbox is not None:
            bbox_center = self.latest_bbox['center']
            bbox_area = self.latest_bbox['area']
            # 수평 오차: 카메라 이미지 중심과 검출된 바운딩 박스 중심의 x 좌표 차이
            error_lateral = self.image_center[0] - bbox_center[0]
            # 면적 오차: 원하는 면적과 실제 면적의 차이 (전진/후진 제어)
            error_forward = self.desired_bbox_area - bbox_area

            self.get_logger().info(f"Error lateral: {error_lateral:.2f}, Error forward: {error_forward:.2f}")
        else:
            self.get_logger().info("No detection received.")

        # PID 업데이트
        correction_forward = self.pid_forward.update(error_forward)
        correction_lateral = self.pid_lateral.update(error_lateral)

        # 새로운 setpoint 계산 (ENU 좌표계에서 계산: x: 전진/후진, y: 좌우 이동, z: 고도는 고정)
        enu_setpoint = self.home_position + np.array([correction_forward, correction_lateral, 0.0])
        # ENU -> NED 변환 (PX4는 NED 좌표계를 사용)
        ned_setpoint = enu_to_ned(enu_setpoint)
        self.publish_trajectory_setpoint(ned_setpoint)
        # offboard heartbeat도 함께 발행
        self.publish_offboard_control_mode(position=True)

    def publish_trajectory_setpoint(self, position_sp):
        msg = TrajectorySetpoint()
        msg.position = list(position_sp)
        msg.velocity = list(np.nan * np.zeros(3))
        msg.yaw = float('nan')
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Published setpoint (NED): {position_sp}")

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

def main(args=None):
    rclpy.init(args=args)
    node = IrisCameraController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
