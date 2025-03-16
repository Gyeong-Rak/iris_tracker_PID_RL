import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('YOLOv8_detector_node')

        """
        0. Configure QoS profile for publishing and subscribing
        """
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        """
        1. Create Subscribers
        """
        self.subscription = self.create_subscription(
            Image, 'iris_camera/camera/image_raw', self.data_callback, qos_profile
        )

        """
        2. Create Publishers
        """
        self.publisher = self.create_publisher(
            Image, '/yolov8/detection_image', qos_profile
        )
        self.bbox_publisher = self.create_publisher(
            String, '/yolov8/bounding_boxes', qos_profile
        )

        self.bridge = CvBridge()
        self.model = YOLO('/home/gr/iris_tracker_PID_RL/data/best.pt')

    def data_callback(self, msg):
        try:
            # ROS 이미지 메시지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error('이미지 변환 오류: ' + str(e))
            return

        # YOLOv8로 객체 검출 수행 (비동기 처리나 스레드 사용 시 성능 개선 가능)
        results = self.model(cv_image)

        # 첫 번째 결과에 대해 bounding box와 label을 포함한 이미지를 생성
        annotated_frame = results[0].plot()

        # 검출된 이미지를 ROS 이미지 메시지로 변환 후 publish
        detection_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.publisher.publish(detection_msg)

        # 검출된 바운딩 박스 좌표 추출 (각 박스: [x1, y1, x2, y2])
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            # 텐서를 numpy 배열로 변환 후 리스트로 변경
            bboxes = boxes.xyxy.cpu().numpy().tolist()
        else:
            bboxes = []

        # 바운딩 박스 좌표를 문자열로 변환하여 publish
        bbox_str = str(bboxes)
        bbox_msg = String()
        bbox_msg.data = bbox_str
        self.bbox_publisher.publish(bbox_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
