import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32
from cv_bridge import CvBridge
import csv
import time
import numpy as np
import torch
import ast
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
# export PYTHONPATH=$PYTHONPATH:/home/gr/FastSAM
from fastsam import FastSAM, FastSAMPrompt

class FastSAMNode(Node):
    def __init__(self):
        super().__init__('fastsam_segmentation_node')

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
        self.image_subscription = self.create_subscription(
            Image, 'iris_camera/camera/image_raw', self.image_callback, qos_profile
        )
        self.bbox_subscription = self.create_subscription(
            String, '/YOLOv8/bounding_boxes', self.bbox_callback, qos_profile
        )        

        """
        2. Create Publishers
        """
        self.mask_publisher = self.create_publisher(
            Image, '/FastSAM/object_mask', qos_profile
        )
        self.pixel_count_publisher = self.create_publisher(
            Int32, '/FastSAM/object_pixel_count', qos_profile
        )

        """
        3. model & global variables
        """
        self.bridge = CvBridge()
        self.model = FastSAM('./weights/FastSAM-s.pt')
        self.bbox_prompt = None
        self.latest_image = None

        # """
        # 4. CSV variables
        # """
        # self.csv_filename = "pixel_count_log.csv"
        # self.csv_file = open(self.csv_filename, mode='w', newline='')
        # self.csv_writer = csv.writer(self.csv_file)
        # self.csv_writer.writerow(["Time (s)", "Pixel count"])
        
        # # I/O 부담을 줄이기 위한 버퍼
        # self.data_buffer = []
        # # 0.5초마다 버퍼 -> CSV 기록
        # self.csv_timer = self.create_timer(0.5, self.csv_flush_callback)
        
    """
    5. Callback Functions
    """

    def csv_flush_callback(self):
        if len(self.data_buffer) > 0:
            self.csv_writer.writerows(self.data_buffer)
            self.csv_file.flush()
            self.data_buffer.clear()

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error("이미지 변환 실패: " + str(e))
            return

        self.latest_image = cv_image

        # bbox_prompt가 아직 없으면 스킵
        if self.bbox_prompt is None:
            return

        try:
            results = self.model(cv_image, retina_masks=True, imgsz=640, conf=0.4, iou=0.9)
            prompt_process = FastSAMPrompt(cv_image, results, device='cuda' if torch.cuda.is_available() else 'cpu')
            masks = prompt_process.box_prompt(bboxes=[self.bbox_prompt])

            if masks is not None and len(masks) > 0:
                mask = (masks[0] * 255).astype(np.uint8)

                # 1) 마스크 퍼블리시
                mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
                self.mask_publisher.publish(mask_msg)

                # 2) 픽셀 수 계산 및 퍼블리시
                pixel_count = int(np.count_nonzero(mask))
                pixel_msg = Int32()
                pixel_msg.data = pixel_count
                self.pixel_count_publisher.publish(pixel_msg)

                # current_time = time.time()
                # self.data_buffer.append([current_time, pixel_count])

                print(f"Mask pixel count: {pixel_count}")

        except Exception as e:
            self.get_logger().error("FastSAM Inference Fail: " + str(e))

    def bbox_callback(self, msg):
        try:
            bbox_list = ast.literal_eval(msg.data)
            if isinstance(bbox_list, list) and len(bbox_list) > 0:
                bbox = bbox_list[0]
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = [int(x) for x in bbox[:4]]
                    w, h = x2 - x1, y2 - y1

                    if self.latest_image is not None:
                        frame_h, frame_w = self.latest_image.shape[:2]
                        bbox_area = w * h
                        frame_area = frame_w * frame_h
                        area_ratio = bbox_area / frame_area

                        if area_ratio > 0.8:
                            self.bbox_prompt = None
                            return

                    self.bbox_prompt = [x1, y1, x2, y2]
        except Exception as e:
            self.get_logger().error("Failed to parse bounding box: " + str(e))

    def destroy_node(self):
        # self.csv_flush_callback()
        # self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FastSAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
