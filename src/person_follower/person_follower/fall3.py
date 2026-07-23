import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from ultralytics import YOLO


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "fall_unified_yolo26m_v1.pt"
)
FALLEN_CLASS_ID = 1


class FallDetectionNode(Node):
    def __init__(self):
        super().__init__("fall_detection_node")

        self.declare_parameter("use_compressed", True)
        self.declare_parameter("image_topic_compressed", "/camera/image_raw/compressed")
        self.declare_parameter("image_topic_raw", "/camera/image_raw")
        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("conf_threshold", 0.5)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "auto")
        self.declare_parameter("publish_debug", True)

        use_compressed = bool(self.get_parameter("use_compressed").value)
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        device_param = str(self.get_parameter("device").value)
        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        model_path = str(self.get_parameter("model_path").value)

        if device_param == "auto":
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = int(device_param) if device_param.isdigit() else device_param
        self.half = self.device != "cpu"

        self.get_logger().info(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        try:
            self.model.to(self.device if self.device != "cpu" else "cpu")
        except Exception as e:
            self.get_logger().warning(f"model.to({self.device}) failed: {e}; falling back to cpu")
            self.device = "cpu"
            self.half = False
        self.class_names = getattr(self.model, "names", {}) or {}
        self.get_logger().info(
            f"Model classes: {self.class_names} | device={self.device} | half={self.half}"
        )

        self.bridge = CvBridge()

        if use_compressed:
            topic = self.get_parameter("image_topic_compressed").value
            self.subscription = self.create_subscription(
                CompressedImage, topic, self.image_callback_compressed, 10
            )
        else:
            topic = self.get_parameter("image_topic_raw").value
            self.subscription = self.create_subscription(
                Image, topic, self.image_callback_raw, 10
            )

        self.fall_pub = self.create_publisher(String, "/Nour_fall_detection", 10)
        self.det_pub = self.create_publisher(
            Detection2DArray, "/fall_detection/detections", 10
        )

        debug_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.debug_pub = self.create_publisher(
            CompressedImage, "/fall_detection/debug/compressed", debug_qos
        )

        self.get_logger().info(
            f"Fall detector listening on {topic} ({'compressed' if use_compressed else 'raw'})"
        )

    def image_callback_compressed(self, msg: CompressedImage):
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        self._process(frame, msg.header)

    def image_callback_raw(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warning(f"cv_bridge failed: {e}")
            return
        self._process(frame, msg.header)

    def _process(self, frame: np.ndarray, header):
        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        det_array = Detection2DArray()
        det_array.header = header

        fall_in_frame = False

        if results:
            r0 = results[0]
            boxes = getattr(r0, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                    det = Detection2D()
                    det.header = header
                    det.bbox.center.position.x = float((x1 + x2) * 0.5)
                    det.bbox.center.position.y = float((y1 + y2) * 0.5)
                    det.bbox.center.theta = 0.0
                    det.bbox.size_x = float(x2 - x1)
                    det.bbox.size_y = float(y2 - y1)
                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = str(int(cls))
                    hyp.hypothesis.score = float(conf)
                    det.results.append(hyp)
                    det_array.detections.append(det)

                    if int(cls) == FALLEN_CLASS_ID and float(conf) >= self.conf_threshold:
                        fall_in_frame = True

                if self.publish_debug:
                    self._publish_debug_frame(frame, header, xyxy, confs, clss)
            elif self.publish_debug:
                self._publish_debug_frame(frame, header, None, None, None)

        self.det_pub.publish(det_array)

        msg = String()
        msg.data = "FALL DETECTED" if fall_in_frame else "NO FALL DETECTED"
        self.fall_pub.publish(msg)

    def _publish_debug_frame(self, frame, header, xyxy, confs, clss):
        annotated = frame.copy()
        if xyxy is not None:
            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                cls_i = int(cls)
                name = self.class_names.get(cls_i, str(cls_i))
                color = (0, 0, 255) if cls_i == FALLEN_CLASS_ID else (0, 255, 0)
                cv2.rectangle(
                    annotated,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2,
                )
                label = f"{name} {conf:.2f}"
                cv2.putText(
                    annotated,
                    label,
                    (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        ok, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ok:
            return
        out = CompressedImage()
        out.header = header
        out.format = "jpeg"
        out.data = jpg.tobytes()
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = FallDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
