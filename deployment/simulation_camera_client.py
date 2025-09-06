import zmq
from dataclasses import dataclass
import numpy as np
from typing import Optional
import logging
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CameraConfig:
    server_address: str = "tcp://localhost:5555"
    timeout: int = 1000
    target_width: int = 640
    target_height: int = 480

class HeadCameraClient:
    """Handles head camera image acquisition via ZMQ"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(config.server_address)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, config.timeout)
        logger.info(f"Camera client connected to {config.server_address}")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from head camera
        Returns: RGB numpy array of shape (480, 640, 3), or None if failed
        """
        try:
            latest_message = self._get_latest_message()
            if latest_message is None:
                return None

            return self._decode_image(latest_message)

        except zmq.Again:
            logger.warning("Camera frame timeout")
            return None
        except Exception as e:
            logger.error(f"Error receiving camera frame: {e}")
            return None

    def _get_latest_message(self) -> Optional[bytes]:
        """Get the most recent message from the socket"""
        latest_message = None

        # Try to get the latest message (non-blocking)
        for _ in range(5):
            try:
                message = self.socket.recv(zmq.NOBLOCK)
                latest_message = message
            except zmq.Again:
                break

        # If no non-blocking message, try one blocking call
        if latest_message is None:
            latest_message = self.socket.recv()

        return latest_message

    def _decode_image(self, message: bytes) -> Optional[np.ndarray]:
        """Decode image message and extract head camera portion"""
        nparr = np.frombuffer(message, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return None

        # Extract head camera (leftmost third)
        height, width = img_bgr.shape[:2]
        head_width = width // 3
        head_camera = img_bgr[:, 0:head_width]

        # Resize and convert to RGB
        img_bgr = cv2.resize(head_camera, (self.config.target_width, self.config.target_height))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        return img_rgb

    def close(self):
        """Clean up ZMQ connection"""
        self.socket.close()
        self.context.term()
