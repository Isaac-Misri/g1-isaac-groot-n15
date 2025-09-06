"""
ZMQ Image Client - Receives and displays images from the ImageServer
Modified to match the server protocol from the original ImageClient
"""
import cv2
import zmq
import numpy as np
import time
import struct
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageClient:
    def __init__(self, server_address="192.168.123.164", port=5555, image_show=True, Unit_Test=False):
        """
        Initialize ZMQ client to receive images

        Args:
            server_address: IP address of the image server
            port: Port number to connect to
            image_show: Whether to display images
            Unit_Test: Enable performance evaluation
        """
        self.running = True
        self._image_show = image_show
        self._server_address = server_address
        self._port = port
        self._enable_performance_eval = Unit_Test

        # Performance metrics (if enabled)
        if self._enable_performance_eval:
            self._frame_count = 0
            self._last_frame_id = -1
            self._lost_frames = 0
            self._total_frames = 0
            self._start_time = time.time()

        logger.info(f"[Image Client] Connecting to image server at {server_address}:{port}")

        # Set up ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{self._server_address}:{self._port}")

        # Subscribe to all messages (empty string means subscribe to everything)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        logger.info("[Image Client] Connected and ready to receive images")

    def _update_performance_metrics(self, timestamp, frame_id, receive_time):
        """Update performance metrics for unit testing"""
        # Calculate latency
        latency = receive_time - timestamp

        # Check for lost frames
        expected_frame_id = self._last_frame_id + 1 if self._last_frame_id != -1 else frame_id
        if frame_id != expected_frame_id:
            lost = frame_id - expected_frame_id
            if lost < 0:
                logger.info(f"[Image Client] Received out-of-order frame ID: {frame_id}")
            else:
                self._lost_frames += lost
                logger.warning(f"[Image Client] Detected lost frames: {lost}, Expected: {expected_frame_id}, Received: {frame_id}")

        self._last_frame_id = frame_id
        self._total_frames = frame_id + 1
        self._frame_count += 1

    def _print_performance_metrics(self):
        """Print performance metrics every 30 frames"""
        if self._frame_count % 30 == 0:
            elapsed_time = time.time() - self._start_time
            fps = self._frame_count / elapsed_time if elapsed_time > 0 else 0
            lost_frame_rate = (self._lost_frames / self._total_frames) * 100 if self._total_frames > 0 else 0

            logger.info(f"[Image Client] FPS: {fps:.2f}, Total Frames: {self._frame_count}, "
                       f"Lost Frame Rate: {lost_frame_rate:.2f}%")

    def receive_and_display(self):
        """
        Continuously receive and display images
        """
        logger.info("[Image Client] Starting to receive images... Press 'q' or ESC to quit")

        try:
            while self.running:
                try:
                    # Receive message
                    message = self.socket.recv()
                    receive_time = time.time()

                    # Handle message format based on whether Unit_Test is enabled
                    if self._enable_performance_eval:
                        # Message has header with timestamp and frame_id
                        header_size = struct.calcsize('dI')  # double + unsigned int
                        try:
                            # Extract header and image data
                            header = message[:header_size]
                            jpg_bytes = message[header_size:]
                            timestamp, frame_id = struct.unpack('dI', header)

                            # Update performance metrics
                            self._update_performance_metrics(timestamp, frame_id, receive_time)
                            self._print_performance_metrics()

                        except struct.error as e:
                            logger.warning(f"[Image Client] Error unpacking header: {e}, discarding message.")
                            continue
                    else:
                        # No header, entire message is image data
                        jpg_bytes = message

                    # Decode JPEG image
                    nparr = np.frombuffer(jpg_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is None:
                        logger.warning("[Image Client] Failed to decode image.")
                        continue

                    if self._image_show:
                        # The original code suggests the image might be concatenated (tv + wrist cameras)
                        # Display the full image, but resize it to fit on screen
                        height, width = img.shape[:2]

                        # If the image is very wide (concatenated cameras), resize appropriately
                        if width > height * 2:  # Likely concatenated image
                            # Resize to reasonable display size
                            display_width = min(1280, width // 2)
                            display_height = int(height * display_width / width)
                            resized_image = cv2.resize(img, (display_width, display_height))
                            cv2.imshow('Image Client Stream (Concatenated Cameras)', resized_image)
                        else:
                            # Single camera or already reasonable size
                            if width > 1280 or height > 720:
                                # Resize large images
                                resized_image = cv2.resize(img, (width // 2, height // 2))
                                cv2.imshow('Image Client Stream', resized_image)
                            else:
                                cv2.imshow('Image Client Stream', img)

                        # Check for quit key
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' or ESC
                            logger.info("[Image Client] Quit key pressed")
                            self.running = False
                            break

                except zmq.Again:
                    # This shouldn't happen since we didn't set a timeout, but handle it anyway
                    logger.info("[Image Client] No message received (timeout)")
                    continue

        except KeyboardInterrupt:
            logger.info("\n[Image Client] Interrupted by user")
        except Exception as e:
            logger.error(f"[Image Client] An error occurred while receiving data: {e}")
        finally:
            self.close()

    def close(self):
        """
        Clean up resources
        """
        if self._image_show:
            cv2.destroyAllWindows()
        self.socket.close()
        self.context.term()
        logger.info("[Image Client] Client closed")

if __name__ == "__main__":
    # Example usage - matches the original code's examples

    # For deployment test (matches the original code's final example)
    client = ImageClient(
        server_address='192.168.123.164',
        port=5555,
        image_show=True,
        Unit_Test=False
    )

    # For local testing with performance evaluation, uncomment below:
    # client = ImageClient(
    #     server_address='127.0.0.1',
    #     port=5555,
    #     image_show=True,
    #     Unit_Test=True
    # )

    client.receive_and_display()
