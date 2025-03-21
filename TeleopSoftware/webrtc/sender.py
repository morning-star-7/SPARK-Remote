import asyncio
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.signaling import TcpSocketSignaling
from av import VideoFrame
import fractions
from datetime import datetime
import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CustomVideoStreamTrack(VideoStreamTrack):
    def __init__(self, camera_id, node):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_id)
        self.frame_count = 0
        self.node = node
        self.bridge = CvBridge()
        self.pub_rgb = self.node.create_publisher(Image, 'video_frames', 10)

    async def recv(self):
        self.frame_count += 1
        print(f"Sending frame {self.frame_count}")
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame from camera")
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, 30)  # Use fractions for time_base

        # Publish frame to ROS 2 topic
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="rgb8")
        self.pub_rgb.publish(msg)

        return video_frame

async def setup_webrtc_and_run(ip_address, port, camera_id, node):
    signaling = TcpSocketSignaling(ip_address, port)
    pc = RTCPeerConnection()
    video_sender = CustomVideoStreamTrack(camera_id, node)
    pc.addTrack(video_sender)

    try:
        await signaling.connect()

        @pc.on("datachannel")
        def on_datachannel(channel):
            print(f"Data channel established: {channel.label}")

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state is {pc.connectionState}")
            if pc.connectionState == "connected":
                print("WebRTC connection established successfully")

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await signaling.send(pc.localDescription)

        while True:
            obj = await signaling.receive()
            if isinstance(obj, RTCSessionDescription):
                await pc.setRemoteDescription(obj)
                print("Remote description set")
            elif obj is None:
                print("Signaling ended")
                break
        print("Closing connection")
    finally:
        await pc.close()

async def main(ip_address, port, camera_id):
    rclpy.init()
    node = rclpy.create_node('video_streamer')
    await setup_webrtc_and_run(ip_address, port, camera_id, node)
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Receiver")
    parser.add_argument("--video-dev", type=int, default=0, help="Video device number")
    args = parser.parse_args()

    ip_address = "192.168.0.26"  # Ip Address of Remote Server/Machine
    port = 9999 + args.video_dev
    camera_id = args.video_dev

    asyncio.run(main(ip_address, port, camera_id))
