import socket
import struct

# Define the UDP IP and Port
UDP_IP = "127.0.0.1"  # Listen on all available network interfaces
UDP_PORT = 5000  # Ensure this matches the sender's port

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for UDP packets on port {UDP_PORT}...")

while True:
    # Receive data (buff
    # er size 1024 bytes)
    data, addr = sock.recvfrom(1024)

    if len(data) == 5:  # Ensure the received packet size is correct
        # Unpack 4-byte float and 1-byte signed integer
        c_offset_m, direction_value = struct.unpack("!f b", data)

        # Convert direction byte to string
        direction = "right" if direction_value == 1 else "left"

        print(data)

        print(f"Received from {addr}: Offset = {c_offset_m:.4f} meters, Direction = {direction}")
    else:
        print("Received incorrect data size.")


#Path - Documents/Autonomous_Vehicle/YOLOV8/image-segmentation-yolov8-main/LaneCalibration/LateralOffsetFinal
