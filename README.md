# Lane Detection and Lateral Offset Estimation for Autonomous Vehicles  

## Description  

This project provides a **real-time lane detection and lateral offset estimation system** for autonomous driving applications. It leverages **YOLOv8 segmentation**, **9-point calibration**, and **UDP communication** to estimate the vehicle's lateral position with centimeter-level accuracy. The system is designed to run on **NVIDIA Jetson AGX Orin**, utilizing **CUDA acceleration** for efficient deep learning inference.  

### Key Features  
- **Lane Segmentation using YOLOv8** – Detects lane boundaries in real-time.  
- **Lateral Offset Calculation** – Uses **regression lines and intersection points** to determine the vehicle's position.  
- **UDP-Based Communication** – Sends offset values to an **Autonomy Core** for steering control.  
- **Error Correction** – Applies bias compensation and scaling adjustments for accurate lane tracking.  
- **Real-Time Processing** – Runs at **30 FPS** for dynamic driving scenarios.  

## Interesting Techniques  

- **YOLOv8-based segmentation** for lane detection.  
- **Nine-point regression-based calibration** to convert image coordinates to real-world distances.  
- **Error mitigation strategies**, including **bias and scaling error adjustments**.  
- **Multi-line reference approach** to improve lateral offset accuracy.  
- **Low-latency UDP communication** for real-time vehicle control.  

## Technologies & Libraries  

This project utilizes the following technologies:  

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) – for lane segmentation.  
- [OpenCV](https://opencv.org/) – for image processing and contour detection.  
- [PyTorch](https://pytorch.org/) – for running deep learning inference on **Jetson AGX Orin**.  
- [Numpy](https://numpy.org/) – for numerical operations (e.g., regression and intersection calculations).  
- [PyYAML](https://pyyaml.org/) – for configuration file management.  
- [Socket](https://docs.python.org/3/library/socket.html) – for UDP communication.  
- [Struct](https://docs.python.org/3/library/struct.html) – for binary data packing before transmission.  

## Project Structure  

```plaintext  
/  
│── config.yaml              # Configuration file for model paths, UDP settings, and calibration  
│── requirements.txt         # List of required Python libraries  
│── yolomodelweights.pt      # Pretrained YOLOv8 model weights for lane segmentation  
│── latoffsetmain.py         # Main script for lane detection and lateral offset estimation  
│── UDPListenTest.py         # Script to test UDP data reception  
│── README.md                # Project documentation  
```

### Key Files  

- **`latoffsetmain.py`** – The main script that processes images, detects lanes, estimates offset, and sends data via UDP.  
- **`UDPListenTest.py`** – A test script to verify UDP communication for real-time vehicle control.  
- **`yolomodelweights.pt`** – The YOLOv8 lane segmentation model used for inference.  
- **`config.yaml`** – Stores paths, calibration data, and communication settings.  
- **`zed2irec.mp4`** – Sample ZED 2i camera footage for testing the pipeline.  
