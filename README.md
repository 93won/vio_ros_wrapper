# Lightweight VIO ROS2 Wrapper

ROS2 wrapper for lightweight Visual-Inertial Odometry system.

Supports **Stereo VIO/VO** and **RGBD VIO/VO** modes.

---

## Requirements

- Ubuntu 22.04
- ROS2 Humble

---

## Installation

### 1. Create Workspace

```bash
mkdir -p <your_path>/vio_ws/src
cd <your_path>/vio_ws/src
git clone https://github.com/93won/vio_ros_wrapper.git
cd vio_ros_wrapper
git submodule update --init --recursive
cd <your_path>/vio_ws
```

### 2. Build

```bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

---

## Sample Datasets

Pre-recorded ROS2 bags for testing:

### 1. OpenLORIS RGBD Dataset (Cafe)
- **File**: `cafe.bag.db3`
- **Description**: Indoor cafe scene with RealSense D435i
- **Topics**: RGB + Depth images
- **Download**: [Google Drive](https://drive.google.com/file/d/1hH-n9-Movdq4sZkGATvNCcl27lDOYep5/view?usp=sharing)

### 2. TUM-VI Dataset (Room1)
- **File**: `room1_stereo_imu.bag_0.db3`
- **Description**: Indoor room with fisheye stereo camera + IMU
- **Topics**: Stereo images + IMU data
- **Download**: [Google Drive](https://drive.google.com/file/d/1-_y33_YkB8pTDkro3VZaTcIBFieqORVa/view?usp=sharing)

### 3. EuRoC Dataset (V101 Easy)
- **File**: `v101_stereo_imu.bag.db3`
- **Description**: Vicon room, easy trajectory
- **Topics**: Rectilinear Stereo images + IMU data
- **Download**: [Google Drive](https://drive.google.com/file/d/1gthuWv-eLEPY3N6G4SFUWppHvh0sTWZo/view?usp=sharing)

---

## Quick Start

### RGBD Visual Odometry (OpenLORIS Cafe Dataset)

```bash
# Terminal 1
ros2 launch vio_ros_wrapper rgbd_vo.launch.py

# Terminal 2
ros2 bag play cafe.bag.db3 --clock
```

### Stereo VIO (TUM-VI Room1)

```bash
# Terminal 1 (in <your_path>/vio_ws)
ros2 launch vio_ros_wrapper vio.launch.py \
    config_file:=<your_path>/vio_ws/src/vio_ros_wrapper/lightweight_vio/config/tum_vio.yaml

# Terminal 2
ros2 bag play room1_stereo_imu.bag_0.db3 --clock
```

### Stereo VIO (EuRoC V101)

```bash
# Terminal 1 (in <your_path>/vio_ws)
ros2 launch vio_ros_wrapper vio.launch.py \
    config_file:=<your_path>/vio_ws/src/vio_ros_wrapper/lightweight_vio/config/euroc_vio.yaml

# Terminal 2
ros2 bag play v101_stereo_imu.bag.db3 --clock
```

---

## Configuration

Config files: `lightweight_vio/config/`

- `euroc_vio.yaml` - EuRoC stereo + IMU
- `d435i_cafe.yaml` - RealSense RGBD
- `tum_vio.yaml` - TUM-VI stereo + IMU

---

## Topics

### Published
- `/vio/odometry` - 6-DOF pose
- `/vio/trajectory` - Path history
- `/vio/tracking_image` - Feature visualization
- `/vio/depth_image` - Depth heatmap (RGBD only)

### Subscribed (Stereo)
- `/cam0/image_raw` - Left camera
- `/cam1/image_raw` - Right camera
- `/imu0` - IMU data

### Subscribed (RGBD)
- `/camera/rgb/image_raw` - RGB image
- `/camera/depth/image_raw` - Depth image

---

## License

MIT License
