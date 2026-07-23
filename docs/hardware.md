# Hardware

![Final hardware](../media/final_hardware_diagram.png)

## Bill of materials

| Part | Qty | Role |
|---|---|---|
| Hoverboard hub motors + FOC mainboard | 1 pair | Drive. Reflashed with open FOC firmware |
| Raspberry Pi | 1 | Sensor nodes, `ros2_control`, safety layer |
| Laptop / PC | 1 | Perception, SLAM, Nav2, assistant |
| RPLiDAR | 1 | Obstacle detection, SLAM, safety polygon |
| Kinect v1 / v2 | 1 | RGB + depth for person tracking |
| BNO08x IMU | 1 | Heading for the EKF |
| ESP32-WROVER | 1 | Pill dispenser camera + Wi-Fi bridge |
| LOLIN32 Lite | 1 | Pill carousel servo |
| ESP32 | 1 | Pill cartridge servos |
| ESP32 + MAX30102 / MAX30205 | 1 | Vital signs: heart rate, SpO2, body temperature |
| Battery (hoverboard pack) | 1 | Drive + electronics |

## Reflashing the hoverboard mainboard

The stock firmware only accepts balance-board input, so the board is reflashed with [hoverboard-firmware-hack-FOC](https://github.com/hoverboard-robotics/hoverboard-firmware-hack-FOC), which exposes a serial protocol.

1. Open the hoverboard, identify the mainboard, solder to the SWD pads (SWDIO, SWCLK, GND, 3V3).
2. Flash with an ST-Link. **Take note of the original firmware first** — it is not recoverable otherwise.
3. Apply our config in [`patches/hoverboard-firmware-hack-FOC.patch`](../patches/hoverboard-firmware-hack-FOC.patch), which sets the serial control mode and baud rate we use.
4. Wire the mainboard's serial to the Pi's UART.

`hoverboard_hardware_interface` then speaks that protocol as a `ros2_control` system, so wheels appear as normal velocity-controlled joints. Our modifications to it are in [`patches/hoverboard_ros2_control.patch`](../patches/hoverboard_ros2_control.patch).

## Assembly

| | |
|---|---|
| ![Pre-build](../media/final_prebuild.jpg) | ![Build 1](../media/final_build_1.jpg) |
| Frame before assembly | Mounting the drive |
| ![Build 2](../media/final_build_2.jpg) | ![Build 3](../media/final_build_3.jpg) |
| Electronics layer | Sensor mast |

![Internals](../media/final_prototype_internals.jpg)

## CAD

| | |
|---|---|
| ![Proposed](../media/cad_proposed_model.jpg) | ![Finished](../media/cad_finished_model.jpg) |
| Proposed model | Final design |

## Vital-sign sensing

The MAX30102 is a reflective pulse-oximeter (red + IR LEDs and a photodiode) and
the MAX30205 is a clinical-accuracy body-temperature sensor. Both sit on I2C to
an ESP32, which pushes readings over Wi-Fi rather than through ROS, so the
wearable stays independent of whether the robot is running.

## Frames

Standard REP-105 layout: `map → odom → base_link → {laser, camera_link, imu_link}`.

`base_link` sits at the centre of the wheel axle at ground level. The URDF is in [`src/wanis_bringup/urdf/`](../src/wanis_bringup/urdf/); sensor mount offsets there must match the physical build or the safety polygon and SLAM will disagree with reality.

## Clock sync

The Pi and server must agree on time or TF drops transforms as too old — which presents as perception failing for no obvious reason. [`scripts/time_sync.py`](../src/wanis_bringup/scripts/time_sync.py) syncs the Pi to the server over SSH via `chrony`.

Set the SSH password through the environment rather than passing it on the command line:

```bash
export ROBOT_SSH_PASSWORD='...'   # or better, use key auth
python3 src/wanis_bringup/scripts/time_sync.py --host <pi-ip>
```
