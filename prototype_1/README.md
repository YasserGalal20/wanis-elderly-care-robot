# Prototype 1

The first build. Kept because the progression from this to the final robot is
the interesting part — this is where the ideas were established and where we
found out what does not work.

## What was here

| File | What it does |
|---|---|
| `person_follower/` | MediaPipe pose + DeepLabV3 segmentation, PID following, rear/side proximity guard |
| `my_bot/` | URDF, launch files, Nav2 / SLAM / EKF configuration |
| `explorer/` | Frontier exploration, used to search for the person when lost |

## What it got right

- **PID following with a standoff distance.** The robot backs away if the person
  comes within 2.5 m, rather than crowding them.
- **Last-known-direction recovery.** When the person leaves the field of view,
  rotate toward the direction they were last moving instead of guessing.
- **Rear and side proximity guard.** A node watching the LiDAR that stops
  translation and allows only rotation when something is close behind or beside
  the robot. This idea became the safety layer in prototype 2.
- **Frontier exploration on loss.** Hand off to Nav2 and go look, rather than
  giving up.

## What it got wrong

**It followed the nearest person, not a specific person.** In a home this fails
almost immediately — anyone walking past captures the robot. Every identity and
re-identification feature in the final build exists because of this.

Secondary issues: MediaPipe pose was slower and less robust than segmentation-based
detection; the single-machine setup meant a laggy laptop could delay motor
commands, which is why the final build moved safety onto the Pi.

## Note on the phone sensor bridge

Prototype 1 also used [ros2-mobile-sensor-bridge](https://github.com/VedantC2307/ros2-mobile-sensor-bridge)
by Vedant Choudhary (MIT) to stream a phone's camera, IMU and GPS into ROS 2 over
WebSocket. That let us prototype sensing without wiring extra hardware. It is a
third-party package and is not included here — see `robot.repos`.
