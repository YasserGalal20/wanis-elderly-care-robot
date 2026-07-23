# Design notes

Short notes on why things are built the way they are. The README covers *what* the system does; this covers *why*.

---

## Why hoverboard wheels

We needed wheels that produce high torque, can carry real weight, and do not drain the battery — a home robot has to push through carpet edges and carry a pill dispenser. Hoverboard hub motors do all three and cost almost nothing second-hand.

The catch is the stock controller runs locked firmware that only accepts balance-board input. The board is reflashed with an open-source FOC firmware, which exposes a serial protocol we drive from a `ros2_control` hardware interface.

## Why the compute is split across two machines

The perception stack runs YOLO11 segmentation plus an OSNet ReID embedding on every tracked person, every frame. That does not run in real time on a Raspberry Pi.

But the safety layer *must* keep running even when Wi-Fi drops. So the rule is: **anything that can stop the wheels lives on the robot.** The Pi runs sensor drivers, `ros2_control` and the safety guard. The laptop runs perception, Nav2 and the assistant. If the network dies, the robot stops safely instead of continuing on a stale command.

A consequence is that clocks must agree, or TF transforms get rejected as too old. `scripts/time_sync.py` handles that over SSH.

## Why identity matters more than detection

Prototype 1 followed the nearest detected person. In a lab that works perfectly. In a home it fails within a minute — someone walks past, the robot switches target and follows the wrong person out of the room.

Person detection is a solved problem. Deciding *which* detection is the person you care about is not. That is why prototype 2 is built around a signature rather than a bounding box.

**The four cues and what each is for:**

| Cue | Fails when | Covered by |
|---|---|---|
| OSNet ReID embedding | Very long absence, changed clothes | HSV histogram |
| HSV histogram (upper/lower split) | Lighting changes | LBP texture |
| LBP texture | Uniform clothing | ReID embedding |
| Depth median over mask | Occlusion | All of the above |

No single cue is reliable. Scored together they are, because they fail in different situations. Splitting the histogram into upper and lower body separately is what stops the robot locking onto a different person wearing a similar-coloured shirt — that specific failure happened often enough in testing to be worth the extra state.

## Why recovery escalates instead of just searching

When the target disappears, the right response depends on *why* they disappeared, and the robot cannot know which it is. So it tries the cheapest explanation first:

1. **BACKUP** — they are probably still nearby and just too close to the camera, or briefly occluded. Backing up widens the field of view. Cheap and fast.
2. **SCAN** — they probably walked out of frame. Rotate toward the last observed direction of motion.
3. **FRONTIER** — they have actually left. Use Nav2 to drive to unexplored frontiers and look.

Each stage costs more time and risk than the last, so escalating in order means the common cases resolve quickly. Re-acquisition at any stage requires a signature match, not just any person detection — otherwise recovery would happily lock onto a stranger.

## Why the safety layer is a separate node

The follower is a 2,700-line node with a state machine, several PID loops and a neural network. It is the most likely thing in the system to have a bug.

So it is not trusted with collision avoidance. It publishes intent to `/cmd_vel_raw`, and a small, boring node with one job decides what actually reaches the motors. The guard is simple enough to read in full and convince yourself it is correct — which is the whole point.

This also means a hung or crashed follower cannot drive the robot into a wall; it just stops producing commands.

## Motion tuning

Two things that mattered more than expected:

**Scale forward speed by heading error, do not gate it.** The obvious approach — "if heading error is large, stop and rotate" — makes the robot stutter between rotating and driving. Continuously scaling forward speed by heading error makes it arc smoothly toward the target instead.

**Hysteresis on the rotate-only threshold.** With a single threshold the robot chatters right at the boundary. Separate enter/exit thresholds fix it.

Also standard, and worth having: D-on-measurement rather than D-on-error (no derivative kick when the setpoint jumps), anti-windup, integral leak on sign flip, output low-pass and slew-rate limiting.

## Known limitations

- **Re-identification degrades if the person changes clothes.** The ReID embedding carries some of it, but colour cues dominate the score. A longer-term signature would need face or gait cues.
- **Depth is Kinect-based**, so bright sunlight through a window degrades it.
- **Frontier recovery assumes a reasonable map.** In an unmapped area it explores rather than searches.
- **The safety guard uses a rectangular footprint**, which is conservative for a roughly circular robot — it stops slightly earlier than strictly necessary. That was the intended trade.
- **Clock sync is a hard dependency.** If the Pi and server drift, TF breaks in ways that look like perception bugs.
