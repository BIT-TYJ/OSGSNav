# Multi-goal navigation to real hardware: handoff

## What is implemented

The uploaded source is the five-goal V5 Habitat runtime plus its two-goal reference. It includes persistent Gaussian mapping, target switching, historical retrieval, coarse-to-fine localization, PVA/recovery and FMM exploration. It is suitable as the algorithm/reference code for a hardware port.

It does not implement a ROS subscriber, calibrated sensor synchronizer, external LIO-to-camera pose adapter, real navigation action server, motor controller or hardware success evaluator. The existing documents are design/interface plans, not claims of tested deployment. Do not launch the Habitat entry and expect it to command a physical robot.

## Integration boundary

1. **Observations:** replace Habitat RGB-D observations with timestamped aligned RGB (`uint8`, RGB order) and metric depth (`float32`, metres; explicit invalid-depth policy). Use measured `CameraInfo.K`, not a guessed simulation HFOV.
2. **Pose:** replace simulator ground truth with the external localization estimate. Specify camera/base/LIO transforms, timestamp alignment, and world-to-camera versus camera-to-world convention before feeding the Gaussian mapper. See the existing coordinate and interface notes; no automatic ROS-to-Habitat transform is provided here.
3. **Query input:** supply an ordered list of goal RGB images and explicit category metadata or an evaluated visual category predictor. The reference simulation CLI uses `goal_cat_source=oracle`, which is dataset metadata and unavailable as a simulator oracle on hardware. Goal depth is not required by the RGB-only localization path.
4. **Control:** replace Habitat stepping with a short-term-goal interface and a separate base controller. Habitat's normalized velocity-action values are not physical `/cmd_vel` velocities. Handle stop, arrival, collision, timeout and target advance explicitly.
5. **Persistence:** preserve mapper/occupancy/keyframe/instance state across queries, while resetting query-local navigation and verification state. Treat observed cross-query semantic-label residue as an issue to validate; the uploaded benchmark source is not silently patched to change its behavior.
6. **Evaluation:** hardware instance identity and success need an independent measurement protocol. Internal `localized` events and high matching scores do not establish real target arrival.

## Recommended first hardware milestone

Replay a short recorded RGB-D + calibrated camera pose sequence into the mapper while the robot is stationary. Verify depth scale, RGB/depth alignment, projected Gaussian geometry and rendering. Then test target switching and retrieval with a fixed map. Integrate physical short-term-goal execution only after these interfaces agree.

For initial physical execution, use the stop-and-go/latest-frame design already documented here; motion/control must remain responsive independently of the potentially slow localization worker. Keep measured runtime and memory growth logs. Existing long simulation runs have encountered high process memory growth, and source publication does not establish that the hardware port is memory-stable.

Supporting documents:

- `habitat_to_real_robot_interface_summary.md`
- `ieve_real_robot_mid360_femto_bolt_migration.md`
- `real_robot/real_robot_mid360_integration.md`
- `real_robot/real_robot_30hz_stop_go_implementation.md`
- `real_robot/停走式导航流程.md`
