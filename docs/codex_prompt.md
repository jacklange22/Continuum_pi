Build a production-oriented Python codebase for a tendon-driven continuum robot that runs entirely on a Raspberry Pi and integrates:

1. Direct Aurora EM tracker serial input into the Pi
2. Direct OpenRB-150 connection from the Pi for DYNAMIXEL control
3. XC330-M288 Robotis servos controlling either:
   - 4 motors for one segment
   - 8 motors for two segments
4. A local GUI on the Pi for setup, calibration, registration, manual control, experiment execution, logging, and troubleshooting

This system replaces any Linux relay box or Arduino intermediary. The Raspberry Pi is the only host. The Pi reads the Aurora serial stream directly, sends motor commands directly to the OpenRB-150, and runs the GUI locally with HDMI monitor, keyboard, and mouse attached.

The codebase should be organized, maintainable, safe, and ready for iterative lab use.

## Core design goals

The software must let a user:
- boot the system on a Raspberry Pi
- connect to Aurora and OpenRB-150 from the GUI
- scan and assign servo IDs when needed
- bring the servos into a usable operating state from a utility button in the GUI
- manually command tendon displacement
- convert tendon displacement into servo position commands around saved tendon neutral setpoints
- monitor current and voltage for safe operation
- perform manual-first tendon neutral calibration and save those setpoints
- optionally run a configurable pretension validation routine at the start of each run
- register the robot frame using Aurora tool 0B and repeated point captures
- compute the robot tip pose from Aurora tool 0A in the registered robot frame
- load an experiment point file and execute a repeatability-style run
- save one `.dat` file per run in the same spirit as the existing repeatability workflow

The v1 emphasis is reliability, safety, operator guidance, and troubleshooting. Keep visualization simple.

## Important references to use while implementing

Use the existing `repeatability.py` as a behavioral reference for experiment execution, point stepping, sampling, transform use, and `.dat` logging style.

Use the existing `rigid_registration.py` as the reference for registration logic, repeated landmark capture handling, transform generation, and validation outputs.

Do not copy these files blindly. Reuse the logic patterns and structure where appropriate, but refactor into a clean modular codebase.

## Hardware assumptions

- Raspberry Pi is the only host computer
- Aurora tracker is connected directly to one Pi serial port
- OpenRB-150 is connected directly to another Pi serial/USB port
- Only two Aurora tools matter:
  - `0A` = robot coil
  - `0B` = pen probe for registration
- Servos are XC330-M288 DYNAMIXEL servos
- OpenRB-150 is used as the DYNAMIXEL controller/interface
- System must support:
  - 4-servo mode for one segment
  - 8-servo mode for two segments
- A configuration file must define whether the robot is in 4-servo or 8-servo mode
- Pulley/spool diameter default is 1.2 cm, but this must be configurable

## Software stack and implementation guidance

Use Python.

Use a robust GUI framework suitable for running locally on a Raspberry Pi. Prefer PySide6 or PyQt if practical.

Use a modular architecture with clear separation between:
- hardware interfaces
- transform/tracking logic
- servo control logic
- registration workflow
- experiment execution
- GUI
- configuration and persistence

Use configuration files rather than hardcoded constants wherever possible.

Avoid hardcoded serial port names.

Implement mock or simulation modes where practical so parts of the GUI can be tested without hardware.

## Required modules

Organize the codebase into something like:

- `app/`
- `gui/`
- `tracking/`
- `servos/`
- `registration/`
- `experiments/`
- `models/`
- `config/`
- `utils/`
- `data/`

This exact structure can vary, but the code must remain modular and clean.

---

# 1. Tracking subsystem

Implement a Pi-native Aurora tracking subsystem that:

- opens the Aurora serial connection directly from the Pi
- parses the raw Aurora serial packet stream
- handles framing, byte stuffing, CRC, and packet extraction
- supports only tools `0A` and `0B`
- returns raw tool data in a structured form
- converts raw tool data into calibrated 4x4 homogeneous transforms
- computes the robot tip pose in the registered robot frame reliably

The runtime output that matters most is:

- robot tip pose derived from tool `0A`, expressed in the robot frame

The registration tool is:

- tool `0B`, used to collect known landmarks on the robot frame

The tracking subsystem must provide:

- raw tool measurements
- parsed tool pose
- 4x4 transforms for tools
- tool status / missing tool state
- tracking quality / validity indicators if available
- tip position and orientation in robot frame

Implement transform handling cleanly. The system should support calibrated transforms such as:
- tracker frame to robot frame
- tool coil frame to tip frame

The exact transform chain should follow the same logic as the existing registration workflow in `rigid_registration.py`.

Provide clear APIs such as:
- connect tracker
- get latest tool data
- get `T_tool_0A_to_tracker`
- get `T_pen_0B_to_tracker`
- get `T_tip_to_robot`
- get tool status

The tracking subsystem should also support live capture during registration and live sampling during experiments.

---

# 2. Registration subsystem

Implement a guided GUI registration workflow based on the logic in `rigid_registration.py`.

Default workflow:
- 4 landmarks
- 5 captures per landmark

Make both configurable.

The registration GUI must:
- walk the user through each landmark in order
- clearly indicate current landmark number and capture count
- use tool `0B` only
- collect repeated samples for each landmark
- save raw captured points
- compare captured points to nominal robot-frame landmark coordinates
- compute the rigid registration transform
- compute validation metrics and show them to the user
- let the user accept or retry registration
- save the resulting transforms for later runtime use

Registration requirements:
- default: 4 landmarks, 5 captures each
- support configurable landmark count and capture count
- show residual / alignment error
- show basic validation indicators
- save all registration data to disk
- persist the accepted registration so the runtime tracking system uses it automatically

For v1, simple visualization is fine:
- landmark positions
- captured points
- current pen position
- maybe axes/frame markers

No fancy rendering is needed.

---

# 3. Servo subsystem

Implement a servo control subsystem for OpenRB-150 + XC330-M288.

The user’s main motion command is:
- tendon displacement

The software must convert tendon displacement into servo position around a saved neutral setpoint using spool geometry.

Default spool diameter:
- 1.2 cm

Make this configurable.

The servo subsystem must support:
- 4 or 8 motors
- connection to OpenRB-150
- scanning for servos
- assigning servo IDs
- reading present position
- reading current and voltage if available
- sending position commands
- safe motion limits
- error handling and troubleshooting

The GUI does not need advanced servo visualization or analytics, but it does need:
- connection status
- ID scan and assignment
- present position display
- current/voltage display
- fault indicators
- troubleshooting messages

## Neutral calibration and pretensioning

Implement a manual-first calibration workflow.

The intended mechanical process is:
1. mechanically mount tendon spool
2. place robot in neutral backbone pose
3. manually jog each motor until the tendon is at desired neutral tension
4. read servo Present Position (address 132)
5. save that as the tendon’s neutral setpoint
6. define safe min/max bounds around that neutral position
7. use those calibrated values for all future control

The GUI must support this directly.

In addition, support a configurable pretension validation routine that can be run at the start of a run.

For v1:
- use position-based control as the main command mode
- monitor current and voltage for safety
- use current-based thresholds to reduce the risk of pulling strings through
- do not implement full advanced feedback pose control in v1

Pretension and startup validation rules must be configurable.

Default pretension validation behavior:
- current balance across tendons is the default enabled validation criterion

The code should be structured so future validation criteria can be added, such as:
- verticality/orientation check using Aurora
- mixed current + tracker-based checks

Support both:
- manual calibration mode for establishing saved neutral setpoints
- startup validation mode that checks whether the current state is close enough to a pretensioned / centered state

The code should expose configuration values for:
- current thresholds
- safe min/max servo travel relative to neutral setpoint
- displacement limits
- velocity/step sizes for jog and motion
- per-servo offsets
- 4-servo or 8-servo tendon mapping

## Safety behavior

The servo subsystem must enforce:
- commanded tendon displacement stays within safe position range
- current limit / over-tension protection
- clear operator-visible errors
- stop or reject unsafe motion

If a command would exceed position bounds or current thresholds, the system must:
- not silently continue
- show the operator a clear error
- optionally stop motion or prevent command dispatch

---

# 4. OpenRB-150 utility support

Do not implement risky low-level reflashing.

Instead, provide a simple GUI utility button that helps put the OpenRB-150 into the correct usable operating state for this application.

This can be a wrapper that:
- sends the right setup command
- invokes a supported utility
- or otherwise prepares the board for operation

The goal is practical usability, not full vendor-tool replacement.

Also provide:
- connect/disconnect status
- port selection
- troubleshooting messages if the board is not reachable

---

# 5. GUI requirements

Create a local GUI that runs on the Raspberry Pi.

Prioritize:
- clarity
- lab usability
- safe operator workflow
- troubleshooting
- reliable status feedback

Suggested tabs or sections:

## System
- serial port selection
- connect/disconnect Aurora
- connect/disconnect OpenRB-150
- status messages
- troubleshooting panel
- utility button to prepare OpenRB-150 for use

## Servos
- scan for servos
- assign IDs
- show connected servo IDs
- manually jog motors
- input tendon displacement and execute motion
- display neutral setpoints
- save/load neutral calibration
- show present position, current, voltage, and faults
- define safe motion ranges
- run startup pretension validation

## Tracking
- connect to Aurora
- show live tool status for `0A` and `0B`
- show live numeric pose outputs
- show robot tip pose in robot frame
- basic visualization of tracked points / landmarks / frames
- simple validation indicators

## Registration
- guided 4-landmark workflow by default
- 5 captures per landmark by default
- show capture progress
- show live pen position
- save captured data
- compute registration
- show registration residual / validation metrics
- accept / retry registration

## Experiment
- load experiment file
- set number of points if needed
- execute points in order
- wait configurable settle time
- sample tracking + servo data
- save exactly one `.dat` file per run
- show simple run progress and operator notes

Keep the 3D visualization simple in v1. Basic landmarks, point clouds, and current positions are enough.

---

# 6. Experiment execution and logging

Experiment execution should be modeled after the existing `repeatability.py`.

Implement:
- load a point/experiment file
- iterate through points
- move servos to the commanded tendon displacement state
- wait configurable settle time
- sample Aurora data
- compute the tip pose in robot frame
- record servo states
- append to the run output
- save exactly one `.dat` file per run

Logging must be practical and traceable.

Each run should produce:
- one `.dat` file per run
- format inspired by the existing repeatability output style

Include in the run data:
- timestamps
- point index
- commanded tendon displacement
- measured servo positions
- measured current and voltage if available
- tracker/tool data as needed
- computed robot tip pose in robot frame
- relevant registration/calibration identifiers or metadata

Use a clean writer class so output formatting is centralized.

If useful, also save sidecar metadata files, but the primary requirement is one `.dat` file per run.

---

# 7. Configuration and persistence

Implement configuration files for:
- robot mode: 4-servo or 8-servo
- spool diameter
- tendon-to-servo mapping
- servo IDs
- neutral setpoints
- safe min/max offsets from neutral
- current thresholds
- serial port defaults
- registration landmark nominal coordinates
- registration capture counts
- experiment settle time
- file output locations

Saved calibrations should include:
- servo neutral setpoints
- registration transforms
- any tool-to-tip transforms needed for tracking

The system must load these on startup and allow updating them from the GUI.

---

# 8. Code quality requirements

The codebase must:
- be production-oriented and readable
- separate GUI from hardware access
- separate transform math from serial parsing
- avoid hardcoded file paths and serial ports
- have clear error handling
- have logging
- include docstrings and comments where useful
- be structured so future closed-loop improvements can be added cleanly

Where appropriate, create interfaces / classes such as:
- tracker interface
- OpenRB / servo bus interface
- registration manager
- experiment runner
- config manager
- run logger
- GUI controller/view models

---

# 9. Deliverables

Generate the following:

1. Full Python source code
2. Clean project structure
3. `requirements.txt`
4. README with Raspberry Pi setup instructions
5. Example configuration files for:
   - 4-servo robot
   - 8-servo robot
6. Example registration landmark config
7. Example experiment file format
8. Launch scripts or commands for:
   - GUI mode
   - diagnostic mode
9. Mock/test mode where practical

Also include clear documentation of:
- how tendon displacement converts to servo position
- how neutral setpoints are established and saved
- how registration is performed
- how experiment files are run
- how output `.dat` files are structured

---

# 10. Constraints and priorities

Priorities for v1:
1. safety
2. reliability
3. good operator workflow
4. practical troubleshooting
5. direct Pi integration
6. clean architecture
7. simple but functional GUI
8. minimal but sufficient visualization

Do not spend excessive effort on flashy graphics or advanced analytics.

Do not assume additional Aurora tools beyond `0A` and `0B`.

Do not require an intermediate Linux box or Arduino.

Use the existing `repeatability.py` and `rigid_registration.py` as reference behaviors and incorporate their useful logic patterns into a better-structured system.

Build the codebase so it is immediately usable as the foundation for a real lab control system.
