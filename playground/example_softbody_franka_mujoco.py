#!/usr/bin/env python
"""Same task as `examples/softbody/example_softbody_franka.py`, but the ARM is
driven by `SolverMuJoCo` (mujoco_warp) instead of `SolverFeatherstone`.

Everything else is reused unchanged from the original example:
  * scene (Franka + table + ground + deformable rubber duck),
  * GPU IK target tracking,
  * `SolverVBD` soft-body solver + Newton `CollisionPipeline` for the duck.

Only two things change vs the original:
  1. The rigid arm is a PD-actuated *dynamics* sim through MuJoCo (position
     actuators tracking the IK solution), rather than Featherstone used as a
     pure kinematic integrator. `disable_contacts=True` keeps the arm in free
     space (no rigid-rigid contacts) so behaviour mirrors the kinematic arm;
     the duck still reacts to the moving gripper via the collision pipeline.
  2. We drive `control.joint_target_pos` instead of assigning `joint_qd`.

Run:  python playground/example_softbody_franka_mujoco.py --viewer null --num-frames 60
      (or --viewer rerun --rerun-address rerun+http://127.0.0.1:9876/proxy)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import warp as wp

import newton
import newton.examples

# ---- load the original example module by path (softbody/ is not a package) ----
_BASE_PATH = Path(__file__).resolve().parents[1] / "newton/examples/softbody/example_softbody_franka.py"
_spec = importlib.util.spec_from_file_location("example_softbody_franka", _BASE_PATH)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

# Franka has 7 arm joints + 2 finger joints = 9 actuated DOFs.
_N_ACT = 9
_ARM_KE, _ARM_KD = 650.0, 100.0
_FINGER_KE, _FINGER_KD = 650.0, 100.0


class Example(_base.Example):
    def __init__(self, viewer, args=None):
        # Build the full original scene (this also creates a Featherstone solver
        # and captures a graph against the base simulate() — both discarded below).
        super().__init__(viewer, args)

        # --- configure PD position actuation on the model arrays (pre-solver) ---
        ke = self.model.joint_target_ke.numpy()
        kd = self.model.joint_target_kd.numpy()
        ke[:7], kd[:7] = _ARM_KE, _ARM_KD
        ke[7:_N_ACT], kd[7:_N_ACT] = _FINGER_KE, _FINGER_KD
        self.model.joint_target_ke.assign(ke)
        self.model.joint_target_kd.assign(kd)

        arm = self.model.joint_armature.numpy()
        arm[:7] = 0.1
        arm[7:_N_ACT] = 0.5
        self.model.joint_armature.assign(arm)

        # --- swap the robot solver: Featherstone -> MuJoCo -------------------
        self.robot_solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            disable_contacts=True,   # arm in free space; duck couples via VBD only
            iterations=10,
            ls_iterations=20,
        )

        # Seed MuJoCo position targets with the current joint config.
        wp.copy(self.control.joint_target_pos, self.model.joint_q,
                dest_offset=0, src_offset=0, count=self.n_coords)

        # Re-capture the graph now that simulate() resolves to *this* class and
        # the MuJoCo solver is in place.
        self.capture()

    def capture(self):
        """Graph-capture the MuJoCo+VBD frame; fall back to eager on failure."""
        self.graph = None
        if wp.get_device().is_cuda:
            try:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph
            except Exception as e:  # noqa: BLE001
                print(f"[mujoco variant] graph capture unavailable, running eager: {e}")
                self.graph = None

    def simulate(self):
        # IK solve once per frame (GPU).
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)

        # Gripper finger positions from the keyframe buffer.
        wp.launch(
            _base.set_gripper_q,
            dim=1,
            inputs=[self.ik_joint_q, self.finger_pos_buf, self.finger_idx0, self.finger_idx1],
        )

        # IK result -> MuJoCo position-actuator targets (2D->1D contiguous copy).
        wp.copy(self.target_joint_q, self.ik_joint_q, dest_offset=0, src_offset=0, count=self.n_coords)
        wp.copy(self.control.joint_target_pos, self.target_joint_q,
                dest_offset=0, src_offset=0, count=self.n_coords)

        self.soft_solver.rebuild_bvh(self.state_0)
        for _step in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.viewer.apply_forces(self.state_0)

            # Rigid arm dynamics through MuJoCo (PD actuators track the target).
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

            # Duck reacts to the moving gripper.
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.soft_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    # Default to streaming to a remote Rerun viewer (override on the CLI if needed):
    #   --viewer null | gl | usd ...   --rerun-address <url>
    parser.set_defaults(
        num_frames=1000,
        viewer="rerun",
        rerun_address="rerun+http://127.0.0.1:9876/proxy",
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
