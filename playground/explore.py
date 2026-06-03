#!/usr/bin/env python
"""Step-by-step reimplementation of the soft-body Franka (MuJoCo) example.

Reference examples:
  newton/playground/example_softbody_franka_mujoco.py
  newton/examples/softbody/example_softbody_franka.py

A Franka Panda arm (driven later by GPU IK) manipulates a deformable rubber
duck on a table. We build it up incrementally so each inner stage can be
inspected. Run after the reverse SSH tunnel to the Mac's Rerun viewer is up:
    # on the Mac:        rerun                 (listens on :9876)
    # reconnect:         ssh -R 9876:localhost:9876 <this-server>

------------------------------------------------------------------------------
STEP 1 — set up Rerun as the visualizer (this step only).
------------------------------------------------------------------------------
"""

# %% 

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.utils
from newton import ModelBuilder, eval_fk
from newton.solvers import SolverMuJoCo, SolverVBD

# Remote Rerun viewer on the Mac, reached through the reverse SSH tunnel.
RERUN_ADDRESS = "rerun+http://127.0.0.1:9876/proxy"


def make_viewer() -> "newton.viewer.ViewerRerun":
    """Connect to the remote Rerun viewer.

    keep_historical_data=True logs every frame at its timestamp so the timeline
    is scrubbable/replayable (matches the example harness).
    """
    viewer = newton.viewer.ViewerRerun(
        address=RERUN_ADDRESS,
        keep_historical_data=True,
    )
    return viewer

wp.init()

# %%

viewer = make_viewer()
print(f"Rerun viewer connected: {RERUN_ADDRESS}")
print("Step 1 done — visualizer is up. (model / sim not built yet.)")


# ----------------------------------------------------------------------------
# STEP 2 — build the scene: Franka arm + table + deformable duck + ground.
# ----------------------------------------------------------------------------
# %%

# Franka keyframe sequence (reused later for IK target tracking):
# [duration, px, py, pz, qx, qy, qz, qw, gripper_activation] (positions in m).
_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = 0.5
_ROBOT_KEY_POSES = np.array(
    [
        # approach: move above the duck
        [2.5, -0.005, -0.5, 0.35, 1, 0.0, 0.0, 0.0, _GRIPPER_OPEN],
        # descend: lower to duck body
        [2.0, -0.005, -0.5, 0.21, 1, 0.0, 0.0, 0.0, _GRIPPER_OPEN],
        # pinch: close gripper on duck
        [2.5, -0.005, -0.5, 0.21, 1, 0.0, 0.0, 0.0, _GRIPPER_CLOSE],
        # lift: raise duck off table
        [2.0, -0.005, -0.5, 0.35, 1, 0.0, 0.0, 0.0, _GRIPPER_CLOSE],
        # hold: pause in air
        [2.0, -0.005, -0.5, 0.35, 1, 0.0, 0.0, 0.0, _GRIPPER_CLOSE],
        # place: lower back to table
        [2.0, -0.005, -0.5, 0.21, 1, 0.0, 0.0, 0.0, _GRIPPER_CLOSE],
        # release: open gripper
        [1.0, -0.005, -0.5, 0.21, 1, 0.0, 0.0, 0.0, _GRIPPER_OPEN],
        # retract: move away
        [2.0, -0.005, -0.5, 0.35, 1, 0.0, 0.0, 0.0, _GRIPPER_OPEN],
    ],
    dtype=np.float32,
)


def build_scene() -> "newton.Model":
    """Assemble the Franka soft-body scene and return the finalized `newton.Model`.

    Mirrors `examples/softbody/example_softbody_franka.py` (meter scale): a
    fixed-base Franka Panda (URDF) reaches over a static table to manipulate a
    deformable rubber duck (tetrahedral mesh, simulated later with VBD); the
    ground plane is global (world -1). Returns the finalized model — no
    states/solvers yet. A per-shape object label (/franka, /table, /ground) is
    attached for make_shape_path_fn; the duck is a particle/tet soft body and
    renders as particles, not shapes. Franka metadata used by later IK steps
    (endeffector body id + keyframe targets) is stashed on the model.
    """
    scene = ModelBuilder(gravity=-9.81)

    # Global (world -1): shared ground plane.
    scene.add_ground_plane()
    n_ground = scene.shape_count   # ground shapes: [0, n_ground)

    # --- One explicit world: Franka + table + deformable duck (env_0) --------
    scene.begin_world(label="env_0")

    # Franka arm (URDF). Fixed-base; URDF is in meters.
    asset_path = newton.utils.download_asset("franka_emika_panda")
    scene.add_urdf(
        str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
        xform=wp.transform((-0.5, -0.5, -0.1), wp.quat_identity()),
        floating=False,
        scale=1.0,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        force_show_colliders=False,
    )
    scene.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]
    # End-effector link (gripper hand) is 3 bodies before the end (hand + 2 fingers).
    endeffector_id = scene.body_count - 3
    n_franka = scene.shape_count   # franka shapes: [n_ground, n_franka)

    # Static table (box shape on the world's static frame, body -1).
    scene.add_shape_box(
        -1,
        wp.transform(wp.vec3(0.0, -0.5, 0.1), wp.quat_identity()),
        hx=0.4,
        hy=0.4,
        hz=0.1,
    )
    n_table = scene.shape_count   # table shape: [n_franka, n_table)

    # Deformable rubber duck (pre-computed tetrahedral mesh from USD).
    # Table top is at z=0.2m; duck center sits ~0.03m above it. add_soft_mesh
    # creates particles/tets (a soft body), not rigid shapes.
    duck_path = newton.utils.download_asset("manipulation_objects/rubber_duck")
    usd_stage = Usd.Stage.Open(str(duck_path / "model.usda"))
    prim = usd_stage.GetPrimAtPath("/root/Model/TetMesh")
    tetmesh = newton.TetMesh.create_from_usd(prim)
    scene.add_soft_mesh(
        pos=wp.vec3(0.0, -0.5, 0.23),
        rot=wp.quat_identity(),
        scale=1.0,  # already in meters
        vel=wp.vec3(0.0, 0.0, 0.0),
        mesh=tetmesh,
        density=100.0,
        k_mu=1.0e6,
        k_lambda=1.0e6,
        k_damp=1e-6,
        particle_radius=0.005,
    )

    scene.end_world()

    # VBD graph coloring for the soft body (harmless before solver setup).
    scene.color()

    model = scene.finalize(requires_grad=False)

    # Per-shape object label by build-order range (robust, no body-index guessing).
    assert model.shape_count == n_table, "shape order/count changed at finalize"
    shape_object = ["franka"] * n_table
    for i in range(n_ground):
        shape_object[i] = "ground"
    for i in range(n_franka, n_table):
        shape_object[i] = "table"
    model._explore_shape_object = shape_object

    # Stash Franka metadata for later IK steps.
    model._explore_endeffector_id = endeffector_id
    model._explore_robot_key_poses = _ROBOT_KEY_POSES

    return model


# %%

model = build_scene()
print("Step 2 done — Franka + table + duck scene built.")
print(f"  bodies   : {model.body_count}")
print(f"  shapes   : {model.shape_count}")
print(f"  joints   : {model.joint_count}  | dofs: {model.joint_dof_count}")
print(f"  particles: {model.particle_count}  (rubber duck soft body)")


# ----------------------------------------------------------------------------
# STEP 3 — send the static scene to Rerun (initial frame at t=0).
# ----------------------------------------------------------------------------
# %%

def make_shape_path_fn(model):
    """Object-centric Rerun entity paths for this scene, from
    `model._explore_shape_object`:
      * franka -> /franka/<group>/shape_N   (visual + collision)
      * table  -> /table/<group>/shape_N
      * ground -> /ground
    The duck is a soft body (particles), logged separately by log_state.
    """
    obj = model._explore_shape_object

    def fn(s, group, batch_index):
        o = obj[s]
        if o == "ground":
            return "/ground"
        return f"/{o}/{group}/shape_{batch_index}"   # /franka/... or /table/...

    return fn


def log_initial_frame(viewer, model, show_collision=True) -> "newton.State":
    """Upload the model to Rerun and log the rest-pose frame at t=0.

    Entity tree is organized by object: /franka/{visual,collision},
    /table/{visual,collision}, /ground, and the duck soft body at /duck.
    show_collision=True also renders the COLLIDE_SHAPES geometry. Returns the
    State for later stepping.
    """
    viewer.show_collision = show_collision   # render collision geometry (read by set_model)
    viewer.show_visual = True                # keep the visual meshes too
    viewer.show_particles = False            # we log the duck ourselves at /duck (not /model/particles)
    viewer.shape_path_fn = make_shape_path_fn(model)  # object-centric shape paths (read by set_model)

    viewer.set_model(model)
    viewer.set_camera(wp.vec3(0.55, 0.55, 0.95), -45.0, -30.0)

    state = model.state()
    eval_fk(model, model.joint_q, model.joint_qd, state)

    viewer.begin_frame(0.0)
    viewer.log_state(state)
    viewer.end_frame()
    return state


# %%

state = log_initial_frame(viewer, model)
print("Step 3 done — Franka + duck scene sent to Rerun at t=0.")


# ----------------------------------------------------------------------------
# STEP 4 — prepare the simulator: contact materials, states, collision
#          pipeline, PD config, and the two solvers (MuJoCo arm + VBD duck).
# ----------------------------------------------------------------------------
# %%

# --- simulation parameters (meter scale, from the reference example) --------
SIM_SUBSTEPS = 10
VBD_ITERATIONS = 5
FPS = 60
FRAME_DT = 1.0 / FPS
SIM_DT = FRAME_DT / SIM_SUBSTEPS

# contact (meter scale)
SOFT_BODY_CONTACT_MARGIN = 0.01
PARTICLE_SELF_CONTACT_RADIUS = 0.003
PARTICLE_SELF_CONTACT_MARGIN = 0.005

SOFT_CONTACT_KE = 2e6
SOFT_CONTACT_KD = 1e-7
SELF_CONTACT_FRICTION = 0.5

# Franka PD position actuation (MuJoCo solver). 7 arm joints + 2 finger joints.
N_ACT = 9
ARM_KE, ARM_KD = 650.0, 100.0
FINGER_KE, FINGER_KD = 650.0, 100.0


def prepare_simulator(model, viewer) -> SimpleNamespace:
    """Stand up the physics simulator on top of the finalized `model`.

    Mirrors the MuJoCo variant of the soft-body Franka example: the rigid arm
    is a PD-actuated dynamics sim through `SolverMuJoCo` (position actuators
    track an IK solution, contacts disabled so the arm stays in free space),
    while the deformable duck is integrated with `SolverVBD` and couples to the
    moving gripper via Newton's `CollisionPipeline`.

    Returns a `SimpleNamespace` holding the two ping-pong states, control,
    collision pipeline + contacts buffer, and both solvers. IK setup and the
    stepping loop are added in later steps.
    """
    # --- contact material properties (duck <-> arm/table/ground) ------------
    model.soft_contact_ke = SOFT_CONTACT_KE
    model.soft_contact_kd = SOFT_CONTACT_KD
    model.soft_contact_mu = SELF_CONTACT_FRICTION

    model.shape_material_ke.fill_(SOFT_CONTACT_KE)
    model.shape_material_kd.fill_(SOFT_CONTACT_KD)
    model.shape_material_mu.fill_(1.5)

    # --- ping-pong states + control -----------------------------------------
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    # --- collision pipeline for soft-body <-> rigid contacts ----------------
    collision_pipeline = newton.CollisionPipeline(
        model,
        soft_contact_margin=SOFT_BODY_CONTACT_MARGIN,
    )
    contacts = collision_pipeline.contacts()

    # --- PD position-actuation config on the model arrays (pre-solver) -------
    ke = model.joint_target_ke.numpy()
    kd = model.joint_target_kd.numpy()
    ke[:7], kd[:7] = ARM_KE, ARM_KD
    ke[7:N_ACT], kd[7:N_ACT] = FINGER_KE, FINGER_KD
    model.joint_target_ke.assign(ke)
    model.joint_target_kd.assign(kd)

    arm = model.joint_armature.numpy()
    arm[:7] = 0.1
    arm[7:N_ACT] = 0.5
    model.joint_armature.assign(arm)

    # --- robot solver: MuJoCo (PD dynamics, arm in free space) --------------
    robot_solver = SolverMuJoCo(
        model,
        solver="newton",
        integrator="implicitfast",
        cone="elliptic",
        disable_contacts=True,   # arm in free space; duck couples via VBD only
        iterations=10,
        ls_iterations=20,
    )

    # --- soft-body solver: VBD ----------------------------------------------
    soft_solver = SolverVBD(
        model,
        iterations=VBD_ITERATIONS,
        integrate_with_external_rigid_solver=True,
        particle_self_contact_radius=PARTICLE_SELF_CONTACT_RADIUS,
        particle_self_contact_margin=PARTICLE_SELF_CONTACT_MARGIN,
        particle_enable_self_contact=False,
        particle_vertex_contact_buffer_size=32,
        particle_edge_contact_buffer_size=64,
        particle_collision_detection_interval=-1,
    )

    # initial state at rest pose
    eval_fk(model, model.joint_q, model.joint_qd, state_0)

    return SimpleNamespace(
        model=model,
        viewer=viewer,
        state_0=state_0,
        state_1=state_1,
        control=control,
        collision_pipeline=collision_pipeline,
        contacts=contacts,
        robot_solver=robot_solver,
        soft_solver=soft_solver,
        sim_time=0.0,
    )


# %%

sim = prepare_simulator(model, viewer)
print("Step 4 done — simulator prepared.")
print(f"  robot solver : {type(sim.robot_solver).__name__}  (PD, contacts disabled)")
print(f"  soft solver  : {type(sim.soft_solver).__name__}  (iterations={VBD_ITERATIONS})")
print(f"  contacts buf : {type(sim.contacts).__name__}")

# %%
