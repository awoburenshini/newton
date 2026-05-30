#!/usr/bin/env python
"""Step-by-step exploration of the Allegro-hand model.

Reference example:
  newton/examples/robot/example_robot_allegro_hand.py

We build it up incrementally so each inner stage can be inspected. Run after
the reverse SSH tunnel to the Mac's Rerun viewer is up:
    # on the Mac:        rerun                 (listens on :9876)
    # reconnect:         ssh -R 9876:localhost:9876 <this-server>

------------------------------------------------------------------------------
STEP 1 — set up Rerun as the visualizer (this step only).
------------------------------------------------------------------------------
"""

# %% 

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.utils
from newton import ModelBuilder, eval_fk

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
# STEP 2 — build the scene: Allegro hand (+ cube) + ground.
# ----------------------------------------------------------------------------
# %%

def build_scene() -> "newton.Model":
    """Assemble the Allegro-hand scene and return the finalized `newton.Model`.

    Loads the Wonik Allegro left hand from USD (the asset bundles a manipulation
    cube as a free-floating body) into one explicit world (env_0); the ground
    plane is global (world -1). Returns the finalized model — no states/solvers
    yet. A per-shape object label (/hand, /cube, /ground) is attached for
    make_shape_path_fn.
    """
    scene = ModelBuilder(gravity=-9.81)
    scene.default_shape_cfg.ke = 1.0e3
    scene.default_shape_cfg.kd = 1.0e2
    scene.default_shape_cfg.margin = 0.005
    scene.default_shape_cfg.gap = 0.015

    # Global (world -1): shared ground plane.
    scene.add_ground_plane()
    n_ground = scene.shape_count   # ground shapes: [0, n_ground)

    # --- One explicit world: Allegro hand + an explicit cube (env_0) --------
    scene.begin_world(label="env_0")

    # HAND ONLY — exclude the USD's bundled cube subtree (/.../object/*); we add
    # our own cube below so the two are decoupled.
    asset_path = newton.utils.download_asset("wonik_allegro")
    asset_file = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")
    scene.add_usd(
        asset_file,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.5)),
        enable_self_collisions=False,
        ignore_paths=[".*Dummy", ".*CollisionPlane", ".*object.*"],  # drop bundled cube
        hide_collision_shapes=False,   # keep collision shapes -> shown under /…/collision
    )
    n_hand = scene.shape_count   # hand shapes: [n_ground, n_hand)

    # Natural finger pose (fixed-base hand: every DOF is a finger revolute).
    # Fingers 0.3 rad; the proximal "_0" joints 0.6.
    for i in range(scene.joint_dof_count):
        scene.joint_q[i] = 0.6 if scene.joint_label[i][-2:] == "_0" else 0.3

    # CUBE added EXPLICITLY (not from USD), but matched to the USD DexCube config:
    # its USD transform is (0, -0.17, 0.56) with identity rotation and box
    # half-extents 0.036; we add the (0,0,0.5) hand load offset -> world (0,-0.17,1.06).
    # add_body already gives the body a 6-DOF FREE joint (free-floating by default),
    # so we do NOT call add_joint_free (that would add a redundant 2nd free joint).
    cube_cfg = ModelBuilder.ShapeConfig(density=500.0)
    cube_body = scene.add_body(
        label="cube",
        xform=wp.transform(wp.vec3(0.0, -0.17, 1.06), wp.quat_identity()),
    )
    scene.add_shape_box(cube_body, hx=0.036, hy=0.036, hz=0.036, cfg=cube_cfg)
    n_cube = scene.shape_count   # cube shapes: [n_hand, n_cube)

    scene.end_world()

    model = scene.finalize(requires_grad=False)

    # Per-shape object label by build-order range (robust, no body-index guessing).
    assert model.shape_count == n_cube, "shape order/count changed at finalize"
    shape_object = ["hand"] * n_cube
    for i in range(n_ground):
        shape_object[i] = "ground"
    for i in range(n_hand, n_cube):
        shape_object[i] = "cube"
    model._explore_shape_object = shape_object

    return model


# %%

model = build_scene()
print("Step 2 done — Allegro hand scene built.")
print(f"  bodies   : {model.body_count}")
print(f"  shapes   : {model.shape_count}")
print(f"  joints   : {model.joint_count}  | dofs: {model.joint_dof_count}")


# ----------------------------------------------------------------------------
# STEP 3 — send the static scene to Rerun (initial frame at t=0).
# ----------------------------------------------------------------------------
# %%

def make_shape_path_fn(model):
    """Object-centric Rerun entity paths for this scene, from
    `model._explore_shape_object`:
      * hand   -> /hand/<group>/shape_N   (visual + collision)
      * cube   -> /cube/<group>/shape_N
      * ground -> /ground
    """
    obj = model._explore_shape_object

    def fn(s, group, batch_index):
        o = obj[s]
        if o == "ground":
            return "/ground"
        return f"/{o}/{group}/shape_{batch_index}"   # /hand/... or /cube/...

    return fn


def log_initial_frame(viewer, model, show_collision=True) -> "newton.State":
    """Upload the model to Rerun and log the rest-pose frame at t=0.

    Entity tree is organized by object: /hand/{visual,collision},
    /cube/{visual,collision}, /ground. show_collision=True also renders the
    COLLIDE_SHAPES geometry. Returns the State for later stepping.
    """
    viewer.show_collision = show_collision   # render collision geometry (read by set_model)
    viewer.show_visual = True                # keep the visual meshes too
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
print("Step 3 done — Allegro hand sent to Rerun at t=0.")

# %%
