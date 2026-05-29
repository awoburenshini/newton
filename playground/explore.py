#!/usr/bin/env python
"""Step-by-step exploration of the softbody Franka pipeline.

Reference example:
  newton/examples/softbody/example_softbody_franka.py

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
from pxr import Usd

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
# STEP 2 — build the scene: Franka + table + deformable duck + ground.
# ----------------------------------------------------------------------------
# %%

def build_scene() -> "newton.Model":
    """Assemble the scene and return the finalized `newton.Model` (meter scale).

    Same scene as the reference example, built with the DECOUPLED world form:
    the ground plane is global (world -1), and the Franka arm + duck + table all
    live in one explicit world (env_0) opened with begin_world / end_world — so
    the arm and the duck share a world and can interact. Returns the finalized
    model with soft-contact material properties applied — no states/solvers yet.
    """
    # Soft-contact material (meter scale).
    particle_radius = 0.005
    soft_contact_ke = 2.0e6
    soft_contact_kd = 1.0e-7
    self_contact_friction = 0.5

    scene = ModelBuilder(gravity=-9.81)

    # Global (world -1): shared ground plane, visible to every world.
    scene.add_ground_plane()
    n_ground = scene.shape_count   # ground shapes: [0, n_ground)

    # --- One explicit world: arm + duck + table go in together (env_0) ------
    scene.begin_world(label="env_0")

    # Franka Panda arm (7 joints + 2 fingers), loaded straight into the world.
    asset_path = newton.utils.download_asset("franka_emika_panda")
    scene.add_urdf(
        str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
        xform=wp.transform((-0.5, -0.5, -0.1), wp.quat_identity()),
        floating=False,
        scale=1.0,  # URDF is in meters
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        force_show_colliders=False,
    )
    scene.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]
    n_franka = scene.shape_count   # franka shapes (incl. fixed-base links @ body -1): [n_ground, n_franka)

    # Deformable rubber duck (tetrahedral mesh from USD) — same world.
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
        k_damp=1.0e-6,
        particle_radius=particle_radius,
    )

    # Static table (body id -1 = fixed to the world frame) — same world.
    scene.add_shape_box(
        -1,
        wp.transform(wp.vec3(0.0, -0.5, 0.1), wp.quat_identity()),
        hx=0.4,
        hy=0.4,
        hz=0.1,
    )
    n_box = scene.shape_count      # table box shapes: [n_franka, n_box)  (duck adds no shapes)

    scene.end_world()         # back to global (-1)

    scene.color()             # graph-color particles for parallel VBD
    model = scene.finalize(requires_grad=False)

    # --- Soft-contact material properties on the finalized model ------------
    model.soft_contact_ke = soft_contact_ke
    model.soft_contact_kd = soft_contact_kd
    model.soft_contact_mu = self_contact_friction
    model.shape_material_ke.fill_(soft_contact_ke)
    model.shape_material_kd.fill_(soft_contact_kd)
    model.shape_material_mu.fill_(1.5)

    # Per-shape object label by build-order range — robust against the Franka's
    # fixed-base links sharing the world body (-1) with the table box. Attached
    # for make_shape_path_fn to build the object-centric entity tree.
    assert model.shape_count == n_box, "shape order/count changed at finalize"
    shape_object = ["franka"] * n_box
    for i in range(n_ground):
        shape_object[i] = "ground"
    for i in range(n_franka, n_box):
        shape_object[i] = "box"
    model._explore_shape_object = shape_object

    return model


# %%

model = build_scene()
print("Step 2 done — scene built.")
print(f"  bodies   : {model.body_count}")
print(f"  shapes   : {model.shape_count}")
print(f"  particles: {model.particle_count}  (duck soft-body verts)")
print(f"  joints   : {model.joint_count}  | dofs: {model.joint_dof_count}")


# ----------------------------------------------------------------------------
# STEP 3 — send the static scene to Rerun (initial frame at t=0).
# ----------------------------------------------------------------------------
# %%

def _surface_edges(model) -> np.ndarray:
    """Unique undirected edges of the soft-mesh surface, as (E, 2) vertex indices.

    `log_state` renders the duck as a filled `Mesh3D` (`/model/triangles`) — Rerun
    draws faces, not edges. We derive the wireframe edges from the surface triangle
    list `model.tri_indices`: each triangle (a,b,c) contributes edges (a,b),(b,c),
    (c,a); we sort each pair and dedupe so shared edges aren't drawn twice.
    """
    tris = model.tri_indices.numpy().reshape(-1, 3)
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
    e = np.unique(np.sort(e, axis=1), axis=0)
    return e


def make_shape_path_fn(model):
    """Map each shape to an object-centric Rerun entity path for this scene.

    Returns a function (shape_index, group, batch_index) -> path, installed on
    `viewer.shape_path_fn`. Uses `model._explore_shape_object` (built by
    build_scene from per-object shape-index ranges) so the Franka's fixed-base
    links — which share the world body (-1) with the table box — are still
    correctly attributed to the arm:
      * franka -> /franka/<group>/shape_N   (visual + collision)
      * ground -> /ground
      * box    -> /box
    The duck (particles, not in the shape system) is logged separately under /duck.
    """
    obj = model._explore_shape_object

    def fn(s, group, batch_index):
        o = obj[s]
        if o == "franka":
            return f"/franka/{group}/shape_{batch_index}"
        return f"/{o}"   # /ground or /box (single static leaf)

    return fn


def log_mesh_edges(viewer, model, state, edges, name="/duck/edge",
                   color=(0.12, 0.12, 0.12), width=-0.5):
    """Log the soft-mesh wireframe for the current frame via `rr.LineStrips3D`.

    Must be called inside a begin_frame/end_frame bracket (like log_state) so the
    edges land on the current timeline frame. `edges` is the precomputed (E,2)
    index array from `_surface_edges` — already deduped, so each undirected edge
    is sent exactly once (topology is constant; only positions move).

    `width` maps to Rerun's `radii`: NEGATIVE = screen-space UI points (crisp,
    constant thin lines — e.g. -0.5), POSITIVE = world-space tube radius in
    meters (renders with a min on-screen size, so it looks thick at this scale
    and balloons where edges meet). Screen-space is what you want for a wireframe.

    Note: the wireframe sits ON the filled `/model/triangles` mesh, so the duck's
    BACK edges can show through the front (looks like doubled/thicker lines on
    thin features). To see a clean wireframe, toggle `/model/triangles` off in
    the Rerun entity tree, or set `viewer.show_triangles = False`.
    """
    pq = state.particle_q.numpy()
    starts = wp.array(pq[edges[:, 0]], dtype=wp.vec3)
    ends = wp.array(pq[edges[:, 1]], dtype=wp.vec3)
    viewer.log_lines(name, starts, ends, color, width=width)


def log_initial_frame(viewer, model, show_collision=True) -> "newton.State":
    """Upload the model to Rerun and log the rest-pose frame at t=0.

    `set_model` uploads the static rigid meshes once; `eval_fk` turns the
    initial joint coordinates into body transforms; `log_state` then logs the
    per-frame body transforms + deformable particle positions. We additionally
    log the duck's surface wireframe with `log_mesh_edges`. Returns the State so
    later steps can keep stepping it.

    show_collision=True also renders the Franka's COLLIDE_SHAPES geometry (the
    convex hulls / primitives used for contact). The entity tree is organized by
    object: /franka/{visual,collision}, /box, /ground, and the duck under
    /duck/{mesh,edge}. Returns the State so later steps can keep stepping it.
    """
    viewer.show_collision = show_collision   # render collision geometry (read by set_model)
    viewer.show_visual = True                # keep the visual meshes too
    viewer.show_triangles = False            # we log the duck mesh ourselves under /duck/mesh
    viewer.shape_path_fn = make_shape_path_fn(model)  # object-centric shape paths (read by set_model)

    viewer.set_model(model)
    viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

    state = model.state()
    eval_fk(model, model.joint_q, model.joint_qd, state)

    edges = _surface_edges(model)
    viewer.begin_frame(0.0)
    viewer.log_state(state)   # franka/box/ground poses (no /model/triangles — disabled above)
    # Duck: filled surface + wireframe, grouped under /duck.
    viewer.log_mesh("/duck/mesh", state.particle_q, model.tri_indices.flatten(), backface_culling=False)
    log_mesh_edges(viewer, model, state, edges, name="/duck/edge")
    viewer.end_frame()
    return state


# %%

state = log_initial_frame(viewer, model)
print("Step 3 done — rest-pose scene sent to Rerun at t=0.")

# %%
