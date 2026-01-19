import os
import torch
import numpy as np
from sympy import Symbol, sqrt, Max

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.geometry.tessellation import Tessellation

# --- SETTINGS ---
DEBUG_MODE = False   # Set to True to test quickly (runs only 100 steps)
INLET_RADIUS = 5.0   # <--- UPDATE THIS to match your CAD geometry radius!

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    print("---------------------------------------")
    print(f"🚀 STARTING T-JUNCTION TRAINING")
    print(f"   Inlet Radius: {INLET_RADIUS}")
    print(f"   Debug Mode:   {DEBUG_MODE}")
    print("---------------------------------------")

    # 1. LOAD GEOMETRY
    path = to_absolute_path(".")
    
    # Ensure these match your uploaded filenames exactly
    inlet_mesh = Tessellation.from_stl(os.path.join(path, "inlet.stl"), airtight=False)
    outlet_mesh = Tessellation.from_stl(os.path.join(path, "outlets.stl"), airtight=False)
    wall_mesh = Tessellation.from_stl(os.path.join(path, "walls.stl"), airtight=False)
    interior_mesh = Tessellation.from_stl(os.path.join(path, "y_junction_closed.stl"), airtight=True)

    # 2. DEFINE PHYSICS
    # Blood-like properties (Kinematic Viscosity & Density)
    ns = NavierStokes(nu=0.0035, rho=1050, dim=3, time=False)

    # 3. NEURAL NETWORK
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = ns.make_nodes() + [flow_net.make_node(name="flow_network")]

    # 4. BOUNDARY CONDITIONS
    domain = Domain()
    y, z = Symbol("y"), Symbol("z")

    # Inlet: Parabolic Profile
    # Formula: Umax * (1 - (r/R)^2)
    profile = 1.5 * Max((1 - (sqrt(y**2 + z**2) / INLET_RADIUS)**2), 0)

    domain.add_constraint(PointwiseBoundaryConstraint(
        nodes=nodes, geometry=inlet_mesh, outvar={"u": profile, "v": 0, "w": 0}, 
        batch_size=500
    ), "Inlet")

    domain.add_constraint(PointwiseBoundaryConstraint(
        nodes=nodes, geometry=outlet_mesh, outvar={"p": 0}, 
        batch_size=500
    ), "Outlet")

    domain.add_constraint(PointwiseBoundaryConstraint(
        nodes=nodes, geometry=wall_mesh, outvar={"u": 0, "v": 0, "w": 0}, 
        batch_size=2000
    ), "Wall")

    domain.add_constraint(PointwiseInteriorConstraint(
        nodes=nodes, geometry=interior_mesh, 
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0}, 
        batch_size=4000
    ), "Interior")

    # 5. RUN SOLVER
    # Override max_steps if debugging
    if DEBUG_MODE:
        cfg.training.max_steps = 100
        cfg.training.rec_results_freq = 50
        print("⚠️ DEBUG MODE ON: Training will stop after 100 steps.")
    
    slv = Solver(cfg, domain)
    slv.solve()

if __name__ == "__main__":
    run()
