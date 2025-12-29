import os
import torch
import numpy as np
from sympy import Symbol, sqrt, Max, sin, cos

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.geometry.tessellation import Tessellation

# --- HYPERPARAMETERS ---
# We define these here so you don't have to dig through YAML files yet.
BATCH_INTERIOR = 4000
BATCH_WALL     = 2000
BATCH_INLET    = 500
BATCH_OUTLET   = 500
MAX_STEPS      = 5000

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    print("-" * 50)
    print("🚀 LAUNCHING Y-JUNCTION SIMULATION")
    print("-" * 50)

    # 1. LOAD GEOMETRY
    # to_absolute_path(".") ensures we look in the current folder, not Hydra's temp folder
    repo_path = to_absolute_path(".")
    
    # Load the 4 separate files
    # airtight=False for open surfaces, True for the closed volume
    inlet_mesh = Tessellation.from_stl(os.path.join(repo_path, "inlet.stl"), airtight=False)
    outlet_mesh = Tessellation.from_stl(os.path.join(repo_path, "outlets.stl"), airtight=False)
    wall_mesh = Tessellation.from_stl(os.path.join(repo_path, "walls.stl"), airtight=False)
    interior_mesh = Tessellation.from_stl(os.path.join(repo_path, "y_junction_closed.stl"), airtight=True)

    # 2. DEFINE PHYSICS
    # Fluid: Blood-like properties
    nu = 0.0035     # Kinematic Viscosity
    rho = 1050      # Density
    inlet_vel = 1.5 # Max velocity (m/s)

    # 3D Navier-Stokes (Steady State)
    ns = NavierStokes(nu=nu, rho=rho, dim=3, time=False)

    # 3. BUILD NEURAL NETWORK
    # Inputs: x,y,z | Outputs: u,v,w,p
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = ns.make_nodes() + [flow_net.make_node(name="flow_network")]

    # 4. CONSTRAINTS (Boundary Conditions)
    domain = Domain()

    # --- INLET: Parabolic Profile ---
    # Our inlet is centered at (0,0,0) facing +X. 
    # Velocity depends on radial distance from X-axis (r = sqrt(y^2 + z^2))
    y, z = Symbol("y"), Symbol("z")
    r_inlet = 5.0
    # Parabola formula: U_max * (1 - (r/R)^2)
    # We use Max(..., 0) to ensure velocity doesn't go negative at the very edge
    profile = inlet_vel * Max((1 - (sqrt(y**2 + z**2) / r_inlet)**2), 0)

    inlet_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_mesh,
        outvar={"u": profile, "v": 0, "w": 0}, # Flow only in X
        batch_size=BATCH_INLET,
    )
    domain.add_constraint(inlet_bc, "Inlet")

    # --- OUTLET: Zero Pressure ---
    outlet_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_mesh,
        outvar={"p": 0},
        batch_size=BATCH_OUTLET,
    )
    domain.add_constraint(outlet_bc, "Outlet")

    # --- WALLS: No Slip (Velocity = 0) ---
    wall_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall_mesh,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=BATCH_WALL,
    )
    domain.add_constraint(wall_bc, "Wall")

    # --- INTERIOR: Physics Residuals ---
    # Samples random points inside the pipe to ensure they obey Navier-Stokes
    interior_bc = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=BATCH_INTERIOR,
    )
    domain.add_constraint(interior_bc, "Interior")

    # 5. MONITORING
    # Track the average pressure at the inlet during training
    pressure_monitor = PointwiseMonitor(
        inlet_mesh.sample_boundary(16),
        output_names=["p"],
        metrics={"avg_inlet_pressure": lambda var: torch.mean(var["p"])},
        nodes=nodes,
    )
    domain.add_monitor(pressure_monitor)

    # 6. RUN TRAINING
    # Override max_steps from config
    cfg.training.max_steps = MAX_STEPS
    
    slv = Solver(cfg, domain)
    slv.solve()

if __name__ == "__main__":
    run()
