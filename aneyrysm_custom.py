import os
import warnings
import torch
import numpy as np
from sympy import Symbol, sqrt, Max

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

# --- CONFIGURATION ---
# We override the config defaults here for simplicity
BATCH_SIZE_INTERIOR = 4000  # <--- Your Interior Points
BATCH_SIZE_WALL = 2000
BATCH_SIZE_INLET = 500
BATCH_SIZE_OUTLET = 500
MAX_STEPS = 5000            # How long to train

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    print("------------------------------------------------")
    print("🚀 STARTING CUSTOM Y-JUNCTION TRAINING")
    print(f"   Interior Points: {BATCH_SIZE_INTERIOR}")
    print("------------------------------------------------")

    # 1. LOAD GEOMETRY
    # We look for files in the CURRENT directory
    point_path = to_absolute_path(".") 
    
    # Load your verified STLs
    inlet_mesh = Tessellation.from_stl(os.path.join(point_path, "inlet.stl"), airtight=False)
    outlet_mesh = Tessellation.from_stl(os.path.join(point_path, "outlets.stl"), airtight=False)
    wall_mesh = Tessellation.from_stl(os.path.join(point_path, "walls.stl"), airtight=False)
    interior_mesh = Tessellation.from_stl(os.path.join(point_path, "y_junction_closed.stl"), airtight=True)

    # 2. DEFINE PHYSICS
    # Fluid Properties (Blood-like)
    nu = 0.0035    # Kinematic Viscosity
    rho = 1050     # Density
    inlet_vel = 1.5 # Max velocity m/s

    ns = NavierStokes(nu=nu, rho=rho, dim=3, time=False)

    # 3. DEFINE NEURAL NETWORK
    # Inputs: x,y,z -> Outputs: u,v,w,p
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = ns.make_nodes() + [flow_net.make_node(name="flow_network")]

    # 4. BOUNDARY CONDITIONS
    domain = Domain()

    # --- A. INLET (Parabolic Profile) ---
    # Center (0,0,0), Flow in +X direction. Radius = 5.0
    y, z = Symbol("y"), Symbol("z")
    r_inlet = 5.0
    # Formula: u = Umax * (1 - r^2/R^2)
    parabola = inlet_vel * Max((1 - (sqrt(y**2 + z**2) / r_inlet)**2), 0)

    inlet_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_mesh,
        outvar={"u": parabola, "v": 0, "w": 0},
        batch_size=BATCH_SIZE_INLET,
    )
    domain.add_constraint(inlet_bc, "inlet")

    # --- B. OUTLET (Pressure = 0) ---
    outlet_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_mesh,
        outvar={"p": 0},
        batch_size=BATCH_SIZE_OUTLET,
    )
    domain.add_constraint(outlet_bc, "outlet")

    # --- C. WALLS (No Slip) ---
    wall_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall_mesh,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=BATCH_SIZE_WALL,
    )
    domain.add_constraint(wall_bc, "wall")

    # --- D. INTERIOR (Navier-Stokes Residuals) ---
    # This samples the blue dots you saw in the plot
    interior_bc = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=BATCH_SIZE_INTERIOR,
    )
    domain.add_constraint(interior_bc, "interior")

    # 5. MONITORING (Watch pressure drop while training)
    # We sample the inlet to see if pressure stabilizes
    pressure_monitor = PointwiseMonitor(
        inlet_mesh.sample_boundary(16),
        output_names=["p"],
        metrics={"inlet_pressure": lambda var: torch.mean(var["p"])},
        nodes=nodes,
    )
    domain.add_monitor(pressure_monitor)

    # 6. RUN SOLVER
    # We override the max_steps from the config with our hardcoded value
    cfg.training.max_steps = MAX_STEPS
    
    slv = Solver(cfg, domain)
    slv.solve()

if __name__ == "__main__":
    run()