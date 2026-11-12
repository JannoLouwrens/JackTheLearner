import sys, os
os.chdir("/home/opc/jackthelearner"); sys.path.insert(0, "/home/opc/jackthelearner")
import numpy as np, mujoco
from experiments import body_probe as B
B.BODY = "plinth"; B.install_body("plinth")
from experiments.tests.lf_03_learner_lives import _calibration
from experiments.tests.lf_04_living_teaches import _Life4
j0, a, _ = _calibration()
L = _Life4(90, j0, a, train=False); w = L.w; m, d = w.model, w.data
mujoco.mj_forward(m, d)
nm = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)
for g in range(m.ngeom):
    n = nm(g)
    if n.startswith("pool") or n == "floor" or n == "rover_foot":
        print(n, "type", int(m.geom_type[g]), "pos", np.round(d.geom_xpos[g], 3), "size", np.round(m.geom_size[g], 3),
              "contype", int(m.geom_contype[g]), "conaff", int(m.geom_conaffinity[g]), "margin", float(m.geom_margin[g]))
for n, (g, _) in w.drives._food.items():
    print("food", n, np.round(d.geom_xpos[g], 2))
print("opt", m.opt.timestep, "noslip", m.opt.noslip_iterations, "cone", m.opt.cone)
