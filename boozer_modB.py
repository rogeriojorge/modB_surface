#!/usr/bin/env python3

import os
import numpy as np
from simsopt.geo import BoozerSurface, boozer_surface_residual, ToroidalFlux, Area, SurfaceRZFourier
from simsopt.mhd import Vmec
from simsopt import load
from booz_xform_init import init_booz_surf
import matplotlib.pyplot as plt

this_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_path)

filename_wout = f'wout_LandremanPaul2021_QA.nc'
coils_file = f'biot_savart_opt_LandremanPaul2021_QA.json'

mpol = 12
ntor = 12
nphi = 2*mpol+1
ntheta = 2*ntor+1
vmec = Vmec(filename_wout)
s = init_booz_surf(vmec, mpol=mpol, ntor=ntor, ntheta=ntheta, nphi=nphi)

bs = load(coils_file)
bs_tf = load(coils_file)
coils = bs.coils
current_sum = sum(abs(c.current.get_value()) for c in coils)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
iota = vmec.iota_edge()

tf = ToroidalFlux(s, bs_tf)
ar = Area(s)
ar_target = ar.J()

boozer_surface = BoozerSurface(bs, s, ar, ar_target)

# compute surface first using LBFGS, this will just be a rough initial guess
res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-10, maxiter=300, constraint_weight=100., iota=iota, G=G0)
print(f"After LBFGS:   iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")

# now drive the residual down using a specialised least squares algorithm
res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=100, constraint_weight=100., iota=res['iota'], G=res['G'], method='manual')
print(f"After Lev-Mar: iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
# s.plot()

bs.set_points(s.gamma().reshape((-1, 3)))

# this is the modB array you want
modB = bs.AbsB().reshape(s.gamma().shape[:2])

plt.contourf(modB.transpose(), levels=20)
plt.xlabel(r'Boozer $\phi$')
plt.ylabel(r'Boozer $\theta$')
plt.colorbar()
plt.savefig('modB.png', dpi=300)
plt.show()

# to vtk
s.to_vtk('surface', extra_data={'modB':modB[..., None]})
