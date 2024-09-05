#!/usr/bin/env python3
import numpy as np
import booz_xform as bx
from simsopt.mhd import Vmec, Boozer
from simsopt.geo import SurfaceXYZTensorFourier

def init_booz_surf(vmec: Vmec, surf_s=0.99, mpol_booz=64, ntor_booz=64, ntheta=30, nphi=30, mpol=10, ntor=10):
    b = Boozer(vmec, mpol=mpol_booz, ntor=ntor_booz)
    b.register([surf_s])
    b.run()

    nfp = b.bx.nfp
    theta1D = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi1D = np.linspace(0, 2 * np.pi/nfp, nphi, endpoint=False)
    varphi, theta = np.meshgrid(phi1D, theta1D, indexing='ij')

    R = np.zeros_like(theta)
    Z = np.zeros_like(theta)
    nu = np.zeros_like(theta)

    js = 0
    for jmn in range(b.bx.mnboz):
        m = b.bx.xm_b[jmn]
        n = b.bx.xn_b[jmn]
        angle = m * theta - n * varphi
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        R += b.bx.rmnc_b[jmn, js] * cosangle
        Z += b.bx.zmns_b[jmn, js] * sinangle
        nu += b.bx.numns_b[jmn, js] * sinangle
        if b.bx.asym:
            R += b.bx.rmns_b[jmn, js] * sinangle
            Z += b.bx.zmnc_b[jmn, js] * cosangle
            nu += b.bx.numnc_b[jmn, js] * cosangle

    # Following the sign convention in the code, to convert from the
    # Boozer toroidal angle to the standard toroidal angle, we
    # *subtract* nu:
    phi = varphi - nu
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    XYZ = np.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1)
    s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, quadpoints_phi=phi1D/2/np.pi, quadpoints_theta=theta1D/2/np.pi,
            stellsym=True, nfp=b.bx.nfp)
    s.least_squares_fit(XYZ)
    return s