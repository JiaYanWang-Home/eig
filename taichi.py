import numpy as np

import taichi as ti

ti.init(arch=ti.cpu,debug=True)  # Try to run on GPU
# 在mpm128基础上只保留弹性部分，增加了重力，出现了数组越界，不知道是哪里的越界

n_particles, n_grid = 9000 , 128
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters

# 粒子的变量
x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float,
                    shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float,
                    shape=n_particles)  # deformation gradient
J = ti.field(dtype=float, shape=n_particles)  # plastic deformation
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
p_mark = ti.field(dtype=int,shape=n_particles)

# 网格的变量
grid_v = ti.Vector.field(2, dtype=float,
                         shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass

gravity = ti.Vector.field(2, dtype=float, shape=())

@ti.kernel
def P2G():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

        F[p] = (ti.Matrix.identity(float, 2) +
                dt * C[p]) @ F[p]  # deformation gradient update

        U, sig, V = ti.svd(F[p])
        J[p] = 1.0
        for d in ti.static(range(2)):
            J[p] *= sig[d, d]
        # pk1 = 2.0 * mu_0 * (F[p] - U @ V.transpose()) + lambda_0 * (J[p] - 1.0) * J[p] * \
        #       (F[p].transpose()).inverse()
        # print(F[p],U@V.transpose(),pk1,J[p])
        # stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * (pk1 @ F[p].transpose())
        stress = 2 * mu_0 * (F[p] - U @ V.transpose()) @ F[p].transpose(
        ) + ti.Matrix.identity(float, 2) * lambda_0 * J[p] * (J[p] - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(
                3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

@ti.kernel
def Grid_Operator():
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i,j] = (1 / grid_m[i, j]) * grid_v[i,j]  # Momentum to velocity

            grid_v[i, j] += dt * gravity[None] * 3000  # gravity

            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0

@ti.kernel
def G2P():
    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(
                3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection



@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [
            ti.random() * 0.4 + 0.3,   # 0.3-0.7
            ti.random() * 0.2 + 0.5    # 0.2-0.7
        ]
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0],
                          [0, 1]])
        J[i] = 1.0
        C[i] = ti.Matrix.zero(float, 2, 2)
        p_mark[i] = 0
        if x[i][0] > 0.49 and x[i][0] < 0.51 and x[i][1] >0.6:
            p_mark[i] = 1
        if x[i][0] < 0.35:
            p_mark[i] = 2
        if x[i][0] > 0.65:
            p_mark[i] = 3





gui = ti.GUI("fixed-corotated", res=512, background_color=0x112F41)
init()
gravity[None] = [0, -1]
while gui.running:
    for s in range(int(2e-3 // dt)):
        P2G()
        Grid_Operator()
        G2P()


    colors = np.array([0x0000FF, 0xFF0000, 0x00FF00, 0xB8860B])
    gui.circles(x.to_numpy(),radius=1.5,color=colors[p_mark.to_numpy()])
    gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk
