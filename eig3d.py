import taichi as ti
ti.init(arch=ti.gpu)
eig_values = ti.field(ti.f32, shape=3)
eig_vectors = ti.field(ti.f32, shape=(3,3))

@ti.func
def QR3(Mat: ti.template()):  # 3x3 mat, Gram–Schmidt Orthogonalization
    # 原理参考：https://blog.csdn.net/jhshanvip/article/details/105896970
    a1 = ti.Vector([Mat[0, 0], Mat[1, 0], Mat[2, 0]])
    a2 = ti.Vector([Mat[0, 1], Mat[1, 1], Mat[2, 1]])
    a3 = ti.Vector([Mat[0, 2], Mat[1, 2], Mat[2, 2]])

    r11 = a1.norm()
    q1 = a1 / r11

    r12 = a2.dot(q1)
    q2 = a2 - r12 * q1
    r22 = q2.norm()
    q2 /= r22

    r13 = a3.dot(q1)
    r23 = a3.dot(q2)
    q3 = a3 - (q1 * r13 + q2 * r23)
    r33 = q3.norm()
    q3 /= r33

    Q = ti.Matrix.cols([q1, q2, q3])
    R = ti.Matrix([[r11, r12, r13], [0, r22, r23], [0, 0, r33]])
    return Q, R

@ti.kernel
def eig_qr(A:ti.template()):
    Ak = A
    Ak0 = ti.Matrix.zero(ti.f32, 3, 3)
    flag = 1
    eps = 1e-7
    n = 0
    while flag:
        Ak0 = Ak
        q, r = QR3(Ak)
        Ak = r @ q
        diag = ti.Vector([ti.abs(Ak - Ak0)[0, 0], ti.abs(Ak - Ak0)[1, 1], ti.abs(Ak - Ak0)[2, 2]])
        if diag.sum() < eps:
            flag = 0
        n = n + 1
    for i in ti.static(range(3)):
        eig_values[i] = Ak[i, i]


mat = ti.Matrix([[-2.0, 1.0, 1.0], [0.0, 2.0, 0.0], [-4.0, 1.0, 3.0]])


eig_qr(mat)
print(eig_values)
