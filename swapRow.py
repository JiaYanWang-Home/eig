import taichi as ti

ti.init(arch=ti.cpu, debug=True)
Mat = ti.Matrix.field(3,3,float,shape=())

@ti.kernel
def te(A: ti.template()):
    for i in ti.static(range(3)):
        maxErow = i
        max_num = ti.abs(A[i, i])
        # 第i次循环，找到一列中A[i,i]及以下的所有元素中的最大的行
        for j in ti.static(range(3)):
            # maxErow=i
            # max_num=ti.abs(A[i,j])
            if i >= j and ti.abs(A[j, i]) > max_num:
                # print("--")
                max_num = ti.abs(A[j, i])
                maxErow = j

        if maxErow != i:
            swapRow(A, i, maxErow)

@ti.func
def swapRow(A:ti.template(), i:ti.template(), maxRow:ti.template()):
    for k in ti.static(range(3)):
        temp = A[i,k]
        A[i,k] = A[maxRow,k]
        A[maxRow,k] = temp



Mat = ti.Matrix([[4.0, 5.0, -3.0, ], [5.0, 5.0, 0.0], [6.0, 6.0, -9.0]])
te(Mat)
#
