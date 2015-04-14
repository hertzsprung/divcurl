#!/usr/bin/env python3
import numpy as np

N = 2
h = 1.0/N
CELLS = N*N
FACES = N**2 * (N+1)**2

# bottom, right, top, left faces
NORMALS = np.array([-h, h, h, -h])
TANGENTS = np.array([1, 1, -1, -1])

flux = np.zeros(FACES)

def bottom_primal_face(i, j):
    return i + (2*N+1)*j

def top_primal_face(i, j):
    return i + (2*N+1)*(j+1)

def left_primal_face(i, j):
    return i + N+(2*N+1)*j

def right_primal_face(i, j):
    return i + N+(2*N+1)*j + 1

def bottom_dual_face(i_, j_):
    return right_primal_face(i_, j_)

def right_dual_face(i_, j_):
    return top_primal_face(i_+1, j_)

def top_dual_face(i_, j_):
    return right_primal_face(i_, j_+1)

def left_dual_face(i_, j_):
    return top_primal_face(i_, j_)

PRIMAL = [bottom_primal_face, right_primal_face, top_primal_face, left_primal_face]
DUAL = [bottom_dual_face, right_dual_face, top_dual_face, left_dual_face]

def faces(mesh, i, j):
    return np.array([fun(i, j) for fun in mesh])

def cell(i, j):
    return i*N + j

def div(i, j):
    d = flux[faces(PRIMAL, i, j)] * NORMALS
    return d.sum()

def curl(i_, j_):
    c = flux[faces(DUAL, i_, j_)] * TANGENTS
    return c.sum()

def div_all():
    normal_coeffs = np.zeros(shape=[CELLS,FACES])
    for j in range(0,N):
        for i in range(0,N):
            face_indices = faces(PRIMAL, i, j)
            for idx, f in enumerate(face_indices):
                normal_coeffs[cell(i,j), f] = flux[f] * NORMALS[idx]

    face_areas = np.ones(FACES)
    return normal_coeffs.dot(face_areas)

flux[bottom_primal_face(0, 0)] = -2.0
flux[top_primal_face(0, 0)] = 1.0
flux[left_primal_face(0, 0)] = -4.0
flux[right_primal_face(0, 0)] = 8.0
flux[top_primal_face(1, 0)] = 16.0
flux[right_primal_face(0, 1)] = -32.0

print(div_all())
print(curl(0,0))
