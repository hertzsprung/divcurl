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

PRIMAL = [bottom_primal_face, right_primal_face, top_primal_face, left_primal_face]

def faces(mesh, i, j):
    return np.array([fun(i, j) for fun in mesh])

def cell(i, j):
    return i*N + j

def div(i, j):
    f = flux[faces(PRIMAL, i, j)] * NORMALS
    return f.sum()

#def curl(i_, j_):
#    dual_faces(i_, j_)

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

print(div_all())
