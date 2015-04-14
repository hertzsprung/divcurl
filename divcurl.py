#!/usr/bin/env python3
import numpy as np

N = 2
h = 1.0/N
CELLS = N*N
FACES = N**2 * (N+1)**2
# normal orientation of bottom, right, top, left faces
NORMALS = np.array([-h, h, h, -h])

flux = np.zeros(FACES)

def bottom_face(i, j):
    return i + (2*N+1)*j

def top_face(i, j):
    return i + (2*N+1)*(j+1)

def left_face(i, j):
    return i + N+(2*N+1)*j

def right_face(i, j):
    return i + N+(2*N+1)*j + 1

def primal_faces(i, j):
    return np.array([bottom_face(i, j), \
            right_face(i, j), \
            top_face(i, j), \
            left_face(i, j)])

def cell(i, j):
    return i*N + j

def div(i, j):
    f = flux[primal_faces(i, j)] * NORMALS
    return f.sum()

def div_all():
    normal_coeffs = np.zeros(shape=[CELLS,FACES])
    for j in range(0,N):
        for i in range(0,N):
            faces = primal_faces(i, j)
            for idx, f in enumerate(faces):
                normal_coeffs[cell(i,j), f] = flux[f] * NORMALS[idx]

    face_areas = np.ones(FACES)
    return normal_coeffs.dot(face_areas)

flux[bottom_face(0, 0)] = -2.0
flux[top_face(0, 0)] = 1.0
flux[left_face(0, 0)] = -4.0
flux[right_face(0, 0)] = 8.0

print(div_all())
