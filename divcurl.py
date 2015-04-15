#!/usr/bin/env python3
import numpy as np

class Mesh:
    def __init__(self, N):
        self.N = N
        self.h = 1.0/N
        self.PRIMAL_CELLS = N**2
        self.PRIMAL_FACES = N**2 * (N+1)**2
        self.DUAL_CELLS = (N-1)**2
        self.DUAL_FACES = (N-1)**2 + N**2
        self.PRIMAL = [self.bottom_primal_face, self.right_primal_face, self.top_primal_face, self.left_primal_face]
        self.DUAL = [self.bottom_dual_face, self.right_dual_face, self.top_dual_face, self.left_dual_face]

    def bottom_primal_face(self, i, j):
        return i + (2*self.N+1)*j

    def top_primal_face(self, i, j):
        return i + (2*self.N+1)*(j+1)

    def left_primal_face(self, i, j):
        return i + self.N+(2*self.N+1)*j

    def right_primal_face(self, i, j):
        return i + self.N+(2*self.N+1)*j + 1

    def bottom_dual_face(self, i_, j_):
        return self.right_primal_face(i_, j_)

    def right_dual_face(self, i_, j_):
        return self.top_primal_face(i_+1, j_)

    def top_dual_face(self, i_, j_):
        return self.right_primal_face(i_, j_+1)

    def left_dual_face(self, i_, j_):
        return self.top_primal_face(i_, j_)

    def faces(self, mesh, i, j):
        return np.array([fun(i, j) for fun in mesh])

    def primal_cell(self, i, j):
        return i*self.N + j

    def dual_cell(self, i_, j_):
        return i_*(self.N-1) + j_

mesh = Mesh(3)

# bottom, right, top, left faces
NORMALS = np.array([-1, 1, 1, -1])
TANGENTS = np.array([1, 1, -1, -1])

flux = np.zeros(mesh.PRIMAL_FACES)

def div(i, j):
    d = flux[mesh.faces(mesh.PRIMAL, i, j)] * NORMALS
    return d.sum() * mesh.h

def curl(i_, j_):
    c = flux[mesh.faces(mesh.DUAL, i_, j_)] * TANGENTS
    return c.sum() * mesh.h

def div_all():
    normal_coeffs = np.zeros(shape=[mesh.PRIMAL_CELLS, mesh.PRIMAL_FACES])
    for j in range(0, mesh.N):
        for i in range(0, mesh.N):
            face_indices = mesh.faces(mesh.PRIMAL, i, j)
            for idx, f in enumerate(face_indices):
                normal_coeffs[mesh.primal_cell(i, j), f] = flux[f] * NORMALS[idx]

    face_areas = np.full(mesh.PRIMAL_FACES, mesh.h)
    return normal_coeffs.dot(face_areas)

def curl_all():
    tangent_coeffs = np.zeros(shape=[mesh.DUAL_CELLS, mesh.DUAL_FACES])
    for j_ in range(0, mesh.N-1):
        for i_ in range(0, mesh.N-1):
            index_mesh = Mesh(mesh.N-1)
            index_faces = index_mesh.faces(index_mesh.PRIMAL, i_, j_)
            dual_face_indices = mesh.faces(mesh.DUAL, i_, j_)
                
            for idx, f in enumerate(zip(index_faces, dual_face_indices)):
                tangent_coeffs[mesh.dual_cell(i_, j_), f[0]] = flux[f[1]] * TANGENTS[idx]

    face_areas = np.full(mesh.DUAL_FACES, mesh.h)
    return tangent_coeffs.dot(face_areas)

flux[mesh.bottom_primal_face(0, 0)] = -2.0
flux[mesh.top_primal_face(0, 0)] = 1.0
flux[mesh.left_primal_face(0, 0)] = -4.0
flux[mesh.right_primal_face(0, 0)] = 8.0
flux[mesh.top_primal_face(1, 0)] = 16.0
flux[mesh.right_primal_face(0, 1)] = -32.0

print("div all:", div_all())
print("curl all:", curl_all())
