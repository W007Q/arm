"""用 CAD 三角网格复核 MuJoCo 凸包产生的自碰撞候选。"""
import hppfcl
import mujoco
import numpy as np


class MeshCollisionCheck:
    def __init__(self, model, robot_bodies):
        self.model = model
        self.meshes = {}
        # 在启动时建好 BVH，避免首次运动时加载大网格、打断控制心跳。
        for geom in range(model.ngeom):
            # HardwareScene can enable the camera after this cache is built.
            # Include initially disabled meshes so a later self-contact can
            # still be checked; shared visual/collision assets are cached once.
            if (int(model.geom_bodyid[geom]) not in robot_bodies
                    or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mesh_id = int(model.geom_dataid[geom])
            if mesh_id in self.meshes:
                continue
            va, nv = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            fa, nf = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
            vertices = hppfcl.StdVec_Vec3s()
            vertices.extend(model.mesh_vert[va:va + nv].astype(float))
            faces = hppfcl.StdVec_Triangle()
            faces.extend([hppfcl.Triangle(*(int(v) for v in face))
                          for face in model.mesh_face[fa:fa + nf]])
            mesh = hppfcl.BVHModelOBBRSS()
            mesh.beginModel(nf, nv)
            mesh.addSubModel(vertices, faces)
            mesh.endModel()
            self.meshes[mesh_id] = mesh
        self.request = hppfcl.CollisionRequest()

    def collides(self, data, geom_a, geom_b):
        """返回网格相交结果；非双网格保留原有保守检查（返回 None）。"""
        ids = (int(geom_a), int(geom_b))
        if any(self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH for g in ids):
            return None
        meshes = [self.meshes[int(self.model.geom_dataid[g])] for g in ids]
        # mesh_vert 已包含 MuJoCo 编译时的缩放/中心变换，直接配 geom 位姿。
        poses = [hppfcl.Transform3f(data.geom_xmat[g].reshape(3, 3),
                                   np.asarray(data.geom_xpos[g])) for g in ids]
        result = hppfcl.CollisionResult()
        hppfcl.collide(meshes[0], poses[0], meshes[1], poses[1], self.request, result)
        return result.isCollision()
