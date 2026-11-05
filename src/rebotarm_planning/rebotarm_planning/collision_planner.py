"""Bounded joint-space RRT with MuJoCo collision checks on sampled edges.

Uses a private model/data so checking a path never moves the simulated robot.
The scene is a known simulation map, not a general perception-based world model.
"""
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# 零位反馈量化会把模型的微米级间隙变成浅穿透；容差为 0.05 mm。
# 只用于自接触，外部障碍物仍要求 2 mm 余量。
SELF_CONTACT_TOLERANCE_M = 0.00005


class PlanningError(RuntimeError):
    pass


class CollisionPlanner:
    def __init__(self, model_path, resolution=.025, clearance=.002, *, exact_meshes=False):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.joints = [self.model.joint(f'joint{i}') for i in range(1, 7)]
        self.addr = np.array([int(j.qposadr[0]) for j in self.joints])
        self.lower = np.array([j.range[0] for j in self.joints])
        self.upper = np.array([j.range[1] for j in self.joints])
        self.resolution = resolution
        self.model.geom_margin[:] = np.maximum(self.model.geom_margin, clearance)
        root = self.model.body('base_link').id
        self.robot = set()
        for body in range(1, self.model.nbody):
            ancestor = body
            while ancestor:
                if ancestor == root:
                    self.robot.add(body)
                    break
                ancestor = int(self.model.body_parentid[ancestor])
        self.fingers = {self.model.body(n).id for n in ['gripper_left', 'gripper_right']}
        self.allowed_object = None
        self.last_collision = ''
        self.carried = None
        self.hand = self.model.body('gripper_end').id
        self.mesh_check = None
        if exact_meshes:
            from .mesh_collision import MeshCollisionCheck
            self.mesh_check = MeshCollisionCheck(self.model, self.robot)

    def capture_grasp(self, measured):
        """Predict a payload with the hand only while both fingers contact it."""
        self.carried = None
        self.data.qpos[self.addr] = measured
        mujoco.mj_forward(self.model, self.data)
        if self.allowed_object is None:
            return
        touching = set()
        supported = False
        for c in self.data.contact:
            a, b = int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])
            if c.dist > .001:
                continue
            if a == self.allowed_object and b in self.fingers:
                touching.add(b)
            if b == self.allowed_object and a in self.fingers:
                touching.add(a)
            # During the approach both fingers may brush an object that is
            # still resting on the table. Treating it as a carried load at
            # that point makes its real support contact look like a collision.
            if (a == self.allowed_object and b not in self.robot) or (
                b == self.allowed_object and a not in self.robot
            ):
                supported = True
        if touching != self.fingers or supported:
            return
        obj = self.allowed_object
        hand_r = self.data.xmat[self.hand].reshape(3, 3)
        offset = hand_r.T @ (self.data.xpos[obj]-self.data.xpos[self.hand])
        relative_r = hand_r.T @ self.data.xmat[obj].reshape(3, 3)
        adr = int(self.model.jnt_qposadr[self.model.body_jntadr[obj]])
        self.carried = (obj, adr, offset, relative_r)

    def valid(self, q):
        self.last_collision = ''
        if not np.all(np.isfinite(q)) or np.any(q < self.lower) or np.any(q > self.upper):
            self.last_collision = 'nonfinite or out-of-limit joint target'
            return False
        self.data.qpos[self.addr] = q
        mujoco.mj_forward(self.model, self.data)
        if self.carried is not None:
            obj, adr, offset, relative_r = self.carried
            hand_r = self.data.xmat[self.hand].reshape(3, 3)
            self.data.qpos[adr:adr+3] = self.data.xpos[self.hand] + hand_r @ offset
            xyzw = Rotation.from_matrix(hand_r @ relative_r).as_quat()
            self.data.qpos[adr+3:adr+7] = xyzw[[3, 0, 1, 2]]
            mujoco.mj_forward(self.model, self.data)
        checked_mesh_pairs = {}
        for contact in self.data.contact:
            a, b = int(self.model.geom_bodyid[contact.geom1]), int(self.model.geom_bodyid[contact.geom2])
            if a not in self.robot and b not in self.robot:
                # A carried object is part of the swept volume. Permit its
                # initial resting contact, but reject penetration of the map.
                if self.carried is not None and self.carried[0] in (a, b) and contact.dist < -.0005:
                    self.last_collision = f'payload: {self.model.body(a).name} / {self.model.body(b).name}'
                    return False
                continue
            if self.allowed_object is not None and ((a in self.fingers and b == self.allowed_object) or (b in self.fingers and a == self.allowed_object)):
                continue
            # 保留实测角度；仅容许量化/网格误差造成的极浅自接触。
            # 机械限位接触另由场景 exclude 指定。
            clearance = -SELF_CONTACT_TOLERANCE_M if a in self.robot and b in self.robot else .002
            if contact.dist < clearance:
                exact = None
                if self.mesh_check is not None and a in self.robot and b in self.robot:
                    pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
                    if pair not in checked_mesh_pairs:
                        checked_mesh_pairs[pair] = self.mesh_check.collides(self.data, *pair)
                    exact = checked_mesh_pairs[pair]
                    if exact is False:
                        continue  # 凸包覆盖了 CAD 的空隙，实际三角网格没有相交。
                self.last_collision = (f'{self.model.body(a).name} / {self.model.body(b).name} '
                                       f'(distance={contact.dist * 1000:.4f} mm, '
                                       f'minimum={clearance * 1000:.4f} mm)')
                if exact:
                    self.last_collision += ' [CAD triangle meshes intersect; distance above is convex-hull distance]'
                return False
        return True

    def edge(self, a, b):
        steps = max(1, int(np.ceil(np.max(np.abs(b-a))/self.resolution)))
        return all(self.valid(a+(b-a)*s/steps) for s in range(steps+1))

    def plan(self, start, goal, timeout=3., seed=0):
        start, goal = np.asarray(start), np.asarray(goal)
        if not self.valid(start):
            raise PlanningError(f'start in collision: {self.last_collision}')
        if not self.valid(goal):
            raise PlanningError(f'goal in collision: {self.last_collision}')
        if self.edge(start, goal):
            return [start.copy(), goal.copy()]
        rng = np.random.default_rng(seed)
        nodes, parents = [start.copy()], [-1]
        deadline = time.monotonic()+timeout
        for _ in range(2500):
            if time.monotonic() >= deadline:
                break
            sample = goal if rng.random() < .25 else rng.uniform(self.lower, self.upper)
            nearest = int(np.argmin(np.linalg.norm(np.asarray(nodes)-sample, axis=1)))
            diff = sample-nodes[nearest]
            new = nodes[nearest]+diff*min(1., .25/max(float(np.linalg.norm(diff)), 1e-9))
            if not self.edge(nodes[nearest], new):
                continue
            nodes.append(new); parents.append(nearest)
            if not self.edge(new, goal):
                continue
            path = [goal.copy()]
            index = len(nodes)-1
            while index >= 0:
                path.append(nodes[index]); index = parents[index]
            path.reverse()
            # Deterministic shortcutting, each shortcut is checked as well.
            output = [path[0]]
            i = 0
            while i < len(path)-1:
                j = len(path)-1
                while j > i+1 and not self.edge(path[i], path[j]):
                    j -= 1
                output.append(path[j]); i = j
            return output
        raise PlanningError('no collision-free path within planning budget')
