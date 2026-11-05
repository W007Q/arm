"""Scene data adapters for the same collision planner.

Hardware uses measured RGB-D objects and a calibrated static fixture model.
No hardware path subscribes to simulator ground truth.
"""
import math
import time
import numpy as np
import mujoco
from .collision_planner import PlanningError

COLOR_BODIES = {'red':'red_cube', 'blue':'blue_block', 'yellow':'yellow_cylinder'}

class HardwareScene:
    def __init__(self, planner, table_top_m, ttl=30., camera_collision=True):
        self.planner, self.table_top, self.ttl = planner, table_top_m, ttl
        if not math.isfinite(table_top_m) or not .001 < ttl < 120:
            raise ValueError('invalid hardware scene table height or freshness limit')
        table = planner.model.body('task_table')
        geom = planner.model.geom('task_table_geom')
        planner.model.body_pos[table.id, 2] = table_top_m - float(geom.size[2])
        # 没安装相机时不加入外壳；安装后必须使用标定后的碰撞代理。
        camera = planner.model.geom('d435_body_visual')
        mask = int(camera_collision)
        planner.model.geom_contype[camera.id] = mask
        planner.model.geom_conaffinity[camera.id] = mask
        # MuJoCo 的粗筛缓存编译时的 body 掩码，改 geom 后也要同步它。
        body = int(planner.model.geom_bodyid[camera.id])
        planner.model.body_contype[body] = mask
        planner.model.body_conaffinity[body] = mask
        self.objects = {}
        self.received = 0.
        self.attached = None
        # Hide unused demonstration objects until measured by the real camera.
        for name in COLOR_BODIES.values():
            body=planner.model.body(name)
            adr=int(planner.model.jnt_qposadr[int(body.jntadr[0])])
            planner.data.qpos[adr:adr+3]=[0.,0.,-10.]

    def update(self, payload, now=None):
        now=time.monotonic() if now is None else now
        detections=payload.get('detections', [])
        if not isinstance(detections, list):
            return
        for item in detections:
            if not isinstance(item,dict):continue
            color=item.get('color')
            if color not in COLOR_BODIES or str(item.get('source','')).startswith('mujoco'):
                continue
            values=[item.get(k, math.nan) for k in ('x','y','z','shortest_m','longest_m')]
            if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
                continue
            x,y,top,short,long=values
            # Attached-object height is already known; top rising is not growth.
            if self.attached is not None and self.attached[0] == self.planner.model.body(COLOR_BODIES[color]).id:
                self.received=now
                continue
            height=top-self.table_top
            if not .005 < height < .3 or not .005 < short <= long < .3:
                continue
            body=self.planner.model.body(COLOR_BODIES[color])
            geom=self.planner.model.geom(int(body.geomadr[0]))
            # Axis-aligned conservative box covers visible outline. Unknown
            # hidden geometry needs a point-cloud scene, not this colour map.
            if int(geom.type[0]) == int(mujoco.mjtGeom.mjGEOM_BOX):
                self.planner.model.geom_size[geom.id]=[long/2, long/2, height/2]
            elif int(geom.type[0]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                self.planner.model.geom_size[geom.id,:2]=[long/2,height/2]
            self._refresh_bounds(body.id, geom.id)
            adr=int(self.planner.model.jnt_qposadr[int(body.jntadr[0])])
            self.objects[color]=(adr,np.array([x,y,self.table_top+height/2,1.,0.,0.,0.]),now)
            self.received=now

    def _refresh_bounds(self, body_id, geom_id):
        m = self.planner.model
        if m.body_geomnum[body_id] != 1 or m.body_bvhnum[body_id] != 1 or not np.allclose(m.geom_pos[geom_id], 0.) or not np.allclose(m.geom_quat[geom_id], [1.,0.,0.,0.]):
            raise PlanningError('hardware object proxy must be a single centered primitive')
        size = m.geom_size[geom_id]
        if m.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
            half = size.copy()
            radius = np.linalg.norm(half)
        elif m.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            half = np.array([size[0], size[0], size[1]])
            radius = np.hypot(size[0], size[1])
        else:
            raise PlanningError('unsupported hardware object proxy')
        # mj_setConst does not rebuild compiled geometry AABBs/radii. Update
        # the single-primitive leaf used by collision broadphase explicitly.
        m.geom_rbound[geom_id] = radius
        m.geom_aabb[geom_id, :3] = 0.
        m.geom_aabb[geom_id, 3:] = half
        m.bvh_aabb[int(m.body_bvhadr[body_id])] = m.geom_aabb[geom_id]

    def attach(self, color, q):
        if color == 'release':
            if self.attached:
                # Keep last predicted held position until the camera sees the release.
                obj,adr,_,_=self.attached
                item_color=next(k for k,v in COLOR_BODIES.items() if self.planner.model.body(v).id==obj)
                self.objects[item_color]=(adr,self.planner.data.qpos[adr:adr+7].copy(),time.monotonic())
            self.attached=None
            return
        if color not in self.objects:
            raise PlanningError('cannot attach object without measured scene pose')
        adr,pose,stamp=self.objects[color]
        if time.monotonic()-stamp > self.ttl:
            raise PlanningError('measured grasp object is stale')
        p=self.planner;p.data.qpos[p.addr]=q;p.data.qpos[adr:adr+7]=pose
        mujoco.mj_forward(p.model,p.data)
        obj=p.model.body(COLOR_BODIES[color]).id
        hand_r=p.data.xmat[p.hand].reshape(3,3)
        self.attached=(obj,adr,hand_r.T@(p.data.xpos[obj]-p.data.xpos[p.hand]),hand_r.T@p.data.xmat[obj].reshape(3,3))

    def prepare(self, selected, q, finger_positions, now=None):
        now=time.monotonic() if now is None else now
        if (selected in COLOR_BODIES and selected not in self.objects) or any(now-stamp>self.ttl for adr,_,stamp in self.objects.values() if self.attached is None or adr!=self.attached[1]):
            raise PlanningError('hardware colour collision map is missing or stale; obtain a fresh view')
        p=self.planner
        p.allowed_object=p.model.body(COLOR_BODIES[selected]).id if selected in self.objects else None
        for color,(adr,pose,_) in self.objects.items():
            if self.attached is None or adr != self.attached[1]:
                p.data.qpos[adr:adr+7]=pose
        for name,value in finger_positions.items():
            p.data.qpos[int(p.model.joint(name).qposadr[0])]=value
        p.data.qpos[p.addr]=q
        p.carried=self.attached
