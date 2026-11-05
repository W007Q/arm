from rebotarm_tasks.gripper import wait_gripper, MujocoGripperAdapter, Pa43GripperAdapter
from rebotarm_tasks.settings import PickSettings

class Gripper:
    def __init__(self,start,stop,backend='mujoco'):
        self.wall=0.;self.position=start;self.stop=stop
        self.settings=PickSettings(backend=backend,hardware_contact_torque_nm=.2,hardware_torque_limit_nm=.6)
        self.stale=False;self.force=1.2
    def snapshot(self):
        p=self.position
        state=MujocoGripperAdapter(self.settings).convert(
            {'gripper_joint1':-p,'gripper_joint2':p},{},
            {'gripper_joint1':self.force if abs(p-self.stop)<1e-7 else .1,'gripper_joint2':.1})
        return state.payload(),1. if self.stale else 0.
    def sleep(self,dt):
        self.wall+=dt;sign=1. if self.stop>self.position else -1.
        self.position+=sign*min(abs(self.stop-self.position),.002*dt)
    def wait(self,cmd,contact=False):
        return wait_gripper(self.snapshot,cmd,self.settings,allow_contact=contact,timeout=40.,clock=lambda:self.wall,sleep=self.sleep)

def test_slow_open_waits_for_measured_target():
    g=Gripper(0.,.046);r=g.wait(90.)
    assert r['ok'] and g.wall>20. and g.position>.0445

def test_slow_close_waits_for_contact():
    g=Gripper(.046,.024);r=g.wait(36.,True)
    assert r['ok'] and r['contact_candidate'] and g.wall>11.

def test_missing_feedback_is_failure():
    g=Gripper(.046,.024);g.stale=True
    assert not g.wait(36.,True)['ok']

def test_stable_preexisting_contact_is_accepted():
    g=Gripper(.024,.024);r=g.wait(15.,True)
    assert r['ok'] and r['contact_or_target']=='contact'

def test_empty_close_is_not_grasp_even_with_endstop_force():
    g=Gripper(0.,0.);g.force=9.
    assert not g.wait(0.,True)['ok']

def test_motor_torque_is_not_slide_force():
    s=PickSettings(backend='sdk')
    sim=MujocoGripperAdapter(s).convert({'gripper_joint1':-.024,'gripper_joint2':.024},{},{'gripper_joint1':2.,'gripper_joint2':2.})
    hw=Pa43GripperAdapter(s).convert(.024/.046*5.,0.,.3)
    assert sim.measured_gap_m==hw.measured_gap_m
    assert sim.effort_unit=='N' and hw.effort_unit=='Nm'

def test_hardware_requires_nonzero_calibrated_contact_threshold():
    s=PickSettings(backend='sdk');now=[0.]
    state=Pa43GripperAdapter(s).convert(2.,0.,9.).payload()
    r=wait_gripper(lambda:(state,0.),0.,s,allow_contact=True,timeout=.5,clock=lambda:now[0],sleep=lambda dt:now.__setitem__(0,now[0]+dt))
    assert not r['ok']

def test_wrong_unit_cannot_prove_contact():
    s=PickSettings(backend='sdk',hardware_contact_torque_nm=.2)
    state=Pa43GripperAdapter(s).convert(2.,0.,9.).payload();state['effort_unit']='N';now=[0.]
    r=wait_gripper(lambda:(state,0.),0.,s,allow_contact=True,timeout=.5,clock=lambda:now[0],sleep=lambda dt:now.__setitem__(0,now[0]+dt))
    assert not r['ok']
