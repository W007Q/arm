import threading
from types import SimpleNamespace
import numpy as np
import pytest
from rebotarm_planning import task_server
from rebotarm_planning.collision_planner import PlanningError


class Plant:
    backend="mujoco"
    command_hz=60.
    max_joint_speed=1.
    def __init__(self,speed):
        self.speed=speed;self.wall=0.;self.q=np.zeros(6);self.target=self.q.copy()
        self._planner_lock=threading.RLock();self._state_lock=threading.RLock();self._cancel=threading.Event()
        self.planner=SimpleNamespace(plan=lambda a,b:[a,b],edge=lambda a,b:True,last_collision='test obstacle')
        self.max_command_lead=.08
        self.peak_lead=0.;self.cancel_at=None;self.jump_at=None
    def sleep(self,dt):
        self.q+=np.clip(self.target-self.q,-dt*self.speed,dt*self.speed);self.wall+=dt
        if self.cancel_at is not None and self.wall>=self.cancel_at:self._cancel.set()
        if self.jump_at is not None and self.wall>=self.jump_at:
            self.q[0]+=1.;self.jump_at=None
    def _measured(self):return self.q.copy()
    def _prepare_collision_scene(self):pass
    def _publish_targets(self,q):
        self.peak_lead=max(self.peak_lead,float(np.max(np.abs(q-self.q))))
        self.target=q.copy()
    _safe_duration=task_server.RsTaskServer._safe_duration


def run(monkeypatch,plant):
    monkeypatch.setattr(task_server,'time',SimpleNamespace(monotonic=lambda:plant.wall,sleep=plant.sleep))
    goal=np.array([.28,1.9,1.9,-1.56,0.,.28])
    result=task_server.RsTaskServer._run_path_impl(plant,[np.zeros(6),goal],[0.,2.])
    return result,goal


@pytest.mark.parametrize('speed',[1.,.2,.1])
def test_slow_feedback_stretches_path_without_false_tracking_abort(monkeypatch,speed):
    plant=Plant(speed);result,goal=run(monkeypatch,plant)
    assert result
    assert max(abs(plant.q-goal))<.025
    assert plant.peak_lead<=.080001
    assert plant.wall<40.


def test_configured_command_lead_can_break_static_friction_without_disabling_guards(monkeypatch):
    plant=Plant(0.2)
    plant.max_command_lead=.2
    result,goal=run(monkeypatch,plant)
    assert result
    assert plant.peak_lead<=.200001
    assert plant.peak_lead > .15


def test_slow_final_approach_waits_for_measurement_beyond_four_seconds(monkeypatch):
    plant = Plant(.005)
    monkeypatch.setattr(task_server, 'time', SimpleNamespace(monotonic=lambda: plant.wall,
                                                           sleep=plant.sleep))
    goal = np.zeros(6)
    goal[2] = .1
    assert task_server.RsTaskServer._run_path_impl(plant, [np.zeros(6), goal], [0., 1.])
    assert plant.wall > 10.
    assert np.max(np.abs(plant.q-goal)) < .025


@pytest.mark.parametrize('event', ['cancel', 'stale', 'collision'])
def test_final_approach_keeps_cancellation_feedback_and_collision_guards(monkeypatch, event):
    plant = Plant(.005)
    monkeypatch.setattr(task_server, 'time', SimpleNamespace(monotonic=lambda: plant.wall,
                                                           sleep=plant.sleep))
    goal = np.zeros(6)
    goal[2] = .05
    if event == 'cancel':
        plant.cancel_at = 2.
        assert not task_server.RsTaskServer._run_path_impl(plant, [np.zeros(6), goal], [0., 1.])
    else:
        if event == 'stale':
            def measured():
                if plant.wall >= 2.:
                    raise PlanningError('joint feedback is missing or stale')
                return plant.q.copy()
            plant._measured = measured
            message = 'joint feedback is missing or stale'
        else:
            plant.planner.edge = lambda a, b: plant.wall < 2.
            message = 'collision while settling at goal'
        with pytest.raises(PlanningError, match=message):
            task_server.RsTaskServer._run_path_impl(plant, [np.zeros(6), goal], [0., 1.])
    # 目标早已下发完毕，此时必须仍在等待实物，而不是提前报告成功。
    np.testing.assert_allclose(plant.target, goal)
    assert plant.q[2] < goal[2]-.025


def test_slow_dense_waypoints_do_not_reset_measured_progress(monkeypatch):
    plant = Plant(.003)
    monkeypatch.setattr(task_server, 'time', SimpleNamespace(monotonic=lambda: plant.wall,
                                                           sleep=plant.sleep))
    path = [np.array([q, 0., 0., 0., 0., 0.]) for q in (0., .04, .08, .12, .2)]
    assert task_server.RsTaskServer._run_path_impl(plant, path, [0., .5, 1., 1.5, 2.])
    assert np.max(np.abs(plant.q-path[-1])) < .025


@pytest.mark.parametrize('frozen_joint', [1, 4])
def test_other_moving_joints_cannot_hide_a_stalled_motor(monkeypatch, frozen_joint):
    plant = Plant(np.full(6, .1))
    plant.speed[frozen_joint] = 0.
    monkeypatch.setattr(task_server, 'time', SimpleNamespace(monotonic=lambda: plant.wall,
                                                           sleep=plant.sleep))
    goal = np.full(6, .5)
    with pytest.raises(PlanningError, match=rf'joint{frozen_joint+1} .*no measured progress'):
        task_server.RsTaskServer._run_path_impl(plant, [np.zeros(6), goal], [0., 4.])
    assert plant.q[0] > .05


def test_only_encoder_jitter_does_not_prevent_stall(monkeypatch):
    plant = Plant(0.)
    def jitter(dt):
        plant.wall += dt
        plant.q[:] = .0003 if int(plant.wall*60) % 2 else -.0003
    plant.sleep = jitter
    with pytest.raises(PlanningError, match='no measured progress'):
        run(monkeypatch, plant)


def test_finished_motion_keeps_final_target_but_failure_and_cancel_hold_measured(monkeypatch):
    for outcome in (True, False, 'error'):
        plant = Plant(.2)
        plant._motion_lock = threading.Lock()
        plant._current_q = np.full(6, .08)
        plant.q = plant._current_q.copy()
        final = np.full(6, .1)
        plant.get_logger = lambda: SimpleNamespace(info=lambda _: None, error=lambda _: None)
        plant._tracking_summary = lambda: ''
        def impl(*args, **kwargs):
            plant._publish_targets(final)
            if outcome == 'error':
                raise PlanningError('test failure')
            return outcome
        plant._run_path_impl = impl
        if outcome == 'error':
            with pytest.raises(PlanningError, match='test failure'):
                task_server.RsTaskServer._run_path(plant, [], [])
        else:
            assert task_server.RsTaskServer._run_path(plant, [], []) is outcome
        np.testing.assert_allclose(plant.target, final if outcome is True else plant.q)
        assert not plant._motion_lock.locked()


def test_stalled_plant_still_aborts(monkeypatch):
    with pytest.raises(PlanningError,match='stalled'):run(monkeypatch,Plant(0.))


def test_collision_guard_preserved(monkeypatch):
    plant=Plant(.2);plant.planner.edge=lambda a,b:False
    with pytest.raises(PlanningError,match='path blocked'):run(monkeypatch,plant)


def test_cancellation_preserved(monkeypatch):
    plant=Plant(.2);plant.cancel_at=1.
    assert run(monkeypatch,plant)[0] is False


def test_large_external_error_still_aborts(monkeypatch):
    plant=Plant(.2);plant.jump_at=1.
    with pytest.raises(PlanningError,match='0.35 rad'):run(monkeypatch,plant)


def test_moveit_receives_measured_trajectory_feedback(monkeypatch):
    from builtin_interfaces.msg import Time
    plant = Plant(.2)
    plant.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time))
    monkeypatch.setattr(task_server, 'time', SimpleNamespace(monotonic=lambda: plant.wall,
                                                           sleep=plant.sleep))
    received = []
    handle = SimpleNamespace(is_cancel_requested=False, publish_feedback=received.append)
    goal = np.zeros(6)
    goal[2] = .1
    assert task_server.RsTaskServer._run_path_impl(plant, [np.zeros(6), goal], [0., 1.],
                                                 goal_handle=handle)
    assert len(received) >= 5
    assert max(f.actual.positions[2] for f in received) > .05
    for feedback in received:
        np.testing.assert_allclose(feedback.error.positions,
                                   np.array(feedback.desired.positions)-feedback.actual.positions)
    elapsed = received[-1].actual.time_from_start
    assert elapsed.sec + elapsed.nanosec/1e9 > .5


def test_planning_heartbeat_preserves_checked_command_instead_of_measured_lag():
    plant = Plant(.2)
    plant.backend = 'uart'
    plant._arm_enabled = True
    plant._motion_lock = threading.Lock()
    plant._command_q = np.full(6, .05)
    plant._motion_lock.acquire()
    task_server.RsTaskServer._hold_timer(plant)
    np.testing.assert_allclose(plant.target, .05)


def test_planning_heartbeat_still_stops_on_missing_feedback():
    plant = Plant(.2)
    plant.backend = 'uart'
    plant._arm_enabled = True
    plant._motion_lock = threading.Lock()
    plant._command_q = np.full(6, .05)
    plant._motion_lock.acquire()
    def stale():
        raise PlanningError('joint feedback is missing or stale')
    plant._measured = stale
    task_server.RsTaskServer._hold_timer(plant)
    np.testing.assert_allclose(plant.target, 0.)


def test_enabled_idle_hold_does_not_follow_sagging_feedback():
    plant = Plant(.2)
    plant.backend = 'uart'
    plant._arm_enabled = True
    plant._motion_lock = threading.Lock()  # 当前没有轨迹。
    plant._command_q = np.zeros(6)
    plant.q[3] = -.004
    task_server.RsTaskServer._hold_timer(plant)
    np.testing.assert_allclose(plant.target, 0.)


def test_enable_transition_discards_previous_hold_position():
    plant = Plant(.2)
    plant.backend = 'uart'
    plant._arm_enabled = False
    plant._motion_lock = threading.Lock()
    plant._command_q = np.full(6, .2)
    task_server.RsTaskServer._status_callback(plant, SimpleNamespace(enabled=True))
    assert plant._command_q is None
    task_server.RsTaskServer._hold_timer(plant)
    np.testing.assert_allclose(plant.target, plant.q)


def trajectory_receiver():
    node = Plant(.2)
    node.max_joint_speed = .2
    node.joints = [task_server.ArmJoint(f'joint{i+1}', i, i, -1., 1.) for i in range(6)]
    node.joints[1] = task_server.ArmJoint('joint2', 1, 1, -2.6179939, 0.)
    node._current_positions = lambda: node.q.copy()
    return node


def test_dense_moveit_trajectory_preserves_timing_and_speed_limit():
    node = trajectory_receiver()
    points = []
    for tick in range(16):
        point = task_server.JointTrajectoryPoint()
        point.positions = [.1 * tick / 15]
        point.time_from_start.sec = tick // 10
        point.time_from_start.nanosec = (tick % 10) * 100000000
        points.append(point)
    path, times = task_server.RsTaskServer._expand_trajectory(node, ['joint3'], points)
    assert times[-1] <= 1.6  # 1.5 s 的规划不能再变成逐点累积的 3.2 s。
    peak_speeds = [1.5 * max(abs(b-a)) / (t1-t0)
                   for a, b, t0, t1 in zip(path, path[1:], times, times[1:])]
    assert max(peak_speeds) <= node.max_joint_speed + 1e-9


def test_invalid_trajectory_identifies_joint_point_and_allowed_range():
    node = trajectory_receiver()
    point = task_server.JointTrajectoryPoint()
    point.positions = [.2]
    with pytest.raises(ValueError, match=r'point 0: joint2 position=0.2 rad, allowed=\[-2.6179939, 0\]'):
        task_server.RsTaskServer._expand_trajectory(node, ['joint2'], [point])
