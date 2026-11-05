"""按实测反馈限制轨迹领先量，并逐轴区分缓慢跟随与真正停滞。"""
import numpy as np


class SegmentProgress:
    """只推进当前段的指令；停滞检查跨段维护，避免每个小路点重新计时。"""

    def __init__(self, start, goal, duration, max_lead=.08):
        self.start = np.asarray(start, dtype=float).copy()
        self.goal = np.asarray(goal, dtype=float).copy()
        self.duration = max(float(duration), 1e-6)
        self.max_lead = float(max_lead)
        if not np.isfinite(self.max_lead) or self.max_lead <= 0:
            raise ValueError('max_lead must be positive')
        self.fraction = 0.

    def at(self, fraction):
        smooth = fraction*fraction*(3.-2.*fraction)
        return self.start+(self.goal-self.start)*smooth

    def advance(self, measured, dt):
        measured = np.asarray(measured)
        proposed = min(1., self.fraction+max(dt, 0.)/self.duration)
        if np.max(np.abs(self.at(proposed)-measured)) <= self.max_lead:
            self.fraction = proposed
        elif np.max(np.abs(self.at(self.fraction)-measured)) <= self.max_lead:
            low, high = self.fraction, proposed
            for _ in range(20):
                mid = .5*(low+high)
                if np.max(np.abs(self.at(mid)-measured)) <= self.max_lead:
                    low = mid
                else:
                    high = mid
            self.fraction = low
        # 实物偏离上一目标时保持该目标；碰撞和大跟踪误差仍由执行器检查。
        if self.fraction > 1.-1e-10:
            self.fraction = 1.
        return self.at(self.fraction)


class MotionWatchdog:
    """整条路径共用的逐轴进展检查，终点等待也使用同一规则。"""

    def __init__(self, measured, now, timeout=4., tolerance=.025):
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('motion stall timeout must be positive')
        self.timeout = float(timeout)
        self.tolerance = float(tolerance)
        self.anchor = np.asarray(measured, dtype=float).copy()
        self.last_progress = np.full(self.anchor.shape, now, dtype=float)
        self.direction = np.zeros_like(self.anchor)

    def check(self, measured, target, now):
        measured = np.asarray(measured)
        error = np.asarray(target)-measured
        active = np.abs(error) > self.tolerance
        direction = np.sign(error)
        # 到位或目标换向时开始新一轮判断。指令继续向前不等于实物前进，
        # 因而不能随每次发包、切路点或其他关节运动而刷新本轴计时。
        reset = ~active | (direction != self.direction)
        self.anchor[reset] = measured[reset]
        self.last_progress[reset] = now
        self.direction = np.where(active, direction, 0.)
        # 累计朝目标移动至少 0.002 rad 才算进展，避免编码器小抖动保活。
        advanced = active & ((measured-self.anchor)*direction >= .002)
        self.anchor[advanced] = measured[advanced]
        self.last_progress[advanced] = now
        stalled = np.flatnonzero(active & (now-self.last_progress > self.timeout))
        if stalled.size:
            i = int(stalled[0])
            raise RuntimeError(
                f'joint motion stalled: joint{i+1} measured={measured[i]:.5f}, '
                f'commanded={target[i]:.5f}; no measured progress toward target '
                f'for {self.timeout:g} s'
            )
