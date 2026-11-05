"""Image localization, capture latency, occlusion and no hidden state access."""
import cv2
import mujoco
import numpy as np
import pytest

from rebotarm_pingpong.vision import (
    CameraBallObserver, CameraCalibration, StereoBallTracker,
    detect_white_ball, triangulate,
)


def calibrated_pair():
    return [CameraCalibration.from_look_at(name, position, [.3, 0., .48])
            for name, position in (("first", [.65, -.75, .70]), ("second", [.65, .55, .70]))]


def ball_images(cameras, position, distractor=False):
    images = []
    for camera in cameras:
        image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
        center = camera.project(position)
        radius = .020 * camera.focal_px / np.linalg.norm(position - camera.position)
        cv2.circle(image, tuple(np.round(center).astype(int)), round(radius), (235, 235, 235), -1)
        if distractor:
            cv2.rectangle(image, (8, 10), (20, 75), (250, 250, 250), -1)
            cv2.circle(image, (40, 40), 8, (255, 10, 10), -1)
        images.append(image)
    return images


def test_rgb_detector_rejects_rectangle_colored_circle_and_occlusion():
    cameras = calibrated_pair()
    images = ball_images(cameras, [.3, 0., .55], distractor=True)
    for image in images:
        assert len(detect_white_ball(image)) == 1
    assert detect_white_ball(np.zeros_like(images[0])) == []
    tracker = StereoBallTracker(cameras, {"centroid_noise_std_px": 0.})
    position, _, detections = tracker.locate([images[0], np.zeros_like(images[1])])
    assert position is None
    assert detections == (None, None)


def test_calibrated_stereo_recovers_three_axes_and_rejects_parallel_rays():
    first, second = calibrated_pair()
    for position in ([.3, 0., .55], [.34, .04, .42], [.27, -.035, .66]):
        measured, gap = triangulate(first, second, first.project(position), second.project(position))
        np.testing.assert_allclose(measured, position, atol=1e-12)
        assert gap < 1e-12
    position, gap = triangulate(first, first, [160, 120], [160, 120])
    assert position is None and np.isinf(gap)


def test_image_estimator_ballistic_velocity_latency_and_stale_validity():
    cameras = calibrated_pair()
    settings = {"centroid_noise_std_px": 0., "position_alpha": 1., "velocity_alpha": 1., "max_age_s": .08}
    tracker = StereoBallTracker(cameras, settings)
    initial = np.array([.30, 0., .65])
    velocity = np.array([.08, -.04, -.1])
    dt = .06
    first = tracker.process(ball_images(cameras, initial), 0., .012)
    assert not first.valid  # velocity needs a second camera acquisition
    second_position = initial + velocity * dt + np.array([0., 0., -.5 * 9.81 * dt ** 2])
    estimate = tracker.process(ball_images(cameras, second_position), dt, dt + .012)
    expected_velocity = velocity + np.array([0., 0., -9.81 * (dt + .012)])
    assert estimate.valid
    assert estimate.age_s == pytest.approx(.012)
    assert estimate.latency_s == pytest.approx(.012)
    np.testing.assert_allclose(estimate.velocity, expected_velocity, atol=.09)
    stale = tracker.estimate(dt + .081)
    assert not stale.valid and stale.confidence == 0.
    assert np.isfinite(stale.position).all() and np.isfinite(stale.velocity).all()
    reacquired = tracker.process(ball_images(cameras, initial), .20, .212)
    assert not reacquired.valid and not reacquired.velocity_valid
    recovered = tracker.process(ball_images(cameras, initial), .22, .232)
    assert recovered.valid


def test_capture_clock_latency_and_dropout_without_ball_state_access():
    cameras = calibrated_pair()
    camera_xml = ''.join(f'<camera name="pingpong_{name}" pos="{position}" xyaxes="{axes}" fovy="48"/>'
                         for name, position, axes in (
                             ("tracking", ".65 -.75 .70", ".906183 .422885 0 -.108636 .232792 .966440"),
                             ("tracking_side", ".65 .55 .70", "-.843661 .536875 0 -.163752 -.257039 .95232")))
    model = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody>{camera_xml}</worldbody></mujoco>')

    class ClockOnly:
        time = 0.

        def __getattr__(self, name):
            raise AssertionError(f"observer read forbidden simulator data: {name}")

    class RGBFixture:
        def __init__(self, model, height, width):
            self.image_index = 0

        def update_scene(self, data, camera):
            self.image_index = int(camera.endswith("side"))
            self.capture_clock = data.time

        def render(self):
            render_clocks.append(self.capture_clock)
            return ball_images(observer.tracker.cameras, [.3, 0., .6])[self.image_index]

        def close(self):
            pass

    settings = {"fps": 50., "latency_s": .01, "latency_jitter_s": 0.,
                "dropout_probability": 0., "image_noise_std": 0.,
                "blur_sigma_px": 0., "centroid_noise_std_px": 0., "max_age_s": .10}
    render_clocks = []
    observer = CameraBallObserver(model, {}, settings, renderer_factory=RGBFixture)
    observer.reset(np.random.default_rng(4))
    clock = ClockOnly()
    assert not observer.observe(clock).valid
    assert observer.tracker.measurements == 0  # queued, never immediately visible
    clock.time = .01
    observer.observe(clock)
    assert observer.tracker.measurements == 1
    clock.time = .019
    observer.observe(clock)  # shutter subframe, not a completed sensor capture
    assert observer.frames_captured == 1
    clock.time = .02
    observer.observe(clock)
    clock.time = .03
    assert observer.observe(clock).valid
    assert observer.frames_captured == 2
    assert observer.estimate.measurement_timestamp_s == pytest.approx(.0195)
    assert render_clocks == [0., 0., .019, .019, .02, .02]
    observer.settings["dropout_probability"] = 1.
    clock.time = .16
    assert not observer.observe(clock).valid
    assert observer.frames_missed == 1
    observer.settings.update(calibration_position_std_m=.0005, calibration_rotation_std_deg=.05,
                             calibration_focal_relative_std=.001)
    observer.reset(np.random.default_rng(8))
    first_calibration = observer.tracker.cameras[0]
    assert np.linalg.norm(first_calibration.position-model.cam_pos[0]) > 0
    observer.reset(np.random.default_rng(8))
    np.testing.assert_array_equal(first_calibration.position, observer.tracker.cameras[0].position)
    np.testing.assert_array_equal(first_calibration.rotation, observer.tracker.cameras[0].rotation)
    assert first_calibration.focal_px == observer.tracker.cameras[0].focal_px
    observer.close()


def test_rendered_sphere_localization_uses_rgb_and_reports_occlusion():
    # These truth positions are used only to build a test fixture and score it;
    # the observer receives the same RGB-only interface as physical cameras.
    cameras = calibrated_pair()
    camera_xml = ''
    for camera in cameras:
        position = ' '.join(map(str, camera.position))
        axes = ' '.join(map(str, np.r_[camera.rotation[:, 0], camera.rotation[:, 1]]))
        camera_xml += f'<camera name="{camera.name}" pos="{position}" xyaxes="{axes}" fovy="48"/>'
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><visual><global offwidth="320" offheight="240"/></visual>
        <worldbody><light pos=".3 -.4 1.8" diffuse=".8 .8 .8"/>{camera_xml}
        <geom type="sphere" pos=".32 .025 .56" size=".02" rgba=".98 .98 .98 1"/>
        </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 240, 320)
    try:
        images = []
        for camera in cameras:
            renderer.update_scene(data, camera=camera.name)
            images.append(renderer.render().copy())
        tracker = StereoBallTracker(cameras, {"centroid_noise_std_px": 0.})
        position, confidence, _ = tracker.locate(images)
        assert position is not None and confidence > .5
        np.testing.assert_allclose(position, [.32, .025, .56], atol=.004)
        # A fully blocked RGB view has no stereo measurement and no fallback.
        assert tracker.locate([images[0], np.zeros_like(images[1])])[0] is None
    finally:
        renderer.close()
