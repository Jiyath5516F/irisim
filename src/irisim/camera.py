"""Camera, celestial and shadow matrices exactly like Minecraft + Iris compute them.

Matrices are numpy float64 4x4 in math convention (M @ v), uploaded transposed
to GL (column-major)."""
import math

import numpy as np


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], np.float64)


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], np.float64)


def rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)


def translate(v):
    m = np.eye(4)
    m[:3, 3] = v
    return m


def perspective(fovy_deg, aspect, near, far):
    """JOML Matrix4f.perspective with zZeroToOne=false."""
    h = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4))
    m[0, 0] = h / aspect
    m[1, 1] = h
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (far + far) * near / (near - far)
    m[3, 2] = -1.0
    return m


def ortho_symmetric(width, height, near, far):
    """JOML Matrix4f.setOrthoSymmetric with zZeroToOne=false."""
    m = np.eye(4)
    m[0, 0] = 2.0 / width
    m[1, 1] = 2.0 / height
    m[2, 2] = 2.0 / (near - far)
    m[2, 3] = (far + near) / (near - far)
    return m


def view_rotation(yaw, pitch):
    """Minecraft camera: rotationXYZ(pitch, yaw + 180, 0). yaw 0 looks south (+z), pitch > 0 looks down."""
    return rot_x(pitch) @ rot_y(yaw + 180.0)


def celestial_angle(world_time):
    """Vanilla DimensionType.timeOfDay (sky angle, 0..1, 0 = noon)."""
    d = ((world_time / 24000.0) - 0.25) % 1.0
    e = 0.5 - math.cos(d * math.pi) / 2.0
    return (d * 2.0 + e) / 3.0


def sun_angle_from_sky(sky):
    """Iris sunAngle: (skyAngle*360 + 90) wrapped to [0,360) / 360."""
    c = sky * 360.0 + 90.0
    if c < 0:
        c += 360
    elif c > 360:
        c -= 360
    return c / 360.0


class Celestial:
    def __init__(self, world_time, sun_path_rotation, dimension="overworld", fixed_time=None):
        if fixed_time is not None:
            world_time = fixed_time
        self.world_time = world_time
        self.sky = celestial_angle(world_time)
        self.sun_angle = sun_angle_from_sky(self.sky)
        self.is_day = self.sun_angle < 0.5
        moon_sky = (self.sky + 0.5) % 1.0
        self.shadow_angle = self.sun_angle if self.is_day else sun_angle_from_sky(moon_sky)
        self.spr = sun_path_rotation

    def _celestial(self, mv, angle_frac, y=100.0):
        m = mv @ rot_y(-90.0) @ rot_z(self.spr) @ rot_x(angle_frac * 360.0)
        return (m @ np.array([0.0, y, 0.0, 0.0]))[:3]

    def positions(self, mv):
        sun = self._celestial(mv, self.sky)
        moon = self._celestial(mv, (self.sky + 0.5) % 1.0)   # == (0,-100,0) rotated by the sun angle
        up = (mv @ rot_y(-90.0) @ np.array([0.0, 100.0, 0.0, 0.0]))[:3]
        shadow = sun if self.is_day else moon
        return sun, moon, shadow, up

    def world_dir(self, which="shadow"):
        sun, moon, shadow, _ = self.positions(np.eye(4))
        v = {"sun": sun, "moon": moon, "shadow": shadow}[which]
        return v / np.linalg.norm(v)

    def shadow_model_view(self, camera_pos, interval):
        """Iris ShadowMatrices.createModelViewMatrix."""
        sa = self.shadow_angle
        sky = sa + 0.75 if sa < 0.25 else sa - 0.25
        m = rot_x(90.0) @ rot_z(sky * -360.0) @ rot_x(self.spr)
        if abs(interval) > 0.0:
            off = np.array([math.fmod(float(c), interval) for c in camera_pos], np.float64)
            off -= interval / 2.0
            m = m @ translate(off)
        return m


def normal_matrix(mv):
    m3 = mv[:3, :3]
    try:
        return np.linalg.inv(m3).T
    except np.linalg.LinAlgError:
        return m3


# Iris BuiltinReplacementUniforms.iris_LightmapTextureMatrix: scale 1/256, translate 1/32 (x, y and z)
LIGHTMAP_TEXTURE_MATRIX = np.array([[1 / 256, 0, 0, 1 / 32], [0, 1 / 256, 0, 1 / 32], [0, 0, 1 / 256, 1 / 32],
                                    [0, 0, 0, 1]], np.float64)


class Camera:
    def __init__(self, pos, yaw, pitch, fov=70.0):
        self.pos = np.array(pos, np.float64)
        self.yaw = yaw
        self.pitch = pitch
        self.fov = fov

    def forward(self):
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        return np.array([-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p)])

    def view(self):
        return view_rotation(self.yaw, self.pitch)


def look_at_angles(eye, target):
    d = np.array(target, np.float64) - np.array(eye, np.float64)
    yaw = math.degrees(math.atan2(-d[0], d[2]))
    pitch = math.degrees(-math.atan2(d[1], math.hypot(d[0], d[2])))
    return yaw, pitch
