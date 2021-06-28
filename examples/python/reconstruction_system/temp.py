import pyrealsense2 as rs
import numpy as np

x = [
		152.23756409,
		0.0,
		0.0,
		0.0,
		152.2297821,
		0.0,
		79.4900589,
		62.52014542,
		1.0
	]

x = 4 * np.asarray(x) / 5

print(x)
exit()

def initialize_camera():
    # start the frames pipe
    p = rs.pipeline()
    conf = rs.config()
    conf.enable_stream(rs.stream.accel)
    conf.enable_stream(rs.stream.gyro)
    prof = p.start(conf)
    return p


def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])


def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

p = initialize_camera()
from time import sleep
try:
    while True:
        f = p.wait_for_frames()
        accel = accel_data(f[0].as_motion_frame().get_motion_data())
        gyro = gyro_data(f[1].as_motion_frame().get_motion_data())
        print("accelerometer: ", accel)
        print("gyro: ", gyro)
        sleep(1)
finally:
    p.stop()