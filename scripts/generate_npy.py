import mujoco
import numpy as np

MODEL_PATH = "/home/shourya/rl/urdf/arduinobot.xml"
N_SAMPLES = 20000

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

ee_site_id = model.site("ee_site").id

joint_names = ["joint_1", "joint_2", "joint_3"]
joint_ids = [model.joint(name).id for name in joint_names]
qpos_addrs = [model.jnt_qposadr[jid] for jid in joint_ids]

ranges = []
for jid in joint_ids:
    jnt_range = model.jnt_range[jid]
    ranges.append((jnt_range[0], jnt_range[1]))

ee_positions = np.zeros((N_SAMPLES, 3))

for i in range(N_SAMPLES):
    for addr, (lo, hi) in zip(qpos_addrs, ranges):
        data.qpos[addr] = np.random.uniform(lo, hi)
    mujoco.mj_forward(model, data)
    ee_positions[i] = data.site_xpos[ee_site_id]

mins = ee_positions.min(axis=0)
maxs = ee_positions.max(axis=0)

print(f"Bounding box min: {mins}")
print(f"Bounding box max: {maxs}")

np.save("/home/shourya/rl/scripts/ee_reachable_points.npy", ee_positions)
