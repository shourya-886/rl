"""
Empirically measure how many simulation steps it takes for the arm to
settle within the success threshold, for a batch of random reachable
targets. Use this to set a sensible max_episode_steps instead of guessing.

This drives each joint directly toward the exact angles that produced
the sampled target (since we generate target + solution together here),
giving a realistic "best case" settling time -- i.e. how fast a
well-behaved policy COULD settle, which is what you want as a ceiling
reference for max_episode_steps.

For any trial that never settles, prints diagnostics (target vs final
joint angles, actuator forces) so we can tell whether it's a control
problem (steady-state error / force saturation) or something else.
"""

import numpy as np
import mujoco

MODEL_PATH = "/home/shourya/rl/urdf/arduinobot.xml"
N_TRIALS = 30
MAX_STEPS_PER_TRIAL = 3000
SUCCESS_THRESHOLD = 0.08
VELOCITY_THRESHOLD = 0.05

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

ee_site_id = model.site("ee_site").id
arm_joint_names = ["joint_1", "joint_2", "joint_3"]
qpos_addrs = [model.jnt_qposadr[model.joint(n).id] for n in arm_joint_names]
actuator_ids = [model.actuator(f"servo_{n}").id for n in arm_joint_names]

rng = np.random.default_rng(0)


def sample_target_and_solution():
    ranges = [model.jnt_range[model.joint(n).id] for n in arm_joint_names]
    angles = np.array([rng.uniform(lo, hi) for (lo, hi) in ranges])
    for addr, ang in zip(qpos_addrs, angles):
        data.qpos[addr] = ang
    mujoco.mj_forward(model, data)
    ee_pos = data.site_xpos[ee_site_id].copy()
    return ee_pos, angles


def get_ee_speed():
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, ee_site_id, vel6, 0)
    return np.linalg.norm(vel6[3:6])


settle_steps_list = []
never_settled = 0

for trial in range(N_TRIALS):
    mujoco.mj_resetData(model, data)
    target_pos, target_angles = sample_target_and_solution()

    mujoco.mj_resetData(model, data)  # back to rest pose before driving toward target
    mujoco.mj_forward(model, data)

    settled_at = None
    distance = None
    speed = None

    for step in range(MAX_STEPS_PER_TRIAL):
        for aid, ang in zip(actuator_ids, target_angles):
            data.ctrl[aid] = ang

        mujoco.mj_step(model, data)

        ee_pos = data.site_xpos[ee_site_id]
        distance = np.linalg.norm(target_pos - ee_pos)
        speed = get_ee_speed()

        if distance < SUCCESS_THRESHOLD and speed < VELOCITY_THRESHOLD:
            settled_at = step + 1
            break

    if settled_at is None:
        never_settled += 1
        final_qpos = [data.qpos[addr] for addr in qpos_addrs]
        final_forces = [data.actuator_force[aid] for aid in actuator_ids]
        joint_errors = [t - q for t, q in zip(target_angles, final_qpos)]

        print(f"Trial {trial}: NEVER settled within {MAX_STEPS_PER_TRIAL} steps "
              f"(final distance={distance:.4f}, speed={speed:.4f})")
        print(f"  Target angles:   {target_angles}")
        print(f"  Final qpos:      {final_qpos}")
        print(f"  Joint errors:    {joint_errors}")
        print(f"  Actuator forces: {final_forces}")
    else:
        settle_steps_list.append(settled_at)
        print(f"Trial {trial}: settled in {settled_at} steps")

settle_steps_arr = np.array(settle_steps_list)

print("\n=== Summary ===")
print(f"Trials completed: {N_TRIALS}, never settled: {never_settled}")
if len(settle_steps_arr) > 0:
    print(f"Min steps to settle:    {settle_steps_arr.min()}")
    print(f"Median steps to settle: {int(np.median(settle_steps_arr))}")
    print(f"Mean steps to settle:   {settle_steps_arr.mean():.1f}")
    print(f"Max steps to settle:    {settle_steps_arr.max()}")
    print(f"95th percentile:        {int(np.percentile(settle_steps_arr, 95))}")
