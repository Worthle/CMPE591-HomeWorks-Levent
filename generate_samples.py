import numpy as np
from environmentcnmp import BaseEnv

def generate_bezier_points(p0, p1, p2, p3, n_points=100):
    t_values = np.linspace(0, 1, n_points)
    curve = []
    for t in t_values:
        point = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
        curve.append(point)
    return np.array(curve)

def collect_demonstration(env, h, n_points=100):
    base_z = 1.0
    p0 = np.array([0.7, 0.3, base_z])
    p3 = np.array([0.7, -0.3, base_z])
    hit = np.random.rand() < 0.5
    if hit:
        z1 = np.random.uniform(base_z, h - 0.05)
        z2 = np.random.uniform(base_z, h - 0.05)
    else:
        z1 = np.random.uniform(h + 0.05, h + 0.3)
        z2 = np.random.uniform(h + 0.05, h + 0.3)
    p1 = np.array([0.7, 0.15, z1])
    p2 = np.array([0.7, -0.15, z2])
    curve = generate_bezier_points(p0, p1, p2, p3, n_points)

    t_data = np.linspace(0, 1, n_points)
    e_y, e_z = [], []

    env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180], n_splits=50)

    for point in curve:
        env._set_ee_pose(point, rotation=[-90, 0, 180], max_iters=10)
        ee_pos, _ = env._get_ee_pose()
        e_y.append(ee_pos[1])
        e_z.append(ee_pos[2])

    e_y = np.array(e_y)
    e_z = np.array(e_z)
    o_y = np.zeros(n_points)
    o_z = np.full(n_points, h)
    X = np.stack([t_data, np.full(n_points, h)], axis=1)
    Y = np.stack([e_y, e_z, o_y, o_z], axis=1)
    return X, Y

if __name__ == "__main__":
    env = BaseEnv(render_mode="offscreen")
    n_train, n_val, n_points = 200, 50, 100
    train_X, train_Y = [], []
    val_X, val_Y = [], []

    for _ in range(n_train):
        env.reset()
        h = np.random.uniform(1.1, 1.3)
        X, Y = collect_demonstration(env, h, n_points)
        train_X.append(X)
        train_Y.append(Y)
        print(f"Collected {len(train_X)} training trajectories.", end="\r")

    for _ in range(n_val):
        env.reset()
        h = np.random.uniform(1.1, 1.3)
        X, Y = collect_demonstration(env, h, n_points)
        val_X.append(X)
        val_Y.append(Y)
        print(f"Collected {len(val_X)} validation trajectories.", end="\r")
    np.save("training_X.npy", np.array(train_X))
    np.save("training_Y.npy", np.array(train_Y))
    np.save("validation_X.npy", np.array(val_X))
    np.save("validation_Y.npy", np.array(val_Y))
    print("Data generation complete.")
