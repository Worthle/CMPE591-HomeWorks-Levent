import numpy as np
from homework1 import Hw1Env
import torch
import multiprocessing

for i in range(3):
#def collect_data(i):
    env = Hw1Env(render_mode="offscreen")
    if i == 0:
        N = 800
    elif i == 1:
        N = 200
    elif i == 2:
        N = 200
    #init_positions = torch.zeros(200, 2, dtype=torch.float)
    positions = torch.zeros(N, 2, dtype=torch.float)
    geoms = torch.zeros(N, 9, dtype=torch.float)
    actions = torch.zeros(N, dtype=torch.uint8)
    imgs = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    for j in range(N):
        action_id = np.random.randint(4)
        #obj_geom, obj_pos, pixels = env.state()
        #init_positions[j] = torch.tensor(obj_pos)
        env.step(action_id)
        obj_geom, obj_pos, pixels = env.state()
        positions[j] = torch.tensor(obj_pos)
        actions[j] = action_id
        imgs[j] = pixels
        env.reset()
        print(f"working i:{i} j:{j}")
        #print(f"{obj_geom}")
    #torch.save(init_positions, f"datas/init_positions_{i}.pt")
    torch.save(positions, f"datas/positions_{i}.pt")
    torch.save(geoms, f"datas/geoms_{i}.pt")
    torch.save(actions, f"datas/actions_{i}.pt")
    torch.save(imgs, f"datas/imgs_{i}.pt")
    
#if __name__ == '__main__':
#    pool = multiprocessing.Pool(processes=5)
#
#    pool.map(collect_data, range(5))

#    pool.close()
#    pool.join()
