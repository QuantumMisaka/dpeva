import numpy as np
import os
import shutil

# 配置
BASE_DIR = "test_data_gen"
MIXED_DIR = os.path.join(BASE_DIR, "mixed_data")
NPY_DIR = os.path.join(BASE_DIR, "npy_data")
DESC_MIXED_DIR = os.path.join(BASE_DIR, "desc_mixed")
DESC_NPY_DIR = os.path.join(BASE_DIR, "desc_npy")

# 清理并重建目录
if os.path.exists(BASE_DIR):
    shutil.rmtree(BASE_DIR)
for d in [MIXED_DIR, NPY_DIR, DESC_MIXED_DIR, DESC_NPY_DIR]:
    os.makedirs(d)

# 模拟参数
N_FRAMES = 5
N_ATOMS = 3
N_FEAT = 4  # 描述符特征维度

# 1. 生成 Mixed 数据 (包含 real_atom_types.npy)
# Mixed 格式通常把不同原子数的帧分开，或者全部视为同一类型但用 real_atom_types 区分
# 这里简化：假设所有帧原子数相同，但原子类型是动态的
mixed_folder = os.path.join(MIXED_DIR, "system_0")
os.makedirs(os.path.join(mixed_folder, "set.000"))

# 造数据：坐标、类型、晶胞
coords = np.random.rand(N_FRAMES, N_ATOMS, 3).astype(np.float32)
box = np.eye(3).flatten().astype(np.float32)
box = np.tile(box, (N_FRAMES, 1))
# Mixed 特有：real_atom_types.npy (N_FRAMES, N_ATOMS)
# 假设原子类型在帧间变化（虽然物理上少见，但格式支持）
real_types = np.random.randint(0, 2, size=(N_FRAMES, N_ATOMS)).astype(np.int32)
# type.raw 只是占位符，通常是全 0
type_raw = np.zeros(N_ATOMS, dtype=np.int32)

# 写入 Mixed 文件
np.save(os.path.join(mixed_folder, "set.000", "coord.npy"), coords)
np.save(os.path.join(mixed_folder, "set.000", "box.npy"), box)
np.save(os.path.join(mixed_folder, "set.000", "energy.npy"), np.zeros(N_FRAMES)) # dpdata 需要 energy 或 force
np.save(os.path.join(mixed_folder, "set.000", "real_atom_types.npy"), real_types)
with open(os.path.join(mixed_folder, "type.raw"), "w") as f:
    f.write("\n".join(map(str, type_raw)))
with open(os.path.join(mixed_folder, "type_map.raw"), "w") as f:
    f.write("H\nO\n") # 假装是 H 和 O

# 生成 Mixed 对应的描述符 (N_FRAMES, N_ATOMS, N_FEAT)
# 为了验证，我们手动构造描述符：基于原子类型和坐标的简单函数
desc_mixed = np.zeros((N_FRAMES, N_ATOMS, N_FEAT))
for i in range(N_FRAMES):
    for j in range(N_ATOMS):
        # 描述符 = 坐标 * (类型+1)
        desc_mixed[i, j, :3] = coords[i, j] * (real_types[i, j] + 1)
        desc_mixed[i, j, 3] = real_types[i, j]

np.save(os.path.join(DESC_MIXED_DIR, "system_0.npy"), desc_mixed)


# 2. 生成 NPY 数据 (标准格式，原子顺序可能打乱)
# 我们构建一个 NPY 系统，其每一帧对应 Mixed 的一帧，但原子顺序打乱
npy_folder = os.path.join(NPY_DIR, "system_0")
os.makedirs(os.path.join(npy_folder, "set.000"))
os.makedirs(os.path.join(npy_folder, "set.001"))

npy_coords = np.zeros_like(coords)
npy_types = np.zeros_like(real_types) 

# 但为了配合 verify 脚本的逻辑（它假设 NPY 的 type 是静态的 ls['atom_types']），
real_types[:] = np.array([0, 1, 0]) # 固定为 H O H
# 更新 Mixed 的文件
np.save(os.path.join(mixed_folder, "set.000", "real_atom_types.npy"), real_types)
# 更新描述符
for i in range(N_FRAMES):
    for j in range(N_ATOMS):
        desc_mixed[i, j, :3] = coords[i, j] * (real_types[i, j] + 1)
        desc_mixed[i, j, 3] = real_types[i, j]
np.save(os.path.join(DESC_MIXED_DIR, "system_0.npy"), desc_mixed)


# 构造 NPY 数据：打乱原子顺序 (0, 1, 2) -> (1, 2, 0)
perm = [1, 2, 0] # O H H
npy_types_static = real_types[0][perm] # 1 0 0
npy_coords = coords[:, perm, :] # 坐标也相应打乱

# NPY split into 2 sets: set.000 (3 frames), set.001 (2 frames)
# Set 000
np.save(os.path.join(npy_folder, "set.000", "coord.npy"), npy_coords[:3])
np.save(os.path.join(npy_folder, "set.000", "box.npy"), box[:3])
np.save(os.path.join(npy_folder, "set.000", "energy.npy"), np.zeros(3))

# Set 001
np.save(os.path.join(npy_folder, "set.001", "coord.npy"), npy_coords[3:])
np.save(os.path.join(npy_folder, "set.001", "box.npy"), box[3:])
np.save(os.path.join(npy_folder, "set.001", "energy.npy"), np.zeros(2))

with open(os.path.join(npy_folder, "type.raw"), "w") as f:
    f.write("\n".join(map(str, npy_types_static)))
with open(os.path.join(npy_folder, "type_map.raw"), "w") as f:
    f.write("H\nO\n")

# 生成 NPY 对应的描述符 (也需要打乱)
# desc_npy should match the concatenated order
desc_npy = desc_mixed[:, perm, :]
np.save(os.path.join(DESC_NPY_DIR, "system_0.npy"), desc_npy)

# 3. 构造 Mixed 多 set 场景 (验证 real_atom_types 加载)
# Mixed system 1
mixed_folder_1 = os.path.join(MIXED_DIR, "system_1")
os.makedirs(os.path.join(mixed_folder_1, "set.000"))
os.makedirs(os.path.join(mixed_folder_1, "set.001"))

# 假设 system_1 和 system_0 一样，只是分了 set
# real_types 也需要分 set 存储吗？
# dpdata 对于 Mixed 格式，如果 load 整个 system，它会把所有 set 的 coord 拼起来。
# 关键是 real_atom_types 在哪里？
# 通常 Mixed 格式每个 set 都有自己的 real_atom_types.npy。
# 如果 verify 脚本只读 set.000 的 real_atom_types，那就会出错。

# Set 000 (3 frames)
np.save(os.path.join(mixed_folder_1, "set.000", "coord.npy"), coords[:3])
np.save(os.path.join(mixed_folder_1, "set.000", "box.npy"), box[:3])
np.save(os.path.join(mixed_folder_1, "set.000", "energy.npy"), np.zeros(3))
np.save(os.path.join(mixed_folder_1, "set.000", "real_atom_types.npy"), real_types[:3])

# Set 001 (2 frames)
np.save(os.path.join(mixed_folder_1, "set.001", "coord.npy"), coords[3:])
np.save(os.path.join(mixed_folder_1, "set.001", "box.npy"), box[3:])
np.save(os.path.join(mixed_folder_1, "set.001", "energy.npy"), np.zeros(2))
np.save(os.path.join(mixed_folder_1, "set.001", "real_atom_types.npy"), real_types[3:])

with open(os.path.join(mixed_folder_1, "type.raw"), "w") as f:
    f.write("\n".join(map(str, type_raw)))
with open(os.path.join(mixed_folder_1, "type_map.raw"), "w") as f:
    f.write("H\nO\n")

# Desc for system_1 (should match system_0)
np.save(os.path.join(DESC_MIXED_DIR, "system_1.npy"), desc_mixed)

print("Test data generation complete.")
