# 数据目录（不入 git）

本目录只放**软链接**或本机挂载，不把 HDF5 / ckpt 推进仓库。

## 需要哪些示范

| 数据集 | 用途 | 约体积 |
|--------|------|--------|
| `sim_insertion_human` | 主任务（需含 `/labels`） | ~22GB |
| `sim_transfer_cube_human` | 第二任务 / 泛化 | ~18GB |
| scripted 变体（可选） | 后续标签模型伪标签 | ~18GB×2 |

Kin baseline 在本仓库内用 `third_party/kin_seg` **内存打标签**，不需要单独的 `*_kin` HDF5 副本，也不要往共享盘原始文件写 `/labels_kin`。

官方下载说明见 [DemoSpeedup](https://github.com/lingxiao-guo/DemoSpeedup)。

## 本机一键挂载

```bash
# 若本机已有 demospeedup 数据
bash scripts/link_data.sh /home/lee/projects/demospeedup/aloha/data

# 实验室共享盘
bash scripts/link_data.sh /data/share/aloha_demos
```

挂载后应看到：

```text
data/sim_insertion_human -> ...
data/sim_transfer_cube_human -> ...
data/.local_paths.env
```
