#!/usr/bin/env python
"""
阶段 5：一个脚本训练**所有**实验配置（PLAN §4.1 表 2 的配置组成 × 源码默认超参）。

  · **全局任务池**：所有 (配置, 种子) 扁平提交、固定 --jobs 个进程并发。
    早先版本是「逐配置串行、同配置内 5 种子并发」，但 5 个种子在 3 并发下会
    分成 3+2 两批，第二批留下并发空档；任务彼此完全独立，跨配置同时训练
    不影响正确性，故改为全局池。
  · 每个 (配置, 种子) 训完立即在 OOD 295 与 FEP 42 上推理并存预测值，
    阶段 6 只做集成与统计，不必再加载模型
  · 断点续跑：已存在 done.json 的任务直接跳过
  · 详细 tqdm：主进度条 + 每个并发槽位一条批次级进度条
    （postfix 显示 loss / val PCC / 8轮均值 / 最优轮 / 已用时）

━━━━━━━━━━━━━━ 环境实测（2026-08-19，决定 --workers / --jobs 默认值）━━━━━━━━━━━━━━
本机是容器，`free`/`nproc` 报的是**宿主机**数字（503 GB / 128 核），真实上限在 cgroup v2：
    /sys/fs/cgroup/memory.max = 62 GiB
    /sys/fs/cgroup/cpu.max    = 1200000 100000  →  12 核
    显存 RTX 4080 SUPER 32,760 MiB；ulimit -n 仅 1024

在最大配置（cfg08，148,125 条 / 训练 133,313 条）上实测单进程：

  --workers 3（旧默认）    主进程 + 6 个 DataLoader 子进程
      匿名内存 58 GiB（几乎顶满 62 GiB 上限）→ 5 并发时进程被 SIGKILL，
      报 BrokenProcessPool；且 6 分 52 秒连 1 个 epoch 都没跑完
  --workers 0（现默认）    单进程直接从 mmap 切片
      匿名内存 288 MiB、file(page cache) 8.3 GiB（可回收、多进程共享）
      epoch 1 = 45 s，epoch 2 = 41 s，显存峰值 8,110 MiB，GPU 利用率 34–50%
      2 epoch + 加载 + 两次评测推理合计 1.75 分钟

即 workers>0 在 12 核限制下**慢 5 倍以上且多耗 58 GiB 内存** —— PyG 的 Data 对象
逐 batch 序列化传递的 IPC 开销远超收益，而 workers=0 时是 mmap 零拷贝切片。
（已单独验证 np.load(mmap_mode='r') 工作正常：全量读后计入 cgroup file 而非 anon。）

由此定默认值：--workers 0、--jobs 3（8.1 GiB × 3 = 24.3 GiB 显存，留 8 GiB 余量；
匿名内存 3 × 2.5 GiB + 共享 page cache 8.5 GiB，远低于 62 GiB）。
推算：200 epoch ≈ 2.3 h/种子（最大配置），50 个任务 3 并发约 40 h。

━━━━━━━━━━━━━━━━━━ 超参数口径（重要）━━━━━━━━━━━━━━━━━━
用户指令：「模型超参数以模型源代码默认参数为准」，**覆盖 PLAN §5.2**。
下列常量逐项取自 AEV-PLIG/training.py：

                     PLAN §5.2（已废弃）       源码默认（现在用这个）
  batch_size              64                     128
  learning rate         5e-4       0.00012291937615434127
  weight_decay          1e-5                       0
  max epochs             200                     200
  梯度裁剪               1.0                      无
  lr 调度      ReduceLROnPlateau(10, 0.5)          无
  早停          val PCC patience=20         无（跑满 200 轮）
  检查点选择      单轮 val PCC 最优      **最近 8 轮 val PCC 均值**最优
  损失函数              MSE                     MSE（同）
  优化器               Adam                    Adam（同）

「检查点 = 最近 8 轮滑动均值最优」是参考实现 _train() 的做法：
  low = max(epoch-7, 0); avg_pc = mean(pcs[low:epoch+1]); if avg_pc > best_pc: save
判据是滑动均值、存的却是当轮权重，本脚本原样复现，不做"改进"。

保留 PLAN 的部分（这些不是"模型超参数"）：
  · 训练/验证 9:1 划分，固定 SPLIT_SEED=42（§5.4 可复现要求）
  · 集成成员数 = 5 个随机种子（PLAN §7；§5.3 写 10 个，用户已定为 5 个）
  · 评测集 = OOD 295 + FEP 42（参考实现用 PDBbind core set，与本实验无关）

标签标准化：与参考实现一致，用**训练集**的 mean/std（sklearn StandardScaler，
ddof=0）标准化 pK 后算 MSE，推理时逆变换回 pK。

用法：
  # 冒烟测试
  python 06_train_all.py --configs 1 --seeds 42 43 --max-epochs 3 --limit-train 2000
  # 正式训练（配置 1–9 × 种子 42–46 = 45 个任务，全局 3 并发）
  python 06_train_all.py
  # 想省时间可开早停（非源码行为，会写进 done.json 标记）
  python 06_train_all.py --early-patience 20
"""
import os, sys, json, time, argparse, traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_data import GraphStore, GraphDataset, fit_y_scaler, load_config, load_testset
from lib_model import build_model, n_params, DEFAULT_CONFIG

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = f'{ROOT}/experiments'
OUT = f'{E}/06_train'
LOGD = f'{ROOT}/logs/train'

# ── 源码默认超参（AEV-PLIG/training.py）──
LR = 0.00012291937615434127
WEIGHT_DECAY = 0
BATCH = 128
MAX_EPOCHS = 200
AVG_WINDOW = 8           # 检查点判据的滑动窗口（参考实现 max(epoch-7,0)）
# ── 保留自 PLAN ──
VAL_FRAC = 0.1
SPLIT_SEED = 42          # 所有种子共用同一划分，否则集成成员的验证集互相污染
SEEDS = list(range(42, 47))       # 5 个种子（PLAN §7）
TRAIN_SOURCES = ['pdbbind', 'bindingnet', 'bindingdb_dcs', 'hiqbind', 'gatoraffinity']
TESTSETS = ['ood295', 'fep42']

_SLOTS = None


def _init(lock, slots):
    tqdm.set_lock(lock)
    global _SLOTS
    _SLOTS = slots
    torch.set_num_threads(2)


def pcc_t(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if float(d) > 1e-12 else float('nan')


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    ys, ps, ix = [], [], []
    for b in loader:
        b = b.to(dev)
        ps.append(model(b).view(-1).float().cpu())      # 模型输出 (B,1) → 展平
        ys.append(b.y.view(-1).float().cpu())
        ix.append(b.idx.cpu())
    return (torch.cat(ys), torch.cat(ps), torch.cat(ix))


def train_one(job):
    cid, seed, args = job
    slot = _SLOTS.get()
    d = f'{OUT}/cfg{cid:02d}/seed{seed}'
    os.makedirs(d, exist_ok=True)
    os.makedirs(LOGD, exist_ok=True)
    lg = open(f'{LOGD}/cfg{cid:02d}_s{seed}.log', 'a')
    P = lambda *a: (print(*a, file=lg, flush=True))
    bar = None
    try:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        items = load_config(cid)
        if args.limit_train:
            items = items.sample(n=min(args.limit_train, len(items)),
                                 random_state=SPLIT_SEED).reset_index(drop=True)
        store = GraphStore(sorted(set(items.source)) + TESTSETS)

        # 9:1 划分（固定 SPLIT_SEED）
        rng = np.random.RandomState(SPLIT_SEED)
        perm = rng.permutation(len(items))
        nval = int(round(len(items) * VAL_FRAC))
        vi, ti = perm[:nval], perm[nval:]
        tr, va = items.iloc[ti].reset_index(drop=True), items.iloc[vi].reset_index(drop=True)
        ysc = fit_y_scaler(tr.y.values)
        P(f'配置 {cid} 种子 {seed}：训练 {len(tr)} / 验证 {len(va)}；'
          f'y 标准化 mean={ysc[0]:.4f} std={ysc[1]:.4f}')

        # drop_last=True：模型每层都有 BatchNorm，若最后一批只剩 1 个样本，
        # train 模式下 BatchNorm1d 会抛 "Expected more than 1 value per channel"。
        # 参考实现没 drop（其数据量恰好没踩到），这里必须防。每轮 shuffle，
        # 最多丢 127 条，且每轮丢的不同，对训练无实质影响。
        dl = DataLoader(GraphDataset(store, tr, ysc), batch_size=BATCH, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)
        vl = DataLoader(GraphDataset(store, va, ysc), batch_size=512, shuffle=False,
                        num_workers=args.workers, persistent_workers=args.workers > 0)

        model = build_model().to(dev)
        P(f'GATv2Net 5 层 / hidden {DEFAULT_CONFIG.hidden_dim} / head '
          f'{DEFAULT_CONFIG.head} / {DEFAULT_CONFIG.activation_function}；'
          f'参数量 {n_params(model):,}')
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        lossf = nn.MSELoss()

        max_ep = args.max_epochs
        bar = tqdm(total=len(dl), position=slot, leave=False, ncols=118,
                   desc=f'cfg{cid:02d} s{seed} ep 0/{max_ep}', mininterval=0.5)
        # 参考实现 _train：best_pc 初值 -1.1，判据为最近 8 轮 val PCC 均值
        best_avg, best_ep, best_vp, pcs, hist = -1.1, -1, np.nan, [], []
        for ep in range(1, max_ep + 1):
            model.train()
            bar.reset(total=len(dl))
            bar.set_description_str(f'cfg{cid:02d} s{seed} ep {ep}/{max_ep}')
            tot = nb = 0
            for b in dl:
                b = b.to(dev, non_blocking=True)
                opt.zero_grad()
                loss = lossf(model(b), b.y.view(-1, 1))
                loss.backward()
                opt.step()               # 源码无梯度裁剪
                tot += float(loss); nb += 1
                if nb % 20 == 0 or nb == len(dl):
                    bar.set_postfix_str(
                        f'loss {tot / nb:.4f} avg8 {best_avg:+.4f}@{best_ep} '
                        f'last {pcs[-1] if pcs else float("nan"):+.4f} '
                        f'{(time.time() - t0) / 60:.0f}min', refresh=False)
                bar.update(1)

            y, p, _ = predict(model, vl, dev)
            vp = pcc_t(y, p)
            vr = float(torch.sqrt(((y - p) ** 2).mean()) * ysc[1])
            pcs.append(vp)
            # low = max(epoch-7, 0); mean(pcs[low:epoch+1]) ≡ mean(pcs[-8:])
            avg = float(np.nanmean(pcs[-AVG_WINDOW:]))
            improved = avg > best_avg
            if improved:
                best_avg, best_ep, best_vp = avg, ep, vp
                torch.save({'model': model.state_dict(), 'y_scaler': ysc,
                            'epoch': ep, 'val_pcc': vp, 'avg8_val_pcc': avg,
                            'config': cid, 'seed': seed}, f'{d}/best.pt')
            hist.append({'epoch': ep, 'train_mse': tot / max(nb, 1), 'val_pcc': vp,
                         'avg8_val_pcc': avg, 'val_rmse_pK': vr, 'saved': improved})
            P(f'  ep {ep:3d} train_mse {tot / max(nb, 1):.4f} val_pcc {vp:.4f} '
              f'avg8 {avg:.4f} val_rmse {vr:.4f} '
              f'best_avg8 {best_avg:.4f}@{best_ep}{" *saved" if improved else ""}')
            # 源码默认无早停（--early-patience 0）；>0 时按 avg8 无提升计数
            if args.early_patience and ep - best_ep >= args.early_patience:
                P(f'  早停：avg8 连续 {args.early_patience} 轮无提升（非源码行为）')
                break
        bar.close(); bar = None
        pd.DataFrame(hist).to_csv(f'{d}/history.csv', index=False)

        # 用检查点在两个评测集上推理
        ck = torch.load(f'{d}/best.pt', map_location=dev)
        model.load_state_dict(ck['model'])
        preds = {}
        for ts in TESTSETS:
            df = load_testset(ts, store)
            tl = DataLoader(GraphDataset(store, df[['source', 'unique_id', 'y']], None),
                            batch_size=512, shuffle=False, num_workers=args.workers)
            _, p, ix = predict(model, tl, dev)
            o = np.full(len(df), np.nan)
            o[ix.numpy()] = p.numpy() * ysc[1] + ysc[0]      # 逆标准化回 pK
            np.save(f'{d}/pred_{ts}.npy', o)
            preds[ts] = {'n': int(len(o)), 'nan': int(np.isnan(o).sum())}

        json.dump({'config': cid, 'seed': seed,
                   'best_avg8_val_pcc': best_avg, 'val_pcc_at_best': best_vp,
                   'best_epoch': best_ep, 'epochs_run': len(hist),
                   'n_train': len(tr), 'n_val': len(va), 'y_scaler': ysc,
                   'hyper': {'lr': LR, 'batch': BATCH, 'weight_decay': WEIGHT_DECAY,
                             'max_epochs': max_ep, 'avg_window': AVG_WINDOW,
                             'early_patience': args.early_patience,
                             'source': 'AEV-PLIG/training.py defaults'},
                   'minutes': round((time.time() - t0) / 60, 2), 'preds': preds},
                  open(f'{d}/done.json', 'w'), indent=1)
        P(f'完成：best avg8 val PCC {best_avg:.4f} @ep{best_ep}'
          f'（该轮单点 {best_vp:.4f}），{(time.time() - t0) / 60:.1f} 分钟')
        return (cid, seed, best_avg, best_ep, None)
    except Exception as e:
        P('失败\n' + traceback.format_exc())
        return (cid, seed, np.nan, -1, f'{type(e).__name__}: {e}')
    finally:
        if bar is not None:
            bar.close()
        lg.close()
        _SLOTS.put(slot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='*', type=int,
                    default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--seeds', nargs='*', type=int, default=SEEDS)
    ap.add_argument('--jobs', type=int, default=3,
                    help='全局并发训练进程数（显存 8.1 GiB/进程，32 GiB 卡上 3 稳妥）')
    ap.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    ap.add_argument('--early-patience', type=int, default=0,
                    help='0 = 源码行为（跑满 max-epochs 不早停）；>0 按 avg8 无提升早停')
    ap.add_argument('--limit-train', type=int, default=0, help='冒烟测试用：抽样训练集')
    ap.add_argument('--workers', type=int, default=0,
                    help='每个训练进程的 DataLoader 子进程数。默认 0 —— 本机 12 核 '
                         'cgroup 下 workers=3 比 workers=0 慢 5 倍以上且多耗 58 GiB '
                         '匿名内存，详见文件头「环境实测」')
    ap.add_argument('--force', action='store_true', help='忽略 done.json 重跑')
    a = ap.parse_args()

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(OUT, exist_ok=True)

    jobs = []
    for cid in a.configs:
        for s in a.seeds:
            if not a.force and os.path.exists(f'{OUT}/cfg{cid:02d}/seed{s}/done.json'):
                continue
            jobs.append((cid, s, a))
    print(f'待训练任务 {len(jobs)} 个（配置 {a.configs} × 种子 {a.seeds}），'
          f'全局并发 {a.jobs}，DataLoader workers {a.workers}，'
          f'最多 {a.max_epochs} epoch，'
          f'早停 {"关（源码行为）" if not a.early_patience else a.early_patience}',
          flush=True)
    print(f'超参：lr {LR} batch {BATCH} weight_decay {WEIGHT_DECAY}；'
          f'检查点 = 最近 {AVG_WINDOW} 轮 val PCC 均值最优（源码口径）', flush=True)
    if not jobs:
        print('全部已完成。'); return 0

    # 用 ProcessPoolExecutor 而不是 mp.Pool：Pool 的 worker 是 daemon 进程，
    # 若 --workers > 0 就无法再 fork DataLoader 的子进程
    # （AssertionError: daemonic processes are not allowed to have children）。
    # 默认 --workers 0 时用不到这条，但保留以支持显式开 workers 的场景。
    ctx = mp.get_context('spawn')
    mgr = ctx.Manager()
    slots = mgr.Queue()
    for i in range(1, a.jobs + 1):
        slots.put(i)
    lock = mgr.RLock()      # Manager 代理锁，spawn 下可安全跨进程传递给 tqdm

    results = []
    main_bar = tqdm(total=len(jobs), position=0, ncols=118, desc='总进度')
    # 全局任务池：所有 (配置, 种子) 一次性扁平提交，固定 a.jobs 个进程并发。
    # 不再「逐配置提交」——那样 5 个种子在 3 并发下会分成 3+2 两批，第二批
    # 空出一个槽位。各任务完全独立，跨配置同时训练不影响正确性；各源的
    # mmap page cache 由所有进程共享（file 上限约 8.5 GiB）。
    with ProcessPoolExecutor(max_workers=a.jobs, mp_context=ctx,
                             initializer=_init, initargs=(lock, slots)) as pool:
        for fut in as_completed([pool.submit(train_one, j) for j in jobs]):
            r = fut.result()
            results.append(r)
            main_bar.update(1)
            tag = 'OK ' if r[4] is None else 'ERR'
            main_bar.write(f'  [{tag}] cfg{r[0]:02d} seed{r[1]} '
                           f'best_avg8_val_pcc {r[2]:.4f} @ep{r[3]}'
                           + (f' — {r[4]}' if r[4] else ''))
    main_bar.close()
    df = pd.DataFrame(results, columns=['config', 'seed', 'best_avg8_val_pcc',
                                        'best_epoch', 'error'])
    df.to_csv(f'{OUT}/summary_{int(time.time())}.csv', index=False)
    nerr = int(df.error.notna().sum())
    print(f'\n完成 {len(df) - nerr} / {len(df)}，失败 {nerr}', flush=True)
    if nerr:
        print(df[df.error.notna()].to_string(), flush=True)
    return 1 if nerr else 0


if __name__ == '__main__':
    sys.exit(main())
