+++++#!/usr/bin/env python
"""
Stage 5: a single script that trains **all** experiment configurations
(each configuration's data composition × source-code default hyperparameters).

  · **Global task pool**: all (config, seed) pairs are submitted flat, with a
    fixed --jobs processes running concurrently.
    An earlier version ran "configs serially, 5 seeds concurrently within a
    config", but at 3-way concurrency the 5 seeds split into two batches of
    3+2, leaving an idle slot in the second batch; the tasks are fully
    independent and training across configs simultaneously does not affect
    correctness, hence the switch to a global pool.
  · Each (config, seed) runs inference on OOD 295 and FEP 42 immediately after
    training and saves the predictions, so stage 6 only does ensembling and
    statistics without reloading any model
  · Resumable: tasks that already have a done.json are skipped
  · Detailed tqdm: a main progress bar + one batch-level bar per concurrent
    slot
    (postfix shows loss / val PCC / 8-epoch mean / best epoch / elapsed time)

━━━ Environment measurements (2026-08-19; set --workers / --jobs defaults) ━━━
This machine is a container; `free`/`nproc` report **host** figures
(503 GB / 128 cores). The real limits are in cgroup v2:
    /sys/fs/cgroup/memory.max = 62 GiB
    /sys/fs/cgroup/cpu.max    = 1200000 100000  →  12 cores
    VRAM: RTX 4080 SUPER 32,760 MiB; ulimit -n is only 1024

Single-process measurements on the largest configuration (cfg08, 148,125
entries / 133,313 for training):

  --workers 3 (old default)    main process + 6 DataLoader subprocesses
      anonymous memory 58 GiB (almost at the 62 GiB limit) → with 5
      concurrent jobs the processes were SIGKILLed, raising
      BrokenProcessPool; and after 6 min 52 s not even 1 epoch had finished
  --workers 0 (current default)    single process slicing directly from mmap
      anonymous memory 288 MiB, file (page cache) 8.3 GiB (reclaimable,
      shared across processes)
      epoch 1 = 45 s, epoch 2 = 41 s, peak VRAM 8,110 MiB, GPU util 34–50%
      2 epochs + loading + two evaluation inference passes: 1.75 min in total

I.e. under the 12-core limit, workers>0 is **more than 5x slower and uses
58 GiB more memory** — the IPC overhead of serializing PyG Data objects batch
by batch far outweighs the gain, whereas workers=0 does zero-copy mmap
slicing. (Separately verified that np.load(mmap_mode='r') works correctly:
after a full read, the memory is charged to cgroup file rather than anon.)

Hence the defaults: --workers 0, --jobs 3 (8.1 GiB × 3 = 24.3 GiB VRAM,
leaving 8 GiB headroom; anonymous memory 3 × 2.5 GiB + shared page cache
8.5 GiB, far below 62 GiB).
Estimate: 200 epochs ≈ 2.3 h/seed (largest configuration); 50 tasks at 3-way
concurrency ≈ 40 h.

━━━━━━━━━━━━━━ Hyperparameter convention (important) ━━━━━━━━━━━━━━
User instruction: "model hyperparameters follow the model source code
defaults", which **overrides the original design**.
Each constant below is taken from AEV-PLIG/training.py:

  Item                  Original (superseded)       Source default (in use)
  batch_size            64                          128
  learning rate         5e-4                        0.00012291937615434127
  weight_decay          1e-5                        0
  max epochs            200                         200
  gradient clipping     1.0                         none
  lr schedule           ReduceLROnPlateau(10, 0.5)  none
  early stopping        val PCC patience=20         none (runs all 200 epochs)
  checkpoint selection  best single-epoch val PCC   best **mean val PCC over
                                                    the last 8 epochs**
  loss function         MSE                         MSE (same)
  optimizer             Adam                        Adam (same)

"Checkpoint = best sliding mean over the last 8 epochs" is what the reference
implementation's _train() does:
  low = max(epoch-7, 0); avg_pc = mean(pcs[low:epoch+1]); if avg_pc > best_pc: save
The criterion is the sliding mean, yet the weights saved are those of the
current epoch; this script reproduces that as-is, without "improving" it.

Retained from the original design (these are not "model hyperparameters"):
  · train/validation 9:1 split with a fixed SPLIT_SEED=42 (reproducibility
    requirement)
  · ensemble size = 5 random seeds (the original design specified 10; the
    user settled on 5)
  · evaluation sets = OOD 295 + FEP 42 (the reference implementation uses the
    PDBbind core set, which is irrelevant to this experiment)

Label standardization: as in the reference implementation, pK is
standardized with the **training set** mean/std (sklearn StandardScaler,
ddof=0) before computing the MSE, and inverse-transformed back to pK at
inference time.

Usage:
  # smoke test
  python 06_train_all.py --configs 1 --seeds 42 43 --max-epochs 3 --limit-train 2000
  # full training (configs 1–9 × seeds 42–46 = 45 tasks, 3-way global
  # concurrency)
  python 06_train_all.py
  # early stopping can be enabled to save time (not source-code behavior;
  # flagged in done.json)
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

# ── Source-code default hyperparameters (AEV-PLIG/training.py) ──
LR = 0.00012291937615434127
WEIGHT_DECAY = 0
BATCH = 128
MAX_EPOCHS = 200
AVG_WINDOW = 8           # checkpoint sliding window (ref: max(epoch-7,0))
# ── Retained from the original design ──
VAL_FRAC = 0.1
# All seeds share the same split; otherwise the ensemble members' validation
# sets would contaminate each other.
SPLIT_SEED = 42
SEEDS = list(range(42, 47))       # 5 seeds
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
        ps.append(model(b).view(-1).float().cpu())      # (B,1) → flat
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

        # 9:1 split (fixed SPLIT_SEED)
        rng = np.random.RandomState(SPLIT_SEED)
        perm = rng.permutation(len(items))
        nval = int(round(len(items) * VAL_FRAC))
        vi, ti = perm[:nval], perm[nval:]
        tr, va = items.iloc[ti].reset_index(drop=True), items.iloc[vi].reset_index(drop=True)
        ysc = fit_y_scaler(tr.y.values)
        P(f'Config {cid} seed {seed}: train {len(tr)} / val {len(va)}; '
          f'y standardization mean={ysc[0]:.4f} std={ysc[1]:.4f}')

        # drop_last=True: every layer of the model has BatchNorm; if the last
        # batch holds only 1 sample, BatchNorm1d in train mode raises
        # "Expected more than 1 value per channel". The reference
        # implementation does not drop (its data size happens not to trigger
        # this), but it must be guarded against here. Data are reshuffled
        # every epoch, so at most 127 entries are dropped, different ones each
        # epoch, with no material effect on training.
        dl = DataLoader(GraphDataset(store, tr, ysc), batch_size=BATCH, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)
        vl = DataLoader(GraphDataset(store, va, ysc), batch_size=512, shuffle=False,
                        num_workers=args.workers, persistent_workers=args.workers > 0)

        model = build_model().to(dev)
        P(f'GATv2Net 5 layers / hidden {DEFAULT_CONFIG.hidden_dim} / head '
          f'{DEFAULT_CONFIG.head} / {DEFAULT_CONFIG.activation_function}; '
          f'params {n_params(model):,}')
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        lossf = nn.MSELoss()

        max_ep = args.max_epochs
        bar = tqdm(total=len(dl), position=slot, leave=False, ncols=118,
                   desc=f'cfg{cid:02d} s{seed} ep 0/{max_ep}', mininterval=0.5)
        # Reference _train: best_pc starts at -1.1; the criterion is the mean
        # val PCC over the last 8 epochs
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
                opt.step()               # no gradient clipping in source
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
            # Source default is no early stopping (--early-patience 0); if >0,
            # count epochs without avg8 improvement
            if args.early_patience and ep - best_ep >= args.early_patience:
                P(f'  early stop: no avg8 improvement for '
                  f'{args.early_patience} epochs (not source-code behavior)')
                break
        bar.close(); bar = None
        pd.DataFrame(hist).to_csv(f'{d}/history.csv', index=False)

        # Run inference on both evaluation sets with the checkpoint
        ck = torch.load(f'{d}/best.pt', map_location=dev)
        model.load_state_dict(ck['model'])
        preds = {}
        for ts in TESTSETS:
            df = load_testset(ts, store)
            tl = DataLoader(GraphDataset(store, df[['source', 'unique_id', 'y']], None),
                            batch_size=512, shuffle=False, num_workers=args.workers)
            _, p, ix = predict(model, tl, dev)
            o = np.full(len(df), np.nan)
            o[ix.numpy()] = p.numpy() * ysc[1] + ysc[0]      # back to pK
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
        P(f'Done: best avg8 val PCC {best_avg:.4f} @ep{best_ep}'
          f' (single-epoch {best_vp:.4f}), {(time.time() - t0) / 60:.1f} min')
        return (cid, seed, best_avg, best_ep, None)
    except Exception as e:
        P('Failed\n' + traceback.format_exc())
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
                    help='global number of concurrent training processes '
                         '(8.1 GiB VRAM/process; 3 is safe on a 32 GiB card)')
    ap.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    ap.add_argument('--early-patience', type=int, default=0,
                    help='0 = source-code behavior (run all max-epochs, no '
                         'early stopping); >0 = stop early when avg8 stops '
                         'improving')
    ap.add_argument('--limit-train', type=int, default=0,
                    help='for smoke tests: subsample the training set')
    ap.add_argument('--workers', type=int, default=0,
                    help='DataLoader subprocesses per training process. '
                         'Default 0 — under this machine\'s 12-core cgroup, '
                         'workers=3 is more than 5x slower than workers=0 and '
                         'uses 58 GiB more anonymous memory; see "Environment '
                         'measurements" in the file header')
    ap.add_argument('--force', action='store_true',
                    help='ignore done.json and rerun')
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
    print(f'{len(jobs)} tasks (configs {a.configs} × seeds {a.seeds}), '
          f'global concurrency {a.jobs}, DataLoader workers {a.workers}, '
          f'up to {a.max_epochs} epochs, '
          f'early stopping '
          f'{"off (source default)" if not a.early_patience else a.early_patience}',
          flush=True)
    print(f'Hyperparams: lr {LR} batch {BATCH} weight_decay {WEIGHT_DECAY}; '
          f'checkpoint = best mean val PCC over the last {AVG_WINDOW} epochs '
          f'(source-code convention)', flush=True)
    if not jobs:
        print('All done.'); return 0

    # Use ProcessPoolExecutor rather than mp.Pool: Pool workers are daemon
    # processes, so with --workers > 0 they could not fork DataLoader
    # subprocesses
    # (AssertionError: daemonic processes are not allowed to have children).
    # Not needed with the default --workers 0, but kept to support scenarios
    # where workers are explicitly enabled.
    ctx = mp.get_context('spawn')
    mgr = ctx.Manager()
    slots = mgr.Queue()
    for i in range(1, a.jobs + 1):
        slots.put(i)
    lock = mgr.RLock()      # Manager proxy lock: spawn-safe to pass to tqdm

    results = []
    main_bar = tqdm(total=len(jobs), position=0, ncols=118, desc='Overall')
    # Global task pool: all (config, seed) pairs are submitted flat in one go,
    # with a fixed a.jobs processes running concurrently. No longer "submit
    # per config" — that way 5 seeds at 3-way concurrency split into two
    # batches of 3+2, leaving one slot idle in the second batch. Tasks are
    # fully independent, so training across configs simultaneously does not
    # affect correctness; each source's mmap page cache is shared by all
    # processes (file usage capped at about 8.5 GiB).
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
    print(f'\nDone {len(df) - nerr} / {len(df)}, failed {nerr}', flush=True)
    if nerr:
        print(df[df.error.notna()].to_string(), flush=True)
    return 1 if nerr else 0


if __name__ == '__main__':
    sys.exit(main())
