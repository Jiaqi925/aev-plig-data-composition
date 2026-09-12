#!/usr/bin/env python
"""
阶段 6：评估与统计推断（PLAN §6.2 / §6.3 / §6.4 / §6.5）。

输入：06_train 各 (配置, 种子) 目录下的 pred_ood295.npy / pred_fep42.npy
     （训练脚本已用各自最优检查点推理并逆标准化回 pK）

产出 experiments/07_eval/
  results.json        全部指标与 CI 的结构化结果
  table_main.csv      主表：每配置 FEP wmPCC/wmKτ + OOD PCC/τ_b/RMSE/MAE（含 95% BCa CI）
  table_single.csv    单模型（逐种子）指标，用于展示集成收益
  per_series/cfgXX.csv 每配置的 42 系列逐系列 r / τ / n
  paired_diff.csv     各配置 vs 配置1 的配对差值 CI（PLAN §6.5：不用区间重叠判断显著性）
  scaling.csv         Scaling Law log-linear 拟合（k, b, R²）
  strat.csv           §6.4 序列相似度分层（探索性）
  gator_best.json     配置 3-8 中 FEP wmPCC 最高者 → 配置 10 用哪份 Gator
  report.md           人读摘要

说明（如实记录在 report.md）：
 · §6.4 分层用「源级」靶点池计算最大序列一致性：Gator 的 30%/60% 规模子集是随机抽样，
   靶点覆盖与 100% 版几乎一致，且 §6.4 本身定位为探索性分析，故不按子集细分。
 · BindingNet / BindingDB-DCS 无独立序列池，其靶点即 PDB 条目，用 PDBbind 序列池代表。
"""
import os, sys, json, argparse, subprocess, shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_data import GraphStore, load_testset, config_meta
import lib_eval as LE

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = f'{ROOT}/experiments'
META = f'{E}/00_meta'
TRAIN = f'{E}/06_train'
OUT = f'{E}/07_eval'
TMP = f'{ROOT}/tmp_07'
log = lambda *a: print(*a, flush=True)

SRC_OF_CONFIG_POOL = {          # 配置组成 → 用于分层的序列池
    'pdbbind': 'pdbbind', 'bindingnet': 'pdbbind', 'bindingdb_dcs': 'pdbbind',
    'hiqbind': 'hiqbind', 'gatoraffinity': 'gator'}


# ───────────────────────── 预测值收集与集成 ─────────────────────────

def collect(cid, seeds):
    """返回 {seed: {'ood295': arr, 'fep42': arr}}，只取 done.json 存在的种子。"""
    out = {}
    for s in seeds:
        d = f'{TRAIN}/cfg{cid:02d}/seed{s}'
        if not os.path.exists(f'{d}/done.json'):
            continue
        try:
            out[s] = {t: np.load(f'{d}/pred_{t}.npy') for t in ['ood295', 'fep42']}
        except FileNotFoundError:
            log(f'  ! cfg{cid:02d} seed{s} 缺预测文件，跳过')
    return out


def ensemble(preds, key):
    """简单算术平均（PLAN §5.3）；逐样本忽略 NaN。"""
    a = np.vstack([preds[s][key] for s in sorted(preds)])
    with np.errstate(invalid='ignore'):
        m = np.nanmean(a, axis=0)
    return m, a


# ───────────────────────── §6.4 序列相似度分层 ─────────────────────────

def write_fa(path, recs):
    with open(path, 'w') as f:
        for k, s in recs:
            if s:
                f.write(f'>{k}\n{s}\n')


AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
       'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
       'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
       'TYR': 'Y', 'VAL': 'V', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O'}


def longest_chain_seq(pdb_path):
    """与 02b_extract_smiles_seq.py 同口径：按 CA 原子解析各链，取最长链。"""
    try:
        chains, seen = {}, set()
        with open(pdb_path, 'r', errors='ignore') as f:
            for ln in f:
                if ln.startswith('ENDMDL'):
                    break
                if ln.startswith('ATOM') and ln[12:16].strip() == 'CA':
                    res = ln[17:20].strip().upper()
                    if res not in AA3:
                        continue
                    rid = (ln[21], ln[22:27])
                    if rid in seen:
                        continue
                    seen.add(rid)
                    chains.setdefault(ln[21], []).append(AA3[res])
        return ''.join(max(chains.values(), key=len)) if chains else ''
    except Exception:
        return ''


def ood_seqs():
    """OOD 295 的靶点序列。注意不能查 pdbbind_smiles_seq.csv —— 那张表是从
    pdbbind_clean.csv 生成的，OOD 295 条已按 §3.4.2 剔除，查不到（会静默变空串，
    导致分层的 OOD 部分整体缺失）。这里直接从 refined-set 的 PDB 结构解析，
    并缓存到 00_meta/ood295_seq.fasta。"""
    cache = f'{META}/ood295_seq.fasta'
    ood = pd.read_csv(f'{META}/testset_ood295.csv')
    if os.path.exists(cache):
        out, cur, buf = [], None, []
        for ln in open(cache):
            if ln.startswith('>'):
                if cur:
                    out.append((cur, ''.join(buf)))
                cur, buf = ln[1:].strip(), []
            else:
                buf.append(ln.strip())
        if cur:
            out.append((cur, ''.join(buf)))
        return out
    recs = []
    for p in ood.pdb_id.unique():
        s = longest_chain_seq(f'{ROOT}/datasets/structure/pdbbind/refined-set/'
                             f'{p}/{p}_protein.pdb')
        recs.append((f'ood|{str(p).lower()}', s))
    write_fa(cache, recs)
    n = sum(1 for _, s in recs if s)
    log(f'  OOD 靶点序列 {n}/{len(recs)} 解析成功 → {cache}')
    return recs


def eval_maxid(force=False):
    """每个评测靶点 vs 每个源靶点池的最大序列一致性 → 00_meta/eval_maxid.csv。"""
    cache = f'{META}/eval_maxid.csv'
    if os.path.exists(cache) and not force:
        return pd.read_csv(cache)
    if shutil.which('mmseqs') is None:
        log('  ! 未找到 mmseqs，跳过分层')
        return None
    os.makedirs(TMP, exist_ok=True)

    # query = OOD 295 靶点序列 + FEP 42 系列靶点序列
    q = list(ood_seqs())
    fep_fa = f'{META}/pool_fep42.fasta'
    if os.path.exists(fep_fa):
        cur, buf = None, []
        for ln in open(fep_fa):
            if ln.startswith('>'):
                if cur:
                    q.append((f'fep|{cur}', ''.join(buf)))
                # header 形如 ">FEP|ptp1b"，只取 | 之后的系列名，
                # 否则会拼成 fep|FEP|ptp1b，与评估侧的 fep|ptp1b 对不上
                cur, buf = ln[1:].strip().split()[0].split('|')[-1], []
            else:
                buf.append(ln.strip())
        if cur:
            q.append((f'fep|{cur}', ''.join(buf)))
    qf = f'{TMP}/query.fasta'
    write_fa(qf, q)
    log(f'  分层 query {sum(1 for _, s in q if s)} 条'
        f'（OOD {sum(1 for k, s in q if k.startswith("ood|") and s)} + '
        f'FEP {sum(1 for k, s in q if k.startswith("fep|") and s)}）')

    rows = []
    for src, fa in [('pdbbind', f'{META}/pool_pdbbind.fasta'),
                    ('hiqbind', f'{META}/pool_hiqbind.fasta'),
                    ('gator', f'{E}/03_gator/pool_gator.fasta')]:
        if not os.path.exists(fa):
            log(f'  ! 缺 {fa}，跳过 {src}')
            continue
        o = f'{TMP}/hits_{src}.tsv'
        cmd = ['mmseqs', 'easy-search', qf, fa, o, f'{TMP}/t_{src}',
               '--min-seq-id', '0.0', '-c', '0.5', '--cov-mode', '0', '-e', '1e-3',
               '-s', '7.5', '--max-seqs', '4000', '--threads', '32',
               '--format-output', 'query,target,fident']
        log(f'  [mmseqs] 评测靶点 vs {src} …')
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log(f'  ! mmseqs 失败：{r.stderr[-400:]}')
            continue
        h = pd.read_csv(o, sep='\t', names=['q', 't', 'fident'])
        g = h.groupby('q').fident.max().reset_index()
        g['src'] = src
        rows.append(g)
    if not rows:
        return None
    d = pd.concat(rows, ignore_index=True)
    d = d.pivot(index='q', columns='src', values='fident').reset_index()
    # 对某个池没有任何命中，说明相似度低于 e-value/覆盖率门槛 —— 补 0 而不是丢弃。
    # 若丢弃，这些 query 在下游 lay.get(i,'unknown') 里既不进 high 也不进 low，
    # 等于白扔样本，而它们恰恰是「低相似度」层最该保留的部分。
    d = pd.DataFrame({'q': [k for k, s in q if s]}).merge(d, on='q', how='left')
    d[[c for c in d.columns if c != 'q']] = d[
        [c for c in d.columns if c != 'q']].fillna(0.0)
    d.to_csv(cache, index=False)
    log(f'  分层缓存 → {cache}')
    return d


def strat_layers(maxid, parts, keys):
    """给定配置组成，返回 {query: 'high'|'low'}（阈值 30%）。"""
    pools = sorted({SRC_OF_CONFIG_POOL[p] for p in parts if p in SRC_OF_CONFIG_POOL})
    cols = [c for c in pools if c in maxid.columns]
    if not cols:
        return {}
    v = maxid[cols].max(axis=1).fillna(0.0).values
    return {q: ('high' if x >= 0.30 else 'low') for q, x in zip(maxid['q'], v)}


# ───────────────────────────── 主流程 ─────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='*', type=int, default=list(range(1, 11)))
    ap.add_argument('--seeds', nargs='*', type=int, default=list(range(42, 47)))
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--baseline', type=int, default=1,
                    help='§6.5 配对差值的基准配置（PLAN 表 2 的配置 1 = baseline）')
    ap.add_argument('--skip-strat', action='store_true')
    ap.add_argument('--force-strat', action='store_true',
                    help='重算 00_meta/eval_maxid.csv 缓存（重跑 mmseqs）')
    ap.add_argument('--single-model', action='store_true',
                    help='额外输出逐种子单模型指标（较慢，每个种子各跑一次 BCa 会很久，'
                         '这里只算点估计）')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(f'{OUT}/per_series', exist_ok=True)

    store = GraphStore(['ood295', 'fep42'])
    ood = load_testset('ood295', store)
    fep = load_testset('fep42', store)
    y_ood = ood.y.values.astype(float)
    y_fep = fep.y.values.astype(float)
    ser = fep.series.astype(str).values
    log(f'评测集：OOD {len(ood)} 条；FEP {len(fep)} 配体 / {len(set(ser))} 系列')

    cmeta = config_meta() if os.path.exists(f'{E}/05_configs/configs.json') else {}
    R, main_rows, single_rows = {}, [], []
    ens = {}   # cid → (ood_pred, fep_pred)

    for cid in a.configs:
        pr = collect(cid, a.seeds)
        if not pr:
            log(f'[配置 {cid:2d}] 无已完成的种子，跳过')
            continue
        po, ao = ensemble(pr, 'ood295')
        pf, af = ensemble(pr, 'fep42')
        ens[cid] = (po, pf)
        mo = np.isfinite(po) & np.isfinite(y_ood)
        mf = np.isfinite(pf) & np.isfinite(y_fep)
        name = cmeta.get(str(cid), {}).get('name', f'config{cid}')
        ntr = cmeta.get(str(cid), {}).get('n', np.nan)
        log(f'[配置 {cid:2d}] {name}：种子 {sorted(pr)}；'
            f'OOD 有效 {int(mo.sum())}/{len(po)}，FEP 有效 {int(mf.sum())}/{len(pf)}')

        fm = LE.fep_metrics(ser[mf], y_fep[mf], pf[mf], n_boot=a.n_boot, seed=cid * 10)
        om = LE.ood_metrics(y_ood[mo], po[mo], n_boot=a.n_boot, seed=cid * 10 + 5)
        R[cid] = {'name': name, 'n_train': ntr, 'seeds': sorted(pr),
                  'n_ood_used': int(mo.sum()), 'n_fep_used': int(mf.sum()),
                  'fep': fm, 'ood': om}
        pd.DataFrame(fm['per_series']).to_csv(f'{OUT}/per_series/cfg{cid:02d}.csv', index=False)

        f = lambda c: f"{c['point']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]"
        main_rows.append({
            'config': cid, 'name': name, 'n_train': ntr, 'n_seeds': len(pr),
            'FEP_wmPCC': fm['wmPCC']['point'], 'FEP_wmPCC_lo': fm['wmPCC']['lo'],
            'FEP_wmPCC_hi': fm['wmPCC']['hi'],
            'FEP_wmKtau': fm['wmKtau']['point'], 'FEP_wmKtau_lo': fm['wmKtau']['lo'],
            'FEP_wmKtau_hi': fm['wmKtau']['hi'],
            'OOD_PCC': om['PCC']['point'], 'OOD_PCC_lo': om['PCC']['lo'],
            'OOD_PCC_hi': om['PCC']['hi'],
            'OOD_Ktau': om['Ktau']['point'], 'OOD_Ktau_lo': om['Ktau']['lo'],
            'OOD_Ktau_hi': om['Ktau']['hi'],
            'OOD_RMSE': om['RMSE']['point'], 'OOD_MAE': om['MAE']['point']})
        log(f'    FEP wmPCC {f(fm["wmPCC"])}  wmKτ {f(fm["wmKtau"])}')
        log(f'    OOD PCC   {f(om["PCC"])}  τ_b {f(om["Ktau"])}  '
            f'RMSE {om["RMSE"]["point"]:.3f}')

        if a.single_model:
            for i, s in enumerate(sorted(pr)):
                o1, f1 = ao[i], af[i]
                m1, m2 = np.isfinite(o1), np.isfinite(f1)
                u = LE.series_stats(ser[m2], y_fep[m2], f1[m2])
                single_rows.append({'config': cid, 'seed': s,
                                    'FEP_wmPCC': LE.weighted_mean(u, 1),
                                    'FEP_wmKtau': LE.weighted_mean(u, 2),
                                    'OOD_PCC': LE.pcc(y_ood[m1], o1[m1]),
                                    'OOD_Ktau': LE.ktau(y_ood[m1], o1[m1]),
                                    'OOD_RMSE': LE.rmse(y_ood[m1], o1[m1])})

    if not R:
        log('没有任何配置有结果，退出。'); return 1
    pd.DataFrame(main_rows).to_csv(f'{OUT}/table_main.csv', index=False)
    if single_rows:
        pd.DataFrame(single_rows).to_csv(f'{OUT}/table_single.csv', index=False)

    # ── gator_best：配置 3-8 中 FEP wmPCC 最高（PLAN §4.1 表2 配置10 的定义）──
    cand = {c: R[c]['fep']['wmPCC']['point'] for c in range(3, 9) if c in R}
    if cand:
        bc = max(cand, key=lambda c: cand[c])
        spec = {3: 'filtered_30', 4: 'filtered_60', 5: 'filtered_100',
                6: 'unfiltered_30', 7: 'unfiltered_60', 8: 'unfiltered_100'}[bc]
        json.dump({'best_config': bc, 'spec': spec, 'wmPCC': cand[bc],
                   'candidates': cand}, open(f'{META}/gator_best.json', 'w'), indent=1)
        log(f'\ngator_best = 配置 {bc}（{spec}），FEP wmPCC {cand[bc]:.4f}'
            f'  → 配置 10 用：python 05_build_configs.py --config10 {spec}')

    # ── §6.5 配对差值 CI（基准 = 配置 a.baseline，默认配置 1）──
    pd_rows = []
    B = a.baseline
    if B in ens:
        po1, pf1 = ens[B]
        for cid in sorted(ens):
            if cid == B:
                continue
            po2, pf2 = ens[cid]
            m = np.isfinite(po1) & np.isfinite(po2) & np.isfinite(y_ood)
            ua = list(zip(y_ood[m], po2[m])); ub = list(zip(y_ood[m], po1[m]))
            d_ood = LE.paired_diff_ci(
                ua, ub, lambda U: LE.pcc([u[0] for u in U], [u[1] for u in U]),
                n_boot=a.n_boot, seed=cid * 100)
            m = np.isfinite(pf1) & np.isfinite(pf2) & np.isfinite(y_fep)
            sa = LE.series_stats(ser[m], y_fep[m], pf2[m])
            sb = LE.series_stats(ser[m], y_fep[m], pf1[m])
            d_fep = LE.paired_diff_ci(sa, sb, lambda U: LE.weighted_mean(U, 1),
                                      n_boot=a.n_boot, seed=cid * 100 + 1)
            sig = lambda c: (np.isfinite(c['lo']) and np.isfinite(c['hi'])
                             and (c['lo'] > 0 or c['hi'] < 0))
            pd_rows.append({'config': cid, 'name': R[cid]['name'], 'vs': B,
                            'dFEP_wmPCC': d_fep['point'], 'dFEP_lo': d_fep['lo'],
                            'dFEP_hi': d_fep['hi'], 'dFEP_sig': bool(sig(d_fep)),
                            'dOOD_PCC': d_ood['point'], 'dOOD_lo': d_ood['lo'],
                            'dOOD_hi': d_ood['hi'], 'dOOD_sig': bool(sig(d_ood))})
            log(f'  Δ配置{cid}−配置{B}：FEP wmPCC {d_fep["point"]:+.4f} '
                f'[{d_fep["lo"]:+.4f}, {d_fep["hi"]:+.4f}] '
                f'{"显著" if sig(d_fep) else "不显著"}；'
                f'OOD PCC {d_ood["point"]:+.4f} '
                f'[{d_ood["lo"]:+.4f}, {d_ood["hi"]:+.4f}] '
                f'{"显著" if sig(d_ood) else "不显著"}')
        pd.DataFrame(pd_rows).to_csv(f'{OUT}/paired_diff.csv', index=False)
    else:
        log(f'  ! 基准配置 {B} 无结果，跳过配对差值')

    # ── Scaling Law ──
    sc_rows = []
    for tag, cs in [('filtered', [3, 4, 5]), ('unfiltered', [6, 7, 8])]:
        cs = [c for c in cs if c in R]
        if len(cs) < 3:
            continue
        sizes = [R[c]['n_train'] for c in cs]
        for metric, get in [('FEP_wmPCC', lambda c: R[c]['fep']['wmPCC']['point']),
                            ('OOD_PCC', lambda c: R[c]['ood']['PCC']['point'])]:
            fit = LE.scaling_fit(sizes, [get(c) for c in cs])
            sc_rows.append({'version': tag, 'metric': metric, 'configs': str(cs),
                            'sizes': str(sizes), **fit})
            log(f'  Scaling[{tag}/{metric}] k={fit["k"]:.4f} b={fit["b"]:.4f} '
                f'R²={fit["r2"]:.4f}')
    if sc_rows:
        pd.DataFrame(sc_rows).to_csv(f'{OUT}/scaling.csv', index=False)

    # ── §6.4 分层（探索性）──
    st_rows = []
    if not a.skip_strat:
        mi = eval_maxid(force=a.force_strat)
        if mi is not None:
            for cid in sorted(ens):
                parts = cmeta.get(str(cid), {}).get('parts', [])
                base = [p.split('_')[0] if p.startswith('gator') else p for p in parts]
                base = ['gatoraffinity' if p == 'gator' else p for p in base]
                lay = strat_layers(mi, base, None)
                po, pf = ens[cid]
                for lab, ids, y, p, kind in [
                        ('OOD', ['ood|' + str(x).lower() for x in ood.pdb_id],
                         y_ood, po, 'ood'),
                        ('FEP', ['fep|' + str(x) for x in ser], y_fep, pf, 'fep')]:
                    tags = np.array([lay.get(i, 'unknown') for i in ids])
                    for L in ['high', 'low']:
                        m = (tags == L) & np.isfinite(p) & np.isfinite(y)
                        if m.sum() < 5:
                            continue
                        if kind == 'fep':
                            u = LE.series_stats(np.array(ids)[m], y[m], p[m])
                            v1, v2 = LE.weighted_mean(u, 1), LE.weighted_mean(u, 2)
                        else:
                            v1, v2 = LE.pcc(y[m], p[m]), LE.ktau(y[m], p[m])
                        st_rows.append({'config': cid, 'set': lab, 'layer': L,
                                        'n': int(m.sum()), 'pcc': v1, 'ktau': v2})
            if st_rows:
                pd.DataFrame(st_rows).to_csv(f'{OUT}/strat.csv', index=False)
                log(f'  分层结果 → {OUT}/strat.csv')

    json.dump({'configs': R, 'paired_diff': pd_rows, 'scaling': sc_rows,
               'strat': st_rows, 'n_boot': a.n_boot,
               'n_ood': int(len(ood)), 'n_fep': int(len(fep)),
               'n_fep_series': int(len(set(ser)))},
              open(f'{OUT}/results.json', 'w'), indent=1, default=float)

    # ── report.md ──
    with open(f'{OUT}/report.md', 'w') as fo:
        fo.write('# 阶段 6 评估结果\n\n')
        fo.write(f'- BCa Bootstrap n={a.n_boot}（FEP 系列级重采样，OOD 样本级重采样）\n')
        fo.write(f'- FEP：{len(set(ser))} 系列 / {len(fep)} 配体；OOD：{len(ood)} 条\n')
        fo.write('- 参考值（Valsson 等 2024）：FEP wmPCC 0.59、wmKτ 0.42\n\n')
        fo.write('## 主表\n\n')
        fo.write(pd.DataFrame(main_rows).to_markdown(index=False, floatfmt='.4f'))
        if pd_rows:
            fo.write(f'\n\n## 与配置 {B} 的配对差值（95% BCa CI 不含 0 即显著）\n\n')
            fo.write(pd.DataFrame(pd_rows).to_markdown(index=False, floatfmt='.4f'))
        if sc_rows:
            fo.write('\n\n## Scaling Law（value = k·log10(N) + b）\n\n')
            fo.write(pd.DataFrame(sc_rows).to_markdown(index=False, floatfmt='.4f'))
        if st_rows:
            fo.write('\n\n## 序列相似度分层（探索性，阈值 30%）\n\n')
            fo.write(pd.DataFrame(st_rows).to_markdown(index=False, floatfmt='.4f'))
            sd = pd.DataFrame(st_rows)
            for s in sd['set'].unique():
                miss = {'high', 'low'} - set(sd.loc[sd['set'] == s, 'layer'])
                if miss:
                    fo.write(f'\n\n> {s} 缺「{"/".join(sorted(miss))}」层：该层有效样本 < 5 条，'
                             f'即所有 {s} 靶点与训练源靶点池的最大序列一致性都落在同一侧，'
                             f'该评测集无法做相似度分层。\n')
        fo.write('\n\n### 口径说明\n'
                 '- 分层用源级靶点池（Gator 规模子集为随机抽样，靶点覆盖与 100% 版基本一致）。\n'
                 '- BindingNet / BindingDB-DCS 无独立序列池，其靶点为 PDB 条目，'
                 '用 PDBbind 序列池代表。\n')
    log(f'\n产物 → {OUT}/{{results.json, table_main.csv, paired_diff.csv, '
        f'scaling.csv, strat.csv, per_series/, report.md}}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
