#!/usr/bin/env python
"""
阶段 4.5：数据处理全面合规审计 —— 独立重算，不信任各阶段脚本的自报数字。

逐条对照 PLAN：
  A 原始规模 vs PLAN 表 1
  B 泄漏控制（§3.4.2）：所有源 PDB ID 前 4 位 ∩ (OOD 295 ∪ FEP) 必须为空
  C HiQBind 去重（§3.2）：计数闭合 + 三重判据日志抽检
  D Gator 评测集分离（§3.3.1）：被剔除/保留行与判据一致
  E Gator 规模采样（§4.1）：30% ⊂ 60% ⊂ 100%、比例正确、种子 42 可复现
  F 版本关系（§3.3.2）：筛选版 = 未筛选版 ∩ (confidence ≥ 0.7)
  G pK 口径：各源分布与换算规则
  H 评测集（§2.2 / §6.2.1）：OOD 295 条、FEP 42 系列且每系列 n ≥ 10
  I FEP pK 换算复核（ΔG = −RT·ln10·pK，RT·ln10 = 1.358985 kcal/mol @297K）
  J 图特征（§5.1 输入维度）：节点 367 维、边 4 维、条数与索引表对齐
  K 配置装配（§4.1 表 2）：各配置计数 = 组成部分之和，且不含泄漏 ID
  L 配置级泄漏红线：每个 config_XX.csv 独立回查 —— 训练样本的 PDB ID 与
    unique_id 都不得与 OOD 295 / FEP 42 重合（用户明确要求的硬约束）

退出码 0 = 全部通过（含 SKIP），1 = 有 FAIL。
"""
import os, sys, json, math
import numpy as np
import pandas as pd

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = f'{ROOT}/experiments'
META = f'{E}/00_meta'
G = f'{E}/04_graphs'
RT_LN10 = 1.987204259e-3 * 297.0 * math.log(10.0)

RES = []
def chk(tag, ok, msg=''):
    s = 'SKIP' if ok is None else ('PASS' if bool(ok) else 'FAIL')
    RES.append((s, tag, msg))
    print(f'[{s}] {tag}' + (f' — {msg}' if msg else ''), flush=True)

def head(t):
    print(f'\n══ {t} ' + '═' * max(0, 60 - len(t)), flush=True)


def main():
    # ─────────────────── A 原始规模 ───────────────────
    head('A 原始规模 vs PLAN 表 1')
    r1 = json.load(open(f'{E}/01_baseline_index/report.json'))
    r2a = json.load(open(f'{E}/02_hiqbind/report_02a.json'))
    r3a = json.load(open(f'{E}/03_gator/report_03a.json'))
    plan_raw = {'PDBbind': 19443, 'BindingNet': 69816, 'BindingDB-DCS': 8822,
                'HiQBind': 32275, 'GatorAffinity(结构数)': 65031}
    got = {'PDBbind': r1['pdbbind']['raw'], 'BindingNet': r1['bindingnet']['raw'],
           'BindingDB-DCS': r1['bindingdb_dcs']['raw'], 'HiQBind': r2a['raw'],
           'GatorAffinity(结构数)': r3a['structure_dirs']}
    for k, v in plan_raw.items():
        d = got[k] - v
        chk(f'A.{k}', d == 0 or abs(d) <= 30,
            f'PLAN {v} / 实测 {got[k]}（差 {d:+d}）'
            + ('，PLAN 表 1 为约数，差异记入交接文档' if d else ''))

    # ─────────────────── B 泄漏控制 ───────────────────
    head('B 泄漏控制（§3.4.2）')
    leak = json.load(open(f'{META}/leakage_ids.json'))
    ood_ids, fep_ids = set(leak['ood_pdb_ids']), set(leak['fep_pdb_ids'])
    excl = set(leak['excluded_pdb_ids'])
    chk('B.0 剔除集合 = OOD ∪ FEP', excl == (ood_ids | fep_ids),
        f'OOD {len(ood_ids)} + FEP {len(fep_ids)} → 并集 {len(ood_ids | fep_ids)}，'
        f'剔除集 {len(excl)}')
    srcs = {'pdbbind': f'{E}/01_baseline_index/pdbbind_clean.csv',
            'bindingnet': f'{E}/01_baseline_index/bindingnet_clean.csv',
            'bindingdb_dcs': f'{E}/01_baseline_index/bindingdb_dcs_clean.csv',
            'hiqbind': f'{E}/02_hiqbind/hiqbind_dedup.csv',
            'gatoraffinity': f'{E}/03_gator/gator_dedup.csv'}
    tab = {}
    for s, p in srcs.items():
        d = pd.read_csv(p, low_memory=False)
        tab[s] = d
        if 'pdb_id' in d.columns:
            pid = d.pdb_id.fillna('').astype(str).str.lower().str[:4]
            hit = sorted(set(pid[pid.isin(excl) & (pid != '')]))
        elif 'PDB ID(s) for Ligand-Target Complex' in d.columns:
            # Gator 无真实 PDB ID，但 BindingDB 元数据里偶有该字段（逗号分隔多个），
            # 03b 步骤 1 已按它做安全网剔除，这里独立复核
            col = d['PDB ID(s) for Ligand-Target Complex'].fillna('').astype(str)
            hit = sorted({t.strip().lower()[:4] for v in col for t in v.split(',')
                          if t.strip()} & excl)
        else:
            chk(f'B.{s}', None, '无 PDB ID 列')
            continue
        chk(f'B.{s} PDB ID 无泄漏', len(hit) == 0,
            f'{len(d)} 条，命中 {len(hit)}' + (f' {hit[:8]}' if hit else ''))

    # ─────────────────── C HiQBind 去重 ───────────────────
    head('C HiQBind 去重（§3.2）')
    r2c = json.load(open(f'{E}/02_hiqbind/report_02c.json'))
    chk('C.1 计数闭合', r2c['hiqbind_in'] - r2c['duplicates_removed'] == r2c['net_new']
        == len(tab['hiqbind']),
        f"{r2c['hiqbind_in']} − {r2c['duplicates_removed']} = {r2c['net_new']}，"
        f"文件 {len(tab['hiqbind'])} 行")
    chk('C.2 阈值与 PLAN 一致',
        r2c['thresholds'] == {'tanimoto': 0.9, 'abs_dpK': 1.0, 'tm_score': 0.8,
                             'tm_norm': 'max(chain1,chain2)'},
        str(r2c['thresholds']))
    dp = pd.read_csv(f'{E}/02_hiqbind/hiqbind_dup_pairs.csv')
    bad = dp[(dp.tanimoto <= 0.9) | (dp.abs_dpK > 1.0) | (dp.tm_score <= 0.8)]
    chk('C.3 判定为重复的对确实同时满足三判据', len(bad) == 0,
        f'{len(dp)} 对，违规 {len(bad)}')
    rm = set(dp.hq_unique_id)
    if rm:
        chk('C.4 被判重的样本已从产物剔除',
            len(rm & set(tab['hiqbind'].unique_id)) == 0,
            f'判重 {len(rm)} 个 uid，残留 {len(rm & set(tab["hiqbind"].unique_id))}')

    # ─────────────────── D Gator 评测集分离 ───────────────────
    head('D Gator 评测集分离与跨库去重（§3.3.1）')
    r3b = json.load(open(f'{E}/03_gator/report_03b.json'))
    g = tab['gatoraffinity']
    fl = pd.read_csv(f'{E}/03_gator/gator_testset_flags.csv')
    chk('D.1 计数闭合',
        r3b['gator_in'] - r3b['step1_pdbid_leak'] - r3b['step2']['removed_AND']
        - r3b['step3']['duplicates_removed'] == r3b['gator_final_unfiltered'] == len(g),
        f"{r3b['gator_in']} − {r3b['step1_pdbid_leak']} − {r3b['step2']['removed_AND']}"
        f" − {r3b['step3']['duplicates_removed']} = {r3b['gator_final_unfiltered']}，"
        f'文件 {len(g)} 行')
    if {'seq_hit', 'lig_hit'} <= set(fl.columns):
        both = fl[(fl.seq_hit) & (fl.lig_hit)]
        chk('D.2 同时命中序列+配体的行已剔除',
            len(set(both.unique_id) & set(g.unique_id)) == 0,
            f'AND 命中 {len(both)} 行，残留 {len(set(both.unique_id) & set(g.unique_id))}')
        chk('D.3 读法 A/B 数量已如实记录',
            r3b['step2']['would_remove_OR'] > r3b['step2']['removed_AND'],
            f"AND 剔 {r3b['step2']['removed_AND']}，OR 本会剔 "
            f"{r3b['step2']['would_remove_OR']}（差 "
            f"{r3b['step2']['would_remove_OR'] - r3b['step2']['removed_AND']}）")
    gp = pd.read_csv(f'{E}/03_gator/gator_dup_pairs.csv')
    if len(gp):
        bad = gp[(gp.tanimoto <= 0.9) | (gp.abs_dpK > 1.0) | (gp.tm_score <= 0.8)]
        chk('D.4 跨库判重的对同时满足三判据', len(bad) == 0,
            f'{len(gp)} 对，违规 {len(bad)}')

    # ─────────────────── E/F Gator 采样与版本 ───────────────────
    head('E 规模采样 + F 版本关系（§4.1 / §3.3.2）')
    for ver in ['unfiltered', 'filtered']:
        c = {p: set(g.loc[g[f'in_{ver}_{p}'], 'unique_id']) for p in [30, 60, 100]}
        chk(f'E.{ver} 嵌套 30 ⊂ 60 ⊂ 100',
            c[30] <= c[60] <= c[100],
            f'{len(c[30])} ⊂ {len(c[60])} ⊂ {len(c[100])}')
        base = len(c[100])
        ok = all(abs(len(c[p]) - round(base * p / 100)) <= 1 for p in [30, 60])
        chk(f'E.{ver} 比例正确', ok,
            f'30%: {len(c[30])} vs {round(base * .3)}；60%: {len(c[60])} vs '
            f'{round(base * .6)}')
        # 用种子 42 复现抽样
        idx = g.index.values if ver == 'unfiltered' else g.index[g.pass_conf].values
        order = np.random.RandomState(42).permutation(idx)
        rep = set(g.loc[order[:round(len(order) * 0.3)], 'unique_id'])
        chk(f'E.{ver} 种子 42 可复现', rep == c[30],
            f'重算 {len(rep)} 条，与产物差 {len(rep ^ c[30])}')
    chk('F 筛选版 = 未筛选 ∩ conf≥0.7',
        set(g.loc[g[f'in_filtered_100'], 'unique_id']) ==
        set(g.loc[g.pass_conf, 'unique_id']),
        f"未筛选 {int(g['in_unfiltered_100'].sum())}，"
        f"筛选 {int(g['in_filtered_100'].sum())}，conf 通过 {int(g.pass_conf.sum())}")
    if 'conf' in g.columns:
        v = g.loc[g.pass_conf, 'conf']
        chk('F.2 筛选版 confidence 全部 ≥ 0.7', float(v.min()) >= 0.7 - 1e-9,
            f'min={float(v.min()):.4f}')

    # ─────────────────── G pK 口径 ───────────────────
    head('G pK 口径（§3.1 换算规则）')
    # PLAN 未规定 pK 截断范围，故不过滤，只如实报告分布与极端值条数。
    # pK < 0 的记录源于原始库中极弱结合（如 Kd 达 M 量级）的实验测量，按原样保留。
    for s, d in tab.items():
        v = pd.to_numeric(d.pK, errors='coerce')
        neg = int((v < 0).sum()); hi = int((v > 14).sum())
        chk(f'G.{s} pK 全部可解析', bool(v.notna().all()),
            f'n={len(v)} min={v.min():.2f} max={v.max():.2f} mean={v.mean():.3f}；'
            f'pK<0 共 {neg} 条，pK>14 共 {hi} 条（PLAN 未要求截断，按原样保留）')

    # ─────────────────── H 评测集 ───────────────────
    head('H 评测集规模（§2.2 / §6.2.1）')
    ood = pd.read_csv(f'{META}/testset_ood295.csv')
    chk('H.1 OOD 295 条且 PDB ID 唯一',
        len(ood) == 295 and ood.pdb_id.nunique() == 295,
        f'{len(ood)} 条 / {ood.pdb_id.nunique()} 个唯一 ID')
    fep = pd.read_csv(f'{META}/testset_fep42.csv')
    cnt = fep.series.value_counts()
    chk('H.2 FEP 42 系列且每系列 n ≥ 10',
        fep.series.nunique() == 42 and cnt.min() >= 10,
        f'{fep.series.nunique()} 系列 / {len(fep)} 配体，最小系列 {cnt.min()}，'
        f'最大 {cnt.max()}')
    chk('H.3 FEP 配体数 1087（PLAN 写 1089，差 2 为无实验 ΔG 的中间体节点）',
        len(fep) == 1087, f'{len(fep)} 条')

    # ─────────────────── I FEP pK 换算 ───────────────────
    head('I FEP pK 换算复核')
    if 'dG_exp' in fep.columns:
        back = -fep.dG_exp.astype(float) / RT_LN10
        err = float(np.abs(back - fep.pK.astype(float)).max())
        chk('I.1 pK = −ΔG / (RT·ln10)', err < 1e-6,
            f'RT·ln10 = {RT_LN10:.6f} kcal/mol，最大偏差 {err:.2e}')
    else:
        chk('I.1', None, 'testset_fep42.csv 无 dG_exp 列，跳过')

    # ─────────────────── J 图特征 ───────────────────
    head('J 图特征维度与对齐（§5.1）')
    for s in ['ood295', 'fep42', 'pdbbind', 'bindingnet', 'bindingdb_dcs',
              'hiqbind', 'gatoraffinity']:
        d = f'{G}/{s}'
        if not os.path.exists(f'{d}/ids.npy'):
            chk(f'J.{s}', None, '图未生成')
            continue
        x = np.load(f'{d}/x.npy', mmap_mode='r')
        ea = np.load(f'{d}/ea.npy', mmap_mode='r')
        ei = np.load(f'{d}/ei.npy', mmap_mode='r')
        ids = np.load(f'{d}/ids.npy')
        np_ = np.load(f'{d}/node_ptr.npy'); ep = np.load(f'{d}/edge_ptr.npy')
        ok = (x.shape[1] == 367 and ea.shape[1] == 4 and ei.shape[0] == 2
              and len(np_) == len(ids) + 1 and len(ep) == len(ids) + 1
              and int(np_[-1]) == x.shape[0] and int(ep[-1]) == ea.shape[0]
              and len(set(ids.tolist())) == len(ids))
        chk(f'J.{s} 维度/指针/唯一性', ok,
            f'{len(ids)} 图，x{tuple(x.shape)} ea{tuple(ea.shape)} ei{tuple(ei.shape)}')

    # ─────────────────── K 配置装配 ───────────────────
    head('K 配置装配（§4.1 表 2）')
    cf = f'{E}/05_configs/configs.json'
    if not os.path.exists(cf):
        chk('K', None, '配置未装配')
    else:
        cm = json.load(open(cf))
        for cid, m in sorted(cm.items(), key=lambda kv: int(kv[0])):
            d = pd.read_csv(f'{E}/05_configs/config_{int(cid):02d}.csv')
            chk(f'K.{cid} {m["name"]}',
                len(d) == m['n'] and d.unique_id.nunique() == len(d),
                f'{len(d)} 条，来源分布 {m["by_source"]}')

    # ───────── L 配置级泄漏红线（用户明确要求的硬约束）─────────
    head('L 配置级泄漏红线：任何配置的训练数据都不得与 OOD/FEP 重合')
    if not os.path.exists(cf):
        chk('L', None, '配置未装配')
    else:
        # unique_id → 该样本关联的 PDB ID 集合。
        # 四个基线源有 pdb_id 列；Gator 没有真实 PDB ID，但 BindingDB 元数据的
        # 「PDB ID(s) for Ligand-Target Complex」字段偶有值（52773 行中 4149 行），
        # 用它做精确复合物级复核 —— 这是 B.gatoraffinity 的配置级对应项。
        u2p = {}
        for s, d in tab.items():
            if 'pdb_id' in d.columns:
                for u, p in zip(d.unique_id.astype(str),
                                d.pdb_id.fillna('').astype(str)):
                    u2p[u] = {p.lower()[:4]} if p else set()
            elif 'PDB ID(s) for Ligand-Target Complex' in d.columns:
                col = d['PDB ID(s) for Ligand-Target Complex'].fillna('').astype(str)
                for u, v in zip(d.unique_id.astype(str), col):
                    u2p[u] = {t.strip().lower()[:4]
                              for t in v.replace(';', ',').split(',') if t.strip()}
        ood_uid = set(pd.read_csv(f'{META}/testset_ood295.csv').pdb_id.astype(str))
        ft = pd.read_csv(f'{META}/testset_fep42.csv')
        fep_uid = {f'{a}__{b}' for a, b in zip(ft.series.astype(str), ft.lig_idx)}
        cm2 = json.load(open(cf))
        for cid, m in sorted(cm2.items(), key=lambda kv: int(kv[0])):
            d = pd.read_csv(f'{E}/05_configs/config_{int(cid):02d}.csv')
            u = set(d.unique_id.astype(str))
            hp = sorted({x for uu in u for x in u2p.get(uu, ()) if x in excl})
            hu = sorted((u & ood_uid) | (u & fep_uid))
            chk(f'L.{cid} {m["name"]}', not hp and not hu,
                f'{len(d)} 条：PDB ID 命中 {len(hp)}{hp[:6] if hp else ""}，'
                f'unique_id 命中 {len(hu)}{hu[:6] if hu else ""}')

    head('汇总')
    n = {k: sum(1 for s, _, _ in RES if s == k) for k in ['PASS', 'FAIL', 'SKIP']}
    print(f"PASS {n['PASS']}  FAIL {n['FAIL']}  SKIP {n['SKIP']}", flush=True)
    for s, t, m in RES:
        if s == 'FAIL':
            print(f'  ✗ {t} — {m}', flush=True)
    json.dump([{'status': s, 'check': t, 'detail': m} for s, t, m in RES],
              open(f'{E}/audit_report.json', 'w'), indent=1, ensure_ascii=False)
    print(f'\n产物 → {E}/audit_report.json', flush=True)
    return 1 if n['FAIL'] else 0


if __name__ == '__main__':
    sys.exit(main())
