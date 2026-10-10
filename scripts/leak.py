import pandas as pd, os, json
_R=os.environ.get('STACEY_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
B=f'{_R}/datasets/index/_testsets'
o=pd.read_csv(f'{B}/index_oodtest.csv')
ood=sorted(o[o.split=='test']['PDB_code'].str.lower().unique())
print('OOD test PDB ID:', len(ood)); print(ood[:10])
f=pd.read_csv(f'{B}/fep_benchmark_inputs/benchmark_metadata.csv')
sel=f[f['Number of nodes']>=10]
print()
print('FEP series n>=10:', len(sel),
      'total ligands:', int(sel['Number of nodes'].sum()))
fep=sorted(set(sel['Reference PDB'].str.lower()))
print('FEP reference PDB IDs:', len(fep)); print(fep)
both=set(ood)&set(fep)
print()
print('OOD ∩ FEP:', len(both), sorted(both))
allid=sorted(set(ood)|set(fep))
print('Union:', len(allid))
os.makedirs(f'{_R}/experiments/00_meta', exist_ok=True)
json.dump({'ood_pdb_ids':ood,'fep_pdb_ids':fep,'excluded_pdb_ids':allid,
           'fep_series':int(len(sel)),'fep_ligands':int(sel['Number of nodes'].sum())},
          open(f'{_R}/experiments/00_meta/leakage_ids.json','w'), indent=1)
print('Wrote experiments/00_meta/leakage_ids.json')
