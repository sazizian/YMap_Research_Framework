from pathlib import Path
import json,itertools,math,argparse,hashlib
from datetime import datetime,timezone
import numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr
parser=argparse.ArgumentParser(description="Read-only independent consistency audit of one Y-Map evaluation review bundle")
parser.add_argument('--input',required=True,help='Extracted review directory containing candidates.jsonl, measurements/, and one evaluation directory')
parser.add_argument('--out',required=True,help='Separate scratch/output directory; original inputs are not modified')
parser.add_argument('--evaluation-subdir',default=None,help='Optional evaluation directory name when several are supplied')
args=parser.parse_args()
ROOT=Path(args.input).expanduser().resolve()
if args.evaluation_subdir:
 BASE=ROOT/args.evaluation_subdir
else:
 options=[path for path in ROOT.iterdir() if path.is_dir() and (path/'fold_audit.json').is_file()]
 if len(options)!=1:raise ValueError('Supply exactly one evaluation directory or pass --evaluation-subdir')
 BASE=options[0]
OUT=Path(args.out).expanduser().resolve()
if OUT==ROOT or OUT.is_relative_to(ROOT):raise ValueError('Use an output directory outside original inputs')
if OUT.exists() and any(OUT.iterdir()):raise ValueError('Use a new empty output directory')
OUT.mkdir(parents=True,exist_ok=True)
source_files=[ROOT/'candidates.jsonl',ROOT/'measurements/measurements.csv',ROOT/'measurements/raw_measurements.jsonl',*[BASE/name for name in ['fold_audit.json','summary.json','paired_statistics.json','recommendations.json','training_warnings.json','oof_predictions.csv','calibration_predictions.csv','per_system_metrics.csv','aggregated_measurements.csv','evaluation_config.json']]]
input_hashes={path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
OBJ=['insert_ms','query_ms','storage_mb']
audit=json.loads((BASE/'fold_audit.json').read_text());summ=json.loads((BASE/'summary.json').read_text());stats=json.loads((BASE/'paired_statistics.json').read_text());recs=json.loads((BASE/'recommendations.json').read_text());warnings=json.loads((BASE/'training_warnings.json').read_text())
source=pd.read_csv(ROOT/'measurements/measurements.csv').set_index('schema_id');agg=pd.read_csv(BASE/'aggregated_measurements.csv').set_index('schema_id');pred=pd.read_csv(BASE/'oof_predictions.csv');cal=pd.read_csv(BASE/'calibration_predictions.csv');metrics=pd.read_csv(BASE/'per_system_metrics.csv');candidates=[json.loads(line) for line in (ROOT/'candidates.jsonl').read_text().splitlines() if line.strip()];byid={r['schema_id']:r for r in candidates}
issues=[];diffs=[];check_count=0;numeric_comparisons=0;max_absolute_numeric_difference=0.0
def check(condition,description):
 global check_count
 check_count+=1
 if not condition:issues.append(description)
def equal(value,expected,description,tol=2e-9):
 global numeric_comparisons,max_absolute_numeric_difference
 numeric_comparisons+=1
 if pd.notna(value) and pd.notna(expected):max_absolute_numeric_difference=max(max_absolute_numeric_difference,float(abs(value-expected)))
 if pd.isna(value) and pd.isna(expected):return
 if not np.isclose(value,expected,rtol=tol,atol=tol):diffs.append({'description':description,'supplied':float(value),'recomputed':float(expected)})
def front(v):return np.array([not np.any(np.all(v<=point,axis=1)&np.any(v<point,axis=1)) for point in v])
def hv(v,ref=np.array([1.1]*3)):
 if not len(v):return 0.
 v=np.unique(v[front(v)],axis=0);v=v[np.all(v<ref,axis=1)]
 if not len(v):return 0.
 xs=np.unique(np.r_[v[:,0],ref[0]]);ys=np.unique(np.r_[v[:,1],ref[1]]);zs=np.unique(np.r_[v[:,2],ref[2]])
 cells=np.outer(np.diff(ys),np.diff(zs));total=0.
 for x,dx in zip(xs[:-1],np.diff(xs)):
  dominated=np.zeros(cells.shape,dtype=bool)
  for point in v[v[:,0]<=x]:dominated|=(ys[:-1,None]>=point[1])&(zs[None,:-1]>=point[2])
  total+=dx*cells[dominated].sum()
 return float(total)
check(len(source)==728 and source.index.is_unique,'Source measurement count/identity mismatch')
check(len(byid)==728 and len(candidates)==728,'Candidate ID count mismatch')
check(set(source.index)==set(byid),'Candidate/measurement ID mismatch')
check(len(metrics)==77 and len(recs)==77 and len(pred)==4368,'Expected result counts mismatch')
check(not pred.duplicated(['model','schema_id']).any(),'OOF duplicate model/schema')
check(set(pred.schema_id)==set(source.index),'OOF incomplete schemas')
for name,rows in pred.groupby('model'):check(len(rows)==728,f'Model {name} incomplete OOF')
for sid,row in agg.iterrows():
 for obj in OBJ:equal(row[obj],source.loc[sid,obj],f'Aggregated {sid}/{obj}')
folds={a['fold']:a for a in audit};allids=set(source.index);test_seen=[]
for a in audit:
 parts=[set(a[k]) for k in ['training_schema_ids','calibration_schema_ids','test_schema_ids']];groups=[set(a['training_groups']),set(a['calibration_groups']),{a['test_group']}]
 check(all(not parts[i]&parts[j] for i in range(3) for j in range(i)),'Overlapping schema split '+str(a['fold']))
 check(all(not groups[i]&groups[j] for i in range(3) for j in range(i)),'Overlapping group split '+str(a['fold']))
 check(set.union(*parts)==allids,'Split coverage mismatch '+str(a['fold']))
 check([len(g) for g in groups]==[7,3,1],'Unexpected group counts '+str(a['fold']))
 for ids,gs in zip(parts,groups):check(set(source.loc[list(ids)].object_model)==gs,'ID metadata/group mismatch '+str(a['fold']))
 test_seen+=list(parts[2]);train=source.loc[a['training_schema_ids'],OBJ].to_numpy();normal=a['selection_normalization'];equal(np.max(abs(train.min(0)-normal['minimum'])),0,'Training min '+str(a['fold']));equal(np.max(abs(np.ptp(train,axis=0)-normal['scale'])),0,'Training span '+str(a['fold']))
check(len(test_seen)==len(set(test_seen))==728,'Not exactly once test-fold assignment')
for row in pred.itertuples(index=False):
 sid=row.schema_id;a=folds[row.fold]
 check(sid in a['test_schema_ids'] and row.object_model==a['test_group'],'OOF fold/ID violation '+sid)
 for obj in OBJ:equal(getattr(row,'true_'+obj),source.loc[sid,obj],f'OOF true {row.model}/{sid}/{obj}')
 if row.model=='ensemble5':
  for obj in OBJ:
   vals=np.array([getattr(row,f'member_{i}_{obj}') for i in range(5)])
   equal(vals.mean(),getattr(row,'mean_'+obj),'Ensemble mean '+sid+obj);equal(vals.std(ddof=1),getattr(row,'std_'+obj),'Ensemble std '+sid+obj)
for row in cal.itertuples(index=False):
 a=folds[row.fold];check(row.schema_id in a['calibration_schema_ids'],'Invalid calibration membership')
 for obj in OBJ:equal(getattr(row,'true_'+obj),source.loc[row.schema_id,obj],'Cal truth '+row.schema_id+obj)
for m in metrics.itertuples(index=False):
 label=m.prediction_model;p=pred[(pred.model==label)&(pred.fold==m.fold)];cp=cal[(cal.model==label)&(cal.fold==m.fold)];a=folds[m.fold]
 for obj in OBJ:
  errors=p['mean_'+obj].to_numpy()-p['true_'+obj].to_numpy();rmse=np.sqrt(np.mean(errors**2));span=np.ptp(source.loc[a['training_schema_ids'],obj]);ss=np.sum((p['true_'+obj]-p['true_'+obj].mean())**2)
  equal(getattr(m,obj+'_mae'),np.abs(errors).mean(),'MAE '+m.model+str(m.fold)+obj);equal(getattr(m,obj+'_rmse'),rmse,'RMSE '+m.model+str(m.fold)+obj);equal(getattr(m,obj+'_nrmse'),rmse/span,'nRMSE '+m.model+str(m.fold)+obj);equal(getattr(m,obj+'_r2'),1-np.sum(errors**2)/ss,'R2 '+m.model+str(m.fold)+obj)
  radius=np.quantile(np.abs(cp['true_'+obj]-cp['mean_'+obj]),.9,method='higher');equal(getattr(m,obj+'_interval_width'),2*radius,'Interval width '+m.model+str(m.fold)+obj);equal(getattr(m,obj+'_interval_coverage'),np.mean(np.abs(errors)<=radius),'Interval coverage '+m.model+str(m.fold)+obj)
  equal(np.max(np.abs((p['upper_'+obj]-p['lower_'+obj]).to_numpy()/2-radius)),0,'Stored interval radius '+m.model+str(m.fold)+obj)
  if label=='ensemble5' and np.ptp(p['std_'+obj])>0:equal(getattr(m,obj+'_uncertainty_error_spearman'),spearmanr(np.abs(errors),p['std_'+obj]).statistic,'Spearman '+str(m.fold)+obj)
for r in recs:
 m=r['model'];group=r['object_model'];metric=metrics[(metrics.model==m)&(metrics.object_model==group)].iloc[0];fold=int(metric.fold);p=pred[(pred.model==metric.prediction_model)&(pred.fold==fold)];ids=p.schema_id.tolist();y=p[['true_'+o for o in OBJ]].to_numpy();mu=p[['mean_'+o for o in OBJ]].to_numpy();std=p[['std_'+o for o in OBJ]].to_numpy();norm=folds[fold]['selection_normalization'];risk=(mu-norm['minimum'])/norm['scale']+metric.risk_lambda*std/norm['scale'];selected=np.array([sid in r['selected_schema_ids'] for sid in ids]);expected=front(risk)
 check(np.array_equal(selected,expected),'Selection not predicted-risk Pareto '+m+'/'+group)
 truefront=front(y);tp=(truefront&selected).sum();pr=tp/selected.sum();rc=tp/truefront.sum();f1=2*pr*rc/(pr+rc) if pr+rc else 0.
 scale=np.ptp(y,axis=0);yn=(y-y.min(0))/np.where(scale>0,scale,1);sel=yn[selected];tf=yn[truefront]
 vals={'precision':pr,'recall':rc,'f1':f1,'gd':np.linalg.norm(sel[:,None,:]-tf[None,:,:],axis=2).min(1).mean(),'igd_plus':np.linalg.norm(np.maximum(sel[None,:,:]-tf[:,None,:],0),axis=2).min(1).mean(),'hypervolume':hv(sel),'true_hypervolume':hv(tf),'preference_regret':(sel@np.array(r['selection']['weights'])).min()-(yn@np.array(r['selection']['weights'])).min()}
 for key,value in vals.items():equal(metric[key],value,'Recommendation '+m+'/'+group+'/'+key)
# Independently enumerate every possible sign assignment (n<=11) and Holm family.
recalc=[]
for s in stats:
 ref=metrics[metrics.model==s['reference']].set_index('object_model');other=metrics[metrics.model==s['comparator']].set_index('object_model');d=(ref.loc[s['systems'],s['metric']]-other.loc[s['systems'],s['metric']]).to_numpy();d=d[d!=0];ranks=rankdata(np.abs(d));total=ranks.sum();plus=ranks[d>0].sum();minus=ranks[d<0].sum();w=min(plus,minus)
 possible=[sum(ranks[i] for i in range(len(d)) if signs[i]) for signs in itertools.product([False,True],repeat=len(d))];p=sum(min(x,total-x)<=w+1e-12 for x in possible)/len(possible);equal(s['p_value'],p,'Exact p '+s['comparator']+'/'+s['metric']);equal(s['rank_biserial'],(plus-minus)/total if total else 0,'RBC '+s['comparator']+'/'+s['metric']);check(s['n_nonzero']==len(d),'n_nonzero wrong');recalc.append(p)
order=np.argsort(recalc,kind='stable');adjust=np.empty(len(order));last=0.
for j,i in enumerate(order):last=max(last,(len(order)-j)*recalc[i]);adjust[i]=min(1,last)
for s,p in zip(stats,adjust):equal(s['holm_p_value'],p,'Holm '+s['comparator']+'/'+s['metric'])
# Quantities that matter for interpretation.
neg=[]
for name,p in pred.groupby('model'):
 neg.append({'model':name,**{o:int((p['mean_'+o]<0).sum()) for o in OBJ}})
r2counts=[]
for name,m in metrics.groupby('model'):r2counts.append({'model':name,**{o:int((m[o+'_r2']<0).sum()) for o in OBJ}})
ens=metrics[metrics.model.isin(['ensemble_mean','ensemble_risk'])].pivot(index='object_model',columns='model',values=['f1','hypervolume','gd','preference_regret']);delta=[]
for group in ens.index:
 vals={k:float(ens.loc[group,(k,'ensemble_risk')]-ens.loc[group,(k,'ensemble_mean')]) for k in ['f1','hypervolume','gd','preference_regret']}
 if any(v!=0 for v in vals.values()):delta.append({'object_model':group,**vals})
# Verify profiling summary aggregation against original repeated raw timings.
raw_rows=[json.loads(line) for line in (ROOT/'measurements/raw_measurements.jsonl').read_text().splitlines() if line.strip()]
raw=pd.DataFrame(raw_rows);timed=raw[~raw.warmup]
repeat_counts=timed.groupby('schema_id').size();check(len(raw)==4368 and int(raw.warmup.sum())==728,'Raw repetition/warmup count mismatch');check((repeat_counts==5).all(),'Not exactly five timed repetitions per schema')
medians=timed.groupby('schema_id')[OBJ].median()
for sid,row in medians.iterrows():
 for obj in OBJ:equal(source.loc[sid,obj],row[obj],'Raw median summary '+sid+'/'+obj)
for row in raw_rows:
 equal(row['query_ms'],np.mean(list(row['query_times_ms'].values())),'Raw query averaging '+row['schema_id'])
 equal(row['storage_mb'],row['storage_bytes']/1e6,'Raw decimal MB '+row['schema_id'])
variability={}
for obj in ['insert_ms','query_ms']:
 grouped=timed.groupby('schema_id')[obj];cv=grouped.std(ddof=1)/grouped.mean();ratios=grouped.max()/grouped.min()
 variability[obj]={'descriptive_cv_threshold':.2,'candidates_above_cv_threshold':int((cv>.2).sum()),'median_cv':float(cv.median()),'max_cv':float(cv.max()),'median_max_to_min_ratio':float(ratios.median()),'max_max_to_min_ratio':float(ratios.max())}
raw_report={'raw_rows':len(raw),'warmup_rows':int(raw.warmup.sum()),'nonwarmup_rows':len(timed),'timed_repetitions_per_schema':5,'summary_aggregation_verified':'objective-wise median of five nonwarmup records','max_absolute_summary_median_differences':(medians-source[OBJ]).abs().max().to_dict(),'max_query_aggregation_difference':max(abs(row['query_ms']-np.mean(list(row['query_times_ms'].values()))) for row in raw_rows),'max_storage_decimal_mb_difference':max(abs(row['storage_mb']-row['storage_bytes']/1e6) for row in raw_rows),'unique_realizations_per_system':raw.groupby('object_model').realization_id.nunique().to_dict(),'storage_variable_across_timed_repeats_candidate_count':int((timed.groupby('schema_id').storage_bytes.nunique()>1).sum()),'timing_variability':variability,'interpretation':'Descriptive repeat variability only. No data excluded, labels changed, hyperparameters tuned or model retraining performed.'}
for path in source_files:check(hashlib.sha256(path.read_bytes()).hexdigest()==input_hashes[path.relative_to(ROOT).as_posix()],'Input modified during audit '+str(path))
report={'status':'consistent_with_supplied_artifacts' if not issues and not diffs else 'discrepancies_found','audit_timestamp_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':input_hashes,'boolean_checks':check_count,'numeric_comparisons':numeric_comparisons,'numeric_absolute_tolerance':2e-9,'numeric_relative_tolerance':2e-9,'max_absolute_numeric_difference':max_absolute_numeric_difference,'raw_measurements':raw_report,'checks_performed':['schema and source-group split disjointness/full coverage','every schema held out exactly once per prediction model','source truth and aggregated measurement alignment','training-only objective normalizers','ensemble five-member means and sample standard deviations','calibration membership/residual quantiles/interval coverage','objective MAE/RMSE/nRMSE/R2 and uncertainty-error Spearman','predicted-risk Pareto selection identities','measured-coordinate Pareto precision/recall/F1/GD/IGD+/exact 3D HV/preference regret','all signed-rank p-values via exhaustive sign enumeration and entire Holm family','source summary medians against five nonwarmup repeats','source file hashes unchanged after analysis'],'limits':['Does not rerun training or independently authenticate the Windows database execution','Saved fold records and predictions support split/metric consistency, not proof of every unrecorded internal operation','Findings concern structural baselines and this one measured profiling scenario','Few source systems and many tied outcomes limit inference; no significance is not equivalence','Models or lambda tuned using outer-test results require independent evaluation before confirmatory improvement claims'],'issues':issues,'metric_discrepancies':diffs,'fold_count':len(audit),'oof_prediction_rows':len(pred),'calibration_prediction_rows':len(cal),'source_schema_count':len(source),'per_system_method_rows':len(metrics),'test_model_count':pred.model.nunique(),'stats_count':len(stats),'stats_min_raw_p':float(min(recalc)),'stats_min_holm_p':float(min(adjust)),'stats_significant_holm_count':sum(p<.05 for p in adjust),'ensemble_risk_differences':delta,'negative_predictions':neg,'negative_r2_system_counts':r2counts,'training_warnings':[{'fold':w['fold'],'object_model':folds[w['fold']]['test_group'],'model':w['model'],'warnings':w['warnings']} for w in warnings]}
(OUT/'audit_findings.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=lambda value:value.item()));metrics.to_csv(OUT/'metrics_readonly_copy.csv',index=False)
print(json.dumps(report,indent=2,default=lambda value:value.item()))
print('Ridge per system:');print(metrics[metrics.model=='ridge'][['object_model','test_candidates','insert_ms_r2','query_ms_r2','storage_mb_r2','f1','hypervolume','storage_mb_interval_coverage']].to_string(index=False))
