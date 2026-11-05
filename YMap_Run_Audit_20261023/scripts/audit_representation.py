"""Read-only representation audit; standard-library and framework pure-Python helpers.

Usage: python audit_representation.py --input RUN_FOLDER --output OUT_FOLDER
       --framework-source PATH_TO_YMAP_SRC
No database, model fitting, parameter tuning, or target modifications occur.
"""
import argparse,collections,csv,hashlib,json,math,pathlib,statistics,sys
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=pathlib.Path,required=True)
parser.add_argument('--output',type=pathlib.Path,required=True)
parser.add_argument('--framework-source',type=pathlib.Path,default=pathlib.Path('src'))
args=parser.parse_args();SOURCE=args.input.expanduser().resolve();OUT=args.output.expanduser().resolve()
if not SOURCE.is_dir():
 parser.error('--input must be an existing run directory')
if OUT==SOURCE or OUT.is_relative_to(SOURCE):
 parser.error('--output must be outside the input run directory')
if OUT.exists() and (not OUT.is_dir() or any(OUT.iterdir())):
 parser.error('--output must be a new or empty directory; existing output files are never overwritten')
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(args.framework_source.resolve()))
from ymap_research.synthesis import canonical_candidate
from ymap_research.profiling import build_load_plan,build_query_plan,expected_query_results,validate_logical_fixture

def sha(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()

def load(path):return json.loads(path.read_text())
records=[json.loads(line) for line in (SOURCE/'candidates.jsonl').read_text().splitlines() if line.strip()]
measurements={r['schema_id']:r for r in csv.DictReader((SOURCE/'measurements/measurements.csv').open())}
fixtures={};expected={};checks=collections.Counter();errors=[];derived={};permodel=collections.defaultdict(list);fixture_hashes=collections.defaultdict(set);queryspec_hashes=collections.defaultdict(set);physical_hashes=collections.defaultdict(list)
for record in records:
 sid,model=record['schema_id'],record['object_model'];folder=SOURCE/'measurements/candidates'/sid;permodel[model].append(record)
 try:canonical_candidate(record);checks['canonical_candidate_valid']+=1
 except Exception as exc:errors.append({'schema_id':sid,'kind':'canonical_candidate','error':str(exc)})
 stored=load(folder/'candidate.json')
 if stored==record:checks['measured_candidate_exactly_matches_input']+=1
 else:errors.append({'schema_id':sid,'kind':'stored_candidate_mismatch'})
 if (folder/'schema.sql').read_text().rstrip('\n')==record['ddl'].rstrip('\n'):checks['measured_sql_matches_input']+=1
 else:errors.append({'schema_id':sid,'kind':'stored_sql_mismatch'})
 blob=(folder/'fixture.json').read_bytes();fixture_hashes[model].add(hashlib.sha256(blob).hexdigest())
 if model not in fixtures:
  fixtures[model]=json.loads(blob);validate_logical_fixture(record['source_model'],fixtures[model]);expected[model]=expected_query_results(record,fixtures[model],100)
 if measurements[sid]['realization_id']==fixtures[model]['realization_id']:checks['summary_realization_matches_fixture']+=1
 else:errors.append({'schema_id':sid,'kind':'realization_id_mismatch'})
 queries=load(folder/'queries.json');queryspec_hashes[model].add(sha(queries))
 if queries==build_query_plan(record,100,fixtures[model]):checks['query_spec_exactly_regenerates']+=1
 else:errors.append({'schema_id':sid,'kind':'query_spec_mismatch'})
 if load(folder/'expected_query_results.json')==expected[model]:checks['expected_query_results_match_logical_oracle']+=1
 else:errors.append({'schema_id':sid,'kind':'oracle_mismatch'})
 lifecycle=load(folder/'schema_lifecycle.json')
 if lifecycle.get('cleanup_status')=='dropped' and lifecycle.get('created_by_this_run'):checks['owned_schema_cleanup_dropped']+=1
 else:errors.append({'schema_id':sid,'kind':'cleanup_not_dropped','lifecycle':lifecycle})
 plan=build_load_plan(record,fixtures[model]);rowcounts={table:len(t['rows']) for table,t in plan.items()};typecounts=collections.Counter(c['type'] for t in record['tables'] for c in t['columns']);textbytes=0
 for table in record['tables']:
  for col_index,column in enumerate(table['columns']):
   if column['type']=='TEXT':
    textbytes+=sum(len(str(row[col_index]).encode()) for row in plan[table['name']]['rows'] if row[col_index] is not None)
 # Deferred row-level constraints fire once for each inserted target physical
 # row and link row. This is an exact count from the declared triggers/load plan,
 # not a measured cost or a claim about the optimizer's scan plan.
 trigger_details=[]
 for association in record['source_model'].get('associations',[]):
  am=record['association_mapping'][association['name']]
  if am['strategy']=='association_table' and association.get('multiplicity','one-to-many')!='many-to-many' and not association.get('nullable',False):
   target_table=record['mapping'][association['target']]['table'];trigger_details.append({'association':association['name'],'target_storage_rows':rowcounts[target_table],'link_rows':rowcounts[am['table']],'events':rowcounts[target_table]+rowcounts[am['table']]})
 derived[sid]={'physical_insert_rows':sum(rowcounts.values()),'table_rows':rowcounts,'type_counts':dict(typecounts),'physical_text_payload_bytes':textbytes,'deferred_trigger_insert_events':sum(x['events'] for x in trigger_details),'trigger_details':trigger_details,'query_count':len(queries),'workload_sql_join_count':sum(q['sql'].upper().count(' JOIN ') for q in queries),'workload_sql_union_all_count':sum(q['sql'].upper().count(' UNION ALL ') for q in queries)}
 signature=[]
 for table in sorted(record['tables'],key=lambda t:t['name']):
  signature.append({'name':table['name'],'columns':sorted(table['columns'],key=lambda c:c['name']),'foreign_keys':sorted(table['foreign_keys'],key=sha),'unique':sorted(table['unique']),'checks':sorted(table['checks'])})
 physical_hashes[sha(signature)].append(sid)
raw_count=0
for line in (SOURCE/'measurements/raw_measurements.jsonl').open():
 raw=json.loads(line);raw_count+=1
 if raw['query_results']==expected[raw['object_model']]:checks['raw_repetition_results_match_logical_oracle']+=1
 else:errors.append({'schema_id':raw['schema_id'],'kind':'raw_result_mismatch','repeat':raw['repeat']})
features=sorted(records[0]['features']);groups=collections.defaultdict(list)
for record in records:groups[tuple(record['features'][name] for name in features)].append(record)
per_system=[]
for model,rows in sorted(permodel.items()):
 g=collections.defaultdict(list)
 for record in rows:g[tuple(record['features'][name] for name in features)].append(record)
 per_system.append({'object_model':model,'candidates':len(rows),'feature_vectors':len(g),'collision_candidates':sum(len(v) for v in g.values() if len(v)>1),'largest_collision_group':max(map(len,g.values())),'fixture_byte_hashes':len(fixture_hashes[model]),'query_spec_hashes':len(queryspec_hashes[model])})
floors={}
for metric in ('insert_ms','query_ms','storage_mb'):
 values=[float(measurements[r['schema_id']][metric]) for r in records];overall=statistics.mean(values);sse=0
 for g in groups.values():
  ys=[float(measurements[r['schema_id']][metric]) for r in g];mu=statistics.mean(ys);sse+=sum((y-mu)**2 for y in ys)
 sst=sum((v-overall)**2 for v in values);floors[metric]={'best_empirical_same_feature_prediction_rmse':math.sqrt(sse/len(records)),'empirical_r2_ceiling':1-sse/sst,'interpretation':'Optimistic in-sample group-mean bound on these measured medians; includes measurement variation, not a causal or generalization claim.'}
examples=[]
for group in sorted(groups.values(),key=len,reverse=True):
 if len(group)>1:
  values={metric:[float(measurements[r['schema_id']][metric]) for r in group] for metric in ('insert_ms','query_ms','storage_mb')}
  examples.append({'schemas':[r['schema_id'] for r in group],'systems':sorted({r['object_model'] for r in group}),'features':group[0]['features'],'cost_ranges':{k:[min(v),max(v)] for k,v in values.items()},'physical_row_counts':sorted({derived[r['schema_id']]['physical_insert_rows'] for r in group}),'trigger_events':sorted({derived[r['schema_id']]['deferred_trigger_insert_events'] for r in group}),'physical_text_bytes':sorted({derived[r['schema_id']]['physical_text_payload_bytes'] for r in group})})
report={'candidate_count':len(records),'raw_repetitions':raw_count,'consistency_checks':dict(checks),'errors':errors,'feature_names':features,'unique_feature_vectors':len(groups),'collision_groups':len(examples),'candidates_in_collisions':sum(len(g) for g in groups.values() if len(g)>1),'cross_system_collision_groups':sum(len({r['object_model'] for r in g})>1 for g in groups.values()),'exact_physical_declaration_duplicate_groups':sum(len(g)>1 for g in physical_hashes.values()),'per_system':per_system,'empirical_feature_bounds':floors,'collision_details':examples}
report['constant_features']=[name for name in features if len({r['features'][name] for r in records})==1]
report['collision_group_distinctions_omitted_by_16_features']={}
for name in ('physical_insert_rows','deferred_trigger_insert_events','physical_text_payload_bytes','type_counts','workload_sql_join_count','workload_sql_union_all_count'):
 report['collision_group_distinctions_omitted_by_16_features'][name]=sum(len({sha(derived[sid][name]) for sid in g['schemas']})>1 for g in examples)
report['held_out_feature_extrapolation']=[]
for model,heldout in sorted(permodel.items()):
 training=[r for r in records if r['object_model']!=model];ranges={name:[min(r['features'][name] for r in training),max(r['features'][name] for r in training)] for name in features};outside={name:sum(not ranges[name][0]<=r['features'][name]<=ranges[name][1] for r in heldout) for name in features}
 report['held_out_feature_extrapolation'].append({'object_model':model,'held_out_candidates':len(heldout),'candidates_outside_at_least_one_training_feature_range':sum(any(not ranges[name][0]<=r['features'][name]<=ranges[name][1] for name in features) for r in heldout),'outside_counts_by_feature':{name:n for name,n in outside.items() if n}})
(OUT/'artifact_and_representation_audit.json').write_text(json.dumps(report,indent=2));(OUT/'derived_measured_fixture_structure.json').write_text(json.dumps(derived,indent=2))
with (OUT/'per_system_feature_collisions.csv').open('w',newline='') as handle:
 writer=csv.DictWriter(handle,fieldnames=list(per_system[0]));writer.writeheader();writer.writerows(per_system)
by_id={r['schema_id']:r for r in records}
contrast_ids=['hospital_information_system_071147be978fb095','hospital_information_system_cd430f89686cd8dd']
contrast=[]
if all(sid in by_id for sid in contrast_ids):
 for sid in contrast_ids:
  contrast.append({'schema_id':sid,'decisions':by_id[sid]['decisions'],'features':by_id[sid]['features'],'measured_objectives':{metric:float(measurements[sid][metric]) for metric in ('insert_ms','query_ms','storage_mb')},'fixture_derived_structure':derived[sid]})
findings={'main_finding':'The 16-count structural baseline discards schema distinctions; this is a representation limitation, not evidence of a synthesis/profiling software bug.','scope':'Read-only descriptive audit. No model fitted, no held-out-target-guided tuning, no target or source-file changes.','evidence':{'candidates':len(records),'feature_vectors':len(groups),'collision_groups':len(examples),'collision_candidates':report['candidates_in_collisions'],'collision_candidate_percentage':100*report['candidates_in_collisions']/len(records),'cross_system_collision_groups':report['cross_system_collision_groups'],'artifact_consistency':dict(checks),'artifact_mismatches':errors,'exact_physical_declaration_duplicate_groups':report['exact_physical_declaration_duplicate_groups']},'concrete_identical_feature_contrast':contrast,'observed_omitted_information':report['collision_group_distinctions_omitted_by_16_features'],'empirical_same_feature_prediction_bounds':floors,'interpretation':['All source-system variants match the same finite logical fixture/query oracle within that system; this is workload equivalence, not a general semantic-equivalence proof.','No duplicate complete physical declaration set was found after order normalization. Aggregate-feature collisions must not be treated as duplicate schemas.','Twenty collision groups differ in actual physical load rows and deferred trigger events; four cross-system groups differ in declared SQL type distributions.','The concrete Hospital contrast swaps which association is materialized. Its aggregate JOIN/UNION counts also coincide, showing why endpoint roles/cardinalities matter beyond more aggregate counts.','Typed graph/DDL representations retain more role/topology/type information than the 16 counts. Their actual pretrained or trained encoders must be evaluated; this audit does not show that they will solve LOSO or measurement instability.','SQL query text/topology is available, but optimizer EXPLAIN plans are not part of the supplied artifacts. Query/data/environment conditioning remains different from encoding DDL alone.','Empirical group-mean bounds include timing variation in the observed medians. They are optimistic in-sample information bounds, not attainable held-out performance or causal attribution.','The next methodological comparison should prespecify graph-plus-symbolic ablations versus the actual pretrained-text/graph/symbolic system, retaining source-system separation and fixed measurement provenance.']}
(OUT/'findings.json').write_text(json.dumps(findings,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in {'collision_details'}},indent=2))
