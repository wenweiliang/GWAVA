"""
 Copyright 2013 EMBL - European Bioinformatics Institute

 Licensed under the Apache License, Version 2.0 (the
 "License"); you may not use this file except in
 compliance with the License.  You may obtain a copy of
 the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, 
 software distributed under the License is distributed on 
 an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY 
 KIND, either express or implied. See the License for the 
 specific language governing permissions and limitations 
 under the License.
"""

from gwava import *

# script to generate results for the GWAVA paper

# read the GWAVA_DIR from the environment, but default to the directory above where the script is located
GWAVA_DIR = os.getenv('GWAVA_DIR', '../')

wd = GWAVA_DIR+'/annotated/'
rd = GWAVA_DIR+'/paper_data/'

hgmd = read_pickle(wd+'hgmd_reg.pandas')
reg = read_pickle(wd+'matched_by_region.pandas')
tss = read_pickle(wd+'matched_by_tss.pandas')
unm = read_pickle(wd+'unmatched.pandas')

# dump out the full set of combined data (for plotting in R)

hgmd['cls'] = 1
reg['cls'] = 2
tss['cls'] = 3
unm['cls'] = 4

cmb = hgmd.append(reg).append(tss).append(unm)
cmb.to_csv(rd+'all_combined.csv')

print "Printed combined annotation CSV"

# print out the ROC curve and feature importance data

datasets = {'region': reg, 'tss': tss, 'unmatched': unm}

models = {}

for ds_name in datasets.keys():
    ds = datasets[ds_name]
    ds['cls'] = 0
    ts = hgmd.append(ds)
    (fpr, tpr, auc_res) = plot_cv_roc(10, ts, do_plot=False)
    df = DataFrame.from_dict({'fpr': fpr, 'tpr': tpr, 'auc': auc_res})
    df.to_csv(rd+ds_name+'_roc.csv')

    model = build_full_classifier(ts)
    fis = feat_imp(model, ts)
    fis.to_csv(rd+ds_name+'_feat_imps.csv')

    ds.to_csv(rd+ds_name+'_annotation.csv')

    models[ds_name] = model

print "Printed classifier results & feature importances"

# score distributions

df = combine_chroms(
    [models['unmatched'], models['tss'], models['region']],
    wd,
    file_fmt="g1k_chr%s_%i.pandas",
    chroms=['16']
)

df.to_csv(rd+'chr16_score_distributions.csv')

print "Printed score distributions"

# our various validation experiments

# specific loci

tss['cls'] = 0
model = build_full_classifier(hgmd.append(tss))

for locus in ('sort1', 'tnfaip3', 'tcf7l2'):
    df = read_pickle(wd+locus+'.pandas')
    df['score'] = get_scores(df, model)
    summarise(df, extra_cols=['score']).to_csv(rd+locus+'_summary.csv')
    
print "Done locus specific analysis"

# gwas hits

gwas = read_pickle(wd+'gwas_no_overlap.pandas')
gwas_matches = read_pickle(wd+'gwas_matches.pandas')
gwas['score'] = get_scores(gwas, model)
gwas_matches['score'] = get_scores(gwas_matches, model)
#gwas.to_csv(rd+'gwas_scores.csv', cols=['score'])
gwas[['score']].to_csv(rd+'gwas_scores.csv')
#gwas_matches.to_csv(rd+'gwas_matches_scores.csv', cols=['score'])
gwas_matches[['score']].to_csv(rd+'gwas_matches_scores.csv')

print "Done GWAS analysis"

# recurrent somatic mutations

csm = read_pickle(wd+'cosmic_wgs_64.pandas')
csm_rec = df_from_bed(wd+'cosmic_rec_wgs_64.bed', extra_cols=['rec'])
csm = csm[csm.CDS==0]
csm['score'] = get_scores(csm, model)
csm['rec'] = csm_rec.rec
csm['cls'] = map(lambda x: 1 if x > 1 else 0, csm.rec)
#csm.to_csv(rd+'cosmic_recurrent_scores.csv', cols=['rec', 'cls', 'score'])
csm[['rec', 'cls', 'score']].to_csv(rd+'cosmic_recurrent_scores.csv')

print "Done somatic mutation analysis"

# individual analysis

na = read_pickle(wd+'NA06984_chr22.pandas')
hgmd_chr22 = hgmd[hgmd.chr == 'chr22']
hgmd_rest = hgmd[hgmd.chr != 'chr22']
tss_rest = tss[tss.chr != 'chr22']
tss_rest['cls'] = 0
na_model = build_full_classifier(hgmd_rest.append(tss_rest))
na['score'] = get_scores(na, na_model)
hgmd_chr22['score'] = get_scores(hgmd_chr22, na_model)
na['cls'] = 0
comb = na.append(hgmd_chr22)
#comb.to_csv(rd+'ind_scores.csv', cols=['cls', 'score'])
comb[['cls', 'score']].to_csv(rd+'ind_scores.csv')

print "Done overall individual analysis"

# gene by gene analysis

def permute_gene_test(gene_nums, hits, top3s, perms=100000):
    """This subroutine computes the empirical p values for the 
    gene by gene analysis on the individual genome"""
    hits_matches = 0
    top3_matches = 0
    for i in xrange(perms):
        num_hits = 0
        num_top3 = 0
        for g in gene_nums:
            s = np.random.choice(g, size=3, replace=False)
            if s[0] == 0:
                num_hits += 1
            if 0 in s:
                num_top3 += 1

        if num_hits >= hits:
            hits_matches += 1

        if num_top3 >= top3s:
            top3_matches += 1

    p_hits = float(hits_matches) / perms
    p_top3 = float(top3_matches) / perms

    return p_hits, p_top3

hgmd_genes = read_table(wd+'chr22_hgmd_genes.tsv', index_col=0)
hits = 0
top3s = 0
flank = 5000
gene_results = {}

for v in hgmd_genes.index:
    h = hgmd_genes.ix[v]
    controls = na[(na.end <= h['end'] + flank) & (na.start >= h['start'] - flank) & (na.CDS == 0) & (na.DONOR == 0) & (na.ACCEPTOR == 0)]
    h_score = hgmd_chr22.ix[v]['score']
    rank = len(controls[controls.score > h_score]) + 1

    if rank == 1:
        hits += 1

    if rank <= 3:
        top3s += 1

    gene_results[h['name']] = {
        'n_controls': len(controls),
        'rank': rank,
        'region_length': h['end'] - h['start'] + 1 + (flank * 2),
        'hgmd_variant': v
    }

gene_df = DataFrame(gene_results).T

(p_hits, p_top3) = permute_gene_test(gene_df.n_controls + 1, hits, top3s, perms=1000000) 

gene_df.index.name = "gene"
gene_df.to_csv(rd+'supplementary_table_5.csv')

print "Done gene by gene individual analysis: p_hits =",p_hits,"p_top3 =",p_top3

print "SCRIPT COMPLETED SUCCESSFULLY"

