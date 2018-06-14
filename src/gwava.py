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

import sys, os
import numpy as np
import pylab as pl
from scipy import interp
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, zero_one_score, precision_recall_curve
#from sklearn.ensemble import RandomForestClassifier
from forest import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
from scipy.stats import fisher_exact, chi2_contingency, mannwhitneyu, scoreatpercentile
from pandas import *
from bisect import bisect
from scipy.stats.mstats import hmean
from math import log10

# read the GWAVA_DIR from the environment, but default to the directory above where the script is located
GWAVA_DIR = os.getenv('GWAVA_DIR', '../')

continuous_cols = ['avg_het', 'avg_gerp', 'gerp', 'tss_dist', 'ss_dist', 'GC', 'avg_daf']
cols_to_drop    = ['chr', 'start', 'end']
standard_cols   = continuous_cols + [ 
    'CDS', 'EXON','INTRON', 'UTR5', 'UTR3', 'STOP', 'START', 'DONOR', 'ACCEPTOR', 
    'cpg_island', 'in_cpg', 'repeat', 
    'seq_A', 'seq_C', 'seq_G', 'seq_T'
]

ensembl_feat_classes = {
    'histone_modifications': [
        'H3K23ac',
        'H4K5ac',
        'H2BK120ac',
        'H3K27me3',
        'H3K27me2',
        'H3K27me1',
        'H3K9ac',
        'H3K18ac',
        'H2AZ',
        'H3K36me1',
        'H3K36me3',
        'H4K20me1',
        'H4K20me3',
        'H2BK20ac',
        'H2BK15ac',
        'H3K4me3',
        'H3K4me2',
        'H3K4me1',
        'H4K12ac',
        'H3R2me1',
        'H3R2me2',
        'H3K4ac',
        'H3K9me2',
        'H3K9me1',
        'H3K9me3',
        'H4K8ac',
        'H2AK5ac',
        'H3K79me3',
        'H2BK5ac',
        'H2BK5me1',
        'H3K56ac',
        'H3K36ac',
        'H2AK9ac',
        'H3K14ac',
        'H4R3me2',
        'H4K16ac',
        'H3K27ac',
        'H2BK12ac',
        'H4K91ac',
        'H3K23me2',
        'H3K79me1',
        'H3K79me2'
    ],
    'transcription_factors': [
        'Rad21',
        'Nrsf',
        'Tr4',
        'BHLHE40',
        'SIX5',
        'BCLAF1',
        'SETDB1',
        'SP1',
        'SP2',
        'Gabp',
        'Brf2',
        'Brf1',
        'Pbx3',
        'NFKB',
        'FOXA1',
        'FOXA2',
        'Znf263',
        'CTCF',
        'BATF',
        'Brg1',
        'GTF2B',
        'Tcf12',
        'ZBTB33',
        'Gata2', 
        'Cjun',  
        'Gata1', 
        'Srf',   
        'ZZZ3',  
        'ZNF274',
        'ZEB1',  
        'TAF7',  
        'Nanog', 
        'Sirt6', 
        'TAF1',  
        'IRF4',  
        'HNF4G', 
        'Egr1',  
        'HNF4A', 
        'MEF2C', 
        'CBP',   
        'MOF',   
        'HDAC1', 
        'HDAC3', 
        'HDAC2', 
        'HDAC6', 
        'HDAC8', 
        'NELFe', 
        'PU1',   
        'Junb',  
        'Jund',  
        'USF1',  
        'FOSL1', 
        'Nfe2',  
        'FOSL2', 
        'BAF155',
        'Nrf1',  
        'THAP1', 
        'BCL11A',
        'RXRA',  
        'Pax5',  
        'Ap2alpha',
        'Cmyc',  
        'HEY1',  
        'Bdp1',  
        'PCAF',  
        'ELF1',  
        'Yy1',   
        'CTCFL', 
        'Max',   
        'MEF2A', 
        'XRCC4', 
        'ETS1',  
        'BAF170',
        'RPC155',
        'ZBTB7A',
        'NR4A1', 
        'Sin3Ak20',
        'Cfos',  
        'POU5F1',
        'Ini1',  
        'Ap2gamma',
        'E2F6',  
        'EBF',   
        'E2F4',  
        'Tip60', 
        'E2F1',  
        'BCL3',  
        'p300',  
        'POU2F2',
        'ATF3',  
        'TFIIIC-110'
    ],
    'motifs': [
        'pwm',   
        'bound_motifs'
    ],
    'polymerases': [
        'PolII', 
        'PolIII'
    ],
    'open_chromatin': [
        'FAIRE',
        'DNase1',
        'dnase_fps'
    ],
    'gene_region': [
        'UTR5',  
        'INTRON',
        'STOP',  
        'UTR3',  
        'START', 
        'EXON',  
        'CDS',   
        'DONOR', 
        'ACCEPTOR'
    ],
    'segmentations': [
        'CTCF_REG',
        'ENH',   
        'TSS_FLANK',
        'REP',   
        'TRAN',  
        'TSS',   
        'WEAK_ENH'
    ],
    'sequence_context': [
        'cpg_island',
        'in_cpg',
        'repeat'
    ]
}

encode_feat_classes = {
    'Histone': [
        'H2AFZ',
        'H3K27ac',
        'H3K27me3',
        'H3K36me3',
        'H3K4me1',
        'H3K4me2',
        'H3K4me3',
        'H3K79me2',
        'H3K9ac',
        'H3K9me1',
        'H3K9me3',
        'H4K20me1'
    ],
    'Tfbs': [
        'ATF3',
        'BATF',
        'BCL11A',
        'BCL3',
        'BCLAF1',
        'BDP1',
        'BHLHE40',
        'BRCA1',
        'BRF1',
        'BRF2',
        'CCNT2',
        'CEBPB',
        'CHD2',
        'CTBP2',
        'CTCF',
        'CTCFL',
        'E2F1',
        'E2F4',
        'E2F6',
        'EBF1',
        'EGR1',
        'ELF1',
        'ELK4',
        'EP300',
        'ERALPHAA',
        'ESRRA',
        'ETS1',
        'Eralphaa',
        'FAM48A',
        'FOS',
        'FOSL1',
        'FOSL2',
        'FOXA1',
        'FOXA2',
        'GABPA',
        'GATA1',
        'GATA2',
        'GATA3',
        'GTF2B',
        'GTF2F1',
        'GTF3C2',
        'HDAC2',
        'HDAC8',
        'HEY1',
        'HMGN3',
        'HNF4A',
        'HNF4G',
        'HSF1',
        'IRF1',
        'IRF3',
        'IRF4',
        'JUN',
        'JUNB',
        'JUND',
        'KAT2A',
        'MAFF',
        'MAFK',
        'MAX',
        'MEF2A',
        'MEF2_complex',
        'MXI1',
        'MYC',
        'NANOG',
        'NFE2',
        'NFKB1',
        'NFYA',
        'NFYB',
        'NR2C2',
        'NR3C1',
        'NR4A1',
        'NRF1',
        'PAX5',
        'PBX3',
        'POU2F2',
        'POU5F1',
        'PPARGC1A',
        'PRDM1',
        'RAD21',
        'RDBP',
        'REST',
        'RFX5',
        'RXRA',
        'SETDB1',
        'SIN3A',
        'SIRT6',
        'SIX5',
        'SLC22A2',
        'SMARCA4',
        'SMARCB1',
        'SMARCC1',
        'SMARCC2',
        'SMC3',
        'SP1',
        'SP2',
        'SPI1',
        'SREBF1',
        'SREBF2',
        'SRF',
        'STAT1',
        'STAT2',
        'STAT3',
        'SUZ12',
        'TAF1',
        'TAF7',
        'TAL1',
        'TBP',
        'TCF12',
        'TCF7L2',
        'TFAP2A',
        'TFAP2C',
        'THAP1',
        'TRIM28',
        'USF1',
        'USF2',
        'WRNIP1',
        'XRCC4',
        'YY1',
        'ZBTB33',
        'ZBTB7A',
        'ZEB1',
        'ZNF143',
        'ZNF263',
        'ZNF274',
        'ZZZ3'
    ],
    'motifs': [
        'pwm',   
        'bound_motifs'
    ],
    'polymerases': [
        'POLR2A',
        'POLR2A_elongating',
        'POLR3A'
    ],
    'open_chromatin': [
        'FAIRE',
        'DNase',
        'dnase_fps'
    ],
    'gene_region': [
        'UTR5',  
        'INTRON',
        'STOP',  
        'UTR3',  
        'START', 
        'EXON',  
        'CDS',   
        'DONOR', 
        'ACCEPTOR'
    ],
    'segmentations': [
        'CTCF_REG',
        'ENH',   
        'TSS_FLANK',
        'REP',   
        'TRAN',  
        'TSS',   
        'WEAK_ENH'
    ],
    'sequence_context': [
        'cpg_island',
        'in_cpg',
        'repeat'
    ]
}

def summarise(df, include_position=True, feat_classes=encode_feat_classes, cont_cols=continuous_cols, extra_cols=[]):
    res = {}
    for r in df.index:
        row = df.ix[r]
        row_res = {}
        for cls in feat_classes:
            row_res[cls] = []
            for ft in feat_classes[cls]:
                try:
                    if row[ft] > 0:
                        row_res[cls].append(ft)
                except:
                    print "problem with ft",ft,"in row",r
                    pass
        res[r] = row_res
    for row in res:
        for cls in res[row]:
            res[row][cls] = ','.join(res[row][cls])
    
    sm = DataFrame(index=df.index)

    if include_position:
        sm['chr'] = df.chr
        sm['start'] = df.start
        sm['end'] = df.end

    sm = sm.join(DataFrame(res).T)

    for col in cont_cols + extra_cols:
        sm[col] = df[col]

    return sm

def match_by_percentile(cases, ctrls, n, vs, chunk=10):
    starts = {}
    ps = []
    matches = np.array([])
    for v in vs:
        # store the minimum of each variable, and create our percentile columns
        starts[v] = cases[v].min() - 0.00001 # subtract a small epsilon so we don't miss any in our filters later
        cases[v+'_per'] = -1
        ctrls[v+'_per'] = -1
        ps.append(v+'_per')
    for i in np.arange(0, 100, chunk):
        # assign each instance to its corresponding percentile for each variable
        for v in vs:
            stop = scoreatpercentile(cases[v], i+chunk)
            ctrls.ix[(ctrls[v] > starts[v]) & (ctrls[v] <= stop), v+'_per'] = i+chunk
            cases.ix[(cases[v] > starts[v]) & (cases[v] <= stop), v+'_per'] = i+chunk
            starts[v] = stop
    # a lot of the controls will probably be out of range now, so limit 
    # ourselves to those with valid percentiles
    poss = ctrls.copy()
    for p in ps:
        poss = poss[poss[p] != -1]

    for (i, r) in cases.iterrows():
        # loop over each case instance and find matching control instances
        # according to the percentile ranks

        #print "Matching",i
        cond = None
        for p in ps:
            if cond is None:
                cond = poss[p] == r[p]
            else:
                cond = cond & (poss[p] == r[p])
        ms = poss[cond].index
        # exclude any instances we already have in our list of matches
        ms = ms.drop(ms.intersection(matches))
        #print "Got",len(ms)

        # pick a random n of the matches and add them to our list of matches
        if len(ms) < n:
            print "Unable to find sufficient matches for",i,"( got",len(ms),")"
        matches = np.append(matches, np.random.permutation(ms)[0:n])
    for p in ps:
        # get rid of our percentile columns
        del ctrls[p]
        del cases[p]

    # and return the subset of matches from the original table
    return ctrls.ix[matches]

def feat_imp(model, df, cls='cls'):
    idx = model.feature_importances_.argsort()   
    fis = Series(model.feature_importances_[idx[::-1]], index=df.columns.drop(cols_to_drop + [cls])[idx[::-1]].values)
    return fis

def plot_cv_roc(k, df, clf = None, cls = 'cls', do_plot=True, rs=0, curve_type=roc_curve):
    y = df[cls].values
    X = df.as_matrix(df.columns.drop(cols_to_drop + [cls]))

    if clf is None:
        clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=rs, balance_classes=True, compute_importances=True, oob_score=True)

    cv = StratifiedKFold(y, n_folds=k)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train, test) in enumerate(cv):
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = curve_type(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        if do_plot:
            pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    if do_plot:
        pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Chance')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    if do_plot:
        pl.plot(mean_fpr, mean_tpr, 'k--',
                label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
        pl.xlim([-0.05, 1.05])
        pl.ylim([-0.05, 1.05])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('10-fold cross validation')
        pl.legend(loc="lower right")
        pl.show()

    return (mean_fpr, mean_tpr, mean_auc)

def build_full_classifier(df, cls='cls', rs=0):
    y = df[cls].values
    X = df.as_matrix(df.columns.drop(cols_to_drop + [cls]))
    classifier = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=rs, balance_classes=True, compute_importances=True, oob_score=True)
    model = classifier.fit(X, y)
    return model

def get_scores(df, model, cls='cls'):
    X = df.as_matrix(df.columns.drop(cols_to_drop + [cls]))
    res = model.predict_proba(X)
    return res[:,1]

def combine_chroms(models, path, file_fmt="chr%s_%i.pandas", chroms=None, files_per_chrom=10, cols=['chr', 'start', 'end', 'score'], verbose=True, all_cols=False):
    df = DataFrame()
    if chroms is None:
        chroms = map(str, range(1,22))
        chroms.append('X')
    for c in chroms:
        for i in range(1,files_per_chrom+1):
            fname = file_fmt % (c, i)
            d = read_pickle(path+'/'+fname)
            if len(models) > 1:
                if 'score' in cols:
                    cols.remove('score')
                scores = []
                for model in models:
                    scores.append(get_scores(d, model))
                for num, s in enumerate(scores):
                    colname = 'score'+str(num)
                    d[colname] = s
                    if colname not in cols:
                        cols.append(colname)
            elif len(models) == 1:
                d['score'] = get_scores(d, models[0])

            if all_cols:
                df = df.append(d)
            else:
                df = df.append(d[cols])
            if verbose:
                print "Added",len(d),"scores from tranche",i,"of chromosome",c
    return df

def df_from_bed(fname, extra_cols=[], name_index=True):
    return read_table(fname, names=['chr', 'start', 'end'] + extra_cols, sep="\t", index_col=(3 if name_index else None))

def df_to_bed(df, fname, extra_cols=[], include_index=True, header=False, na_rep='', sort=True):
    bed_cols = ['chr', 'start', 'end']
    if not all([col in df for col in bed_cols]):
        raise Exception("DataFrame doesn't have necessary columns for BED format")
    if include_index:
        # include the index as the name field
        bed_cols.append('index')
    # remove any redundant columns from the extra cols list & add them to our list of columns
    cols = bed_cols + [e for e in extra_cols if e not in bed_cols]
    df.reset_index(inplace=True)
    if sort:
        df.sort(['chr', 'start', 'end'], inplace=True)
    df[cols].to_csv(fname, sep="\t", index=False, header=header, na_rep=na_rep)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage:",sys.argv[0],"<model> annotated.csv output.bed"
        sys.exit(0)

    model = sys.argv[1]
    annot_file = sys.argv[2]
    output_file = sys.argv[3]

    ann_dir = GWAVA_DIR+'/annotated/'

    hgmd = read_pickle(ann_dir+'hgmd_reg.pandas') 

    if model == 'region':
        control_file = 'matched_by_region.pandas'
    elif model == 'tss':
        control_file = 'matched_by_tss.pandas'
    elif model == 'unmatched':
        control_file = 'unmatched.pandas'
    else:
        print "Unrecognised model option, pick one of 'region', 'tss' or 'unmatched'"
        sys.exit(0)

    control = read_pickle(ann_dir+control_file)

    model = build_full_classifier(hgmd.append(control))

    annot = read_csv(annot_file, index_col=0)

    annot['gwava_score'] = get_scores(annot, model)
    
    df_to_bed(annot, output_file, extra_cols=['gwava_score'], sort=False)

