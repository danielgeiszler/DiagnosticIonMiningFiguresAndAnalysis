import re
import sys

import numpy as np
import seaborn
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.pyplot import figure
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as plt3d
from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib_venn import venn2, venn2_circles
import glob

import scipy
from statsmodels.nonparametric.smoothers_lowess import lowess

aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
cm = 1/2.54

def makePtptmsFigures():
    # makeVennDiagram()
    # makeBoxPlots()
    # makeCombinedBoxPlot()
    # makeBYScatter()
    print()


def makeAdprFigures():
    #makeNewAdprBoxplots()
    makeADPRPurity136()
    #makeADPRUnmodAvgIntensityCorr()


def makeCysProbeFigures():
    makeCysProbeScatter()


def makeRNAXLinkFigures():
    # makeRNAXLinkHistos()
    # makeRNAXBarChart()
    makeRNAXLinkBaeHistos()
    # makeRNAXLinkBaeBarChart()


def makeLuadFigures():
    # makeIonRatioScatter()
    makeLocalizationHistos()
    # makeLocalizationBarCharts() # deprecated
    makeFdrFilteredLocalizationBarChart()
    print()


def makeGlycoFigures():
    #makeCCRCCHistos()
    #makeCCRCCDiagnosticHeatMap()
    makeCCRCCUnfilteredGlycoPeptideIonsScatter()
    makeCCRCCUnfilteredGlycoDiagnosticIonsScatter()
    # makeCCRCCPhosphoGlycoRemainderMasses()
    """unused"""
    # makeCCRCCPhosphoGlycoPurity()
    # makeCCRCCYIonImpovementsBarChart()


def makeCCRCCGlycoFigures():
    makeGlycoDiagnosticFeatureScatters()


def makeMouseKidneyGlycoFigures():
    makeMouseKidneyGlycoDiagnosticIons2DScatter()
    makeMouseKidneyGlycoYIons2DScatter()


def make136n137Scatter():
    diagData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\final_data\\136n137\\dataset01.diagnosticIons.tsv",
        sep='\t')

    diagData = diagData[diagData['Mass Shift'] > 50]
    diagData = diagData[diagData['ox_137.0463_intensity'] > 0.01]

    shift541 = diagData[diagData['Mass Shift'] > 540.5]
    shift541 = shift541[shift541['Mass Shift'] < 541.5]
    shift542 = diagData[diagData['Mass Shift'] > 541.5]
    shift542 = shift542[shift542['Mass Shift'] < 542.5]
    shift543 = diagData[diagData['Mass Shift'] > 542.5]
    shift543 = shift543[shift543['Mass Shift'] < 543.5]

    print(shift541['Mass Shift'])

    res = linear_model.LinearRegression()
    res.fit(shift541['Mass Shift'].values.reshape(-1, 1), shift541['ox_137.0463_intensity'].values.reshape(-1, 1))
    print(res.coef_)
    print(shift541['ox_137.0463_intensity'].mean(), shift541['ox_137.0463_intensity'].median())
    res.fit(shift542['Mass Shift'].values.reshape(-1, 1), shift542['ox_137.0463_intensity'].values.reshape(-1, 1))
    print(res.coef_)
    print(shift542['ox_137.0463_intensity'].mean(), shift542['ox_137.0463_intensity'].median())
    res.fit(shift543['Mass Shift'].values.reshape(-1, 1), shift543['ox_137.0463_intensity'].values.reshape(-1, 1))
    print(res.coef_)
    print(shift543['ox_137.0463_intensity'].mean(), shift543['ox_137.0463_intensity'].median())

    # diagCols = [col for col in diagData.columns if 'ox_' in col]
    sns.regplot(data=diagData, x='Mass Shift', y='ox_137.0463_intensity')
    plt.savefig('ADPR_newTools/final_ADPR/136n137/test.png')


def makeCCRCCPhosphoGlycoRemainderMasses():
    ions = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\CCRCCPhosphoGlyco\\remainderAndYIons\\combined.diagnosticIons.tsv",
        sep="\t")
    ions = ions[ions['Mass Shift'] > 50]
    print(ions)

    remMassCols = ['Y_83.0371_intensity', 'Y_203.0794_intensity']

    remData = pd.melt(ions, id_vars=['Spectrum'], value_vars=remMassCols)
    print(remData)

    sns.boxplot(data=remData, x='value', hue='variable')
    # sns.boxplot(data=ions,x='Y_203.0794_intensity')

    plt.savefig('CCRCCPhosphoGlyco/remainderAndYIons/boxPlots.png')
    plt.clf()


def makeADPRPurityRidge():
    psms_hela = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\final_data\\136_extraction\\psm.tsv_HeLa",
        sep="\t")
    ions_hela = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\final_data\\136_extraction\\HeLa.diagnosticIons.tsv",
        sep="\t")

    dfHela = psms_hela.merge(ions_hela, on='Spectrum')
    dfHela['bins'] = dfHela.apply(multiBinMarkerHistogramBins, axis=1)

    g = sns.FacetGrid(dfHela, row='bins', aspect=9, height=1.2)
    g.set_titles("")
    g.set(yticks=[])
    g.despine(left=True)
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g.map_dataframe(sns.histplot, x='ox_136.0623_intensity', fill=True, alpha=1)
    g.fig.subplots_adjust(hspace=-.5)

    plt.savefig('ADPR_newTools\\final_ADPR\\136_purity\\136_purity_ridge.png')
    plt.clf()


def makeADPRUnmodAvgIntensityCorr():
    mions = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\final_data\\Mouse\\mouse_global_diagmine.txt",
        sep="\t")
    hions = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\final_data\\HeLa\\hela_global.diagmine.txt",
        sep="\t")

    mions = mions[mions['peak_apex'] > 541.0607]
    mions = mions[mions['peak_apex'] < 541.0609]
    mions = mions[mions['ion_type'] == 'diagnostic']
    hions = hions[hions['peak_apex'] > 541.0615]
    hions = hions[hions['peak_apex'] < 541.0617]
    hions = hions[hions['ion_type'] == 'diagnostic']


    mions['avg_intensity_mod'] = np.log10(mions['avg_intensity_mod'])
    hions['avg_intensity_mod'] = np.log10(hions['avg_intensity_mod'])
    print(len(mions['avg_intensity_mod']))
    print(len(hions['avg_intensity_mod']))
    mions['percent_unmod'] = np.log10(mions['percent_unmod'])
    hions['percent_unmod'] = np.log10(hions['percent_unmod'])
    mions.to_csv("mouse_purity.tsv", sep='\t')
    hions.to_csv("hela_purity.tsv", sep='\t')
    mions.dropna(axis=1, inplace=True)
    hions.dropna(axis=1, inplace=True)

    res = linear_model.LinearRegression()
    res.fit(mions['avg_intensity_mod'].values.reshape(-1, 1), mions['percent_unmod'].values.reshape(-1, 1))
    print(res.score(mions['avg_intensity_mod'].values.reshape(-1, 1), mions['percent_unmod'].values.reshape(-1, 1)))
    res = linear_model.LinearRegression()
    res.fit(hions['avg_intensity_mod'].values.reshape(-1, 1), hions['percent_unmod'].values.reshape(-1, 1))
    print(res.score(hions['avg_intensity_mod'].values.reshape(-1, 1), hions['percent_unmod'].values.reshape(-1, 1)))

    plt.figure(figsize=(6*cm, 6*cm))

    sns.regplot(data=hions, x='avg_intensity_mod', y='percent_unmod', fit_reg=True,
                color='#7DC242',  scatter_kws={'s':2}, line_kws={'linewidth':1})
    sns.regplot(data=mions, x='avg_intensity_mod', y='percent_unmod', fit_reg=True,
                color='#72CDDF',  scatter_kws={'s':2}, line_kws={'linewidth':1})
    sns.despine()
    #plt.tight_layout(pad=0)
    plt.ylim(-0.5, 2)
    plt.xlim(-0.5, 2)

    patches = [matplotlib.patches.Patch(color=i, label=j) for i, j in zip(['#7DC242', '#72CDDF'], ['HeLa', 'mouse'])]
    plt.legend(handles=patches)

    plt.xlabel('log(avg ion intensity in mod spectra)',fontsize=6)
    plt.ylabel('log(pct unmod spectra with ion)',fontsize=6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout(pad=0)
    plt.savefig('ADPR_newTools\\final_ADPR\\136_purity\\intensity_unmod_corr.pdf')


def makeADPRPurity136():
    psms_mouse = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\final_data\\136_extraction\\psm.tsv_Mouse",
        sep="\t")
    psms_hela = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\final_data\\136_extraction\\psm.tsv_HeLa",
        sep="\t")
    ions_mouse = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\final_data\\136_extraction\\Mouse.diagnosticIons.tsv",
        sep="\t")
    ions_hela = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\final_data\\136_extraction\\HeLa.diagnosticIons.tsv",
        sep="\t")

    dfHela = psms_hela.merge(ions_hela, on='Spectrum')
    dfMouse = psms_mouse.merge(ions_mouse, on='Spectrum')

    dfHela = dfHela[~dfHela['Assigned Modifications'].str.contains('541', na=False)]
    print(len(dfMouse))
    dfMouse = dfMouse[~dfMouse['Assigned Modifications'].str.contains('541', na=False)]
    print(len(dfMouse))

    dfHela = dfHela[dfHela['Mass Shift'] < 0.5]
    dfHela = dfHela[dfHela['Mass Shift'] > -0.5]
    dfMouse = dfMouse[dfMouse['Mass Shift'] > -0.5]
    dfMouse = dfMouse[dfMouse['Mass Shift'] < 0.5]

    dfHela.to_csv("hela_purity.tsv", sep='\t')
    dfMouse.to_csv("mouse_purity.tsv", sep='\t')

    dfHelaPure = dfHela[dfHela['Purity'] == 1.0]
    dfHelaPure.to_csv('pureHelaSpectra.tsv', sep='\t')

    cols = ['ox_136.0623_intensity']

    dfHela['bins'] = dfHela.apply(multiBinMarkerHistogramBins, axis=1)
    dfMouse['bins'] = dfMouse.apply(multiBinMarkerHistogramBins, axis=1)
    dfHela[cols[0]] = np.log10(dfHela[cols[0]])
    print(len(dfHela))
    dfHela = dfHela[np.isfinite(dfHela[cols[0]])]
    print(len(dfHela))
    dfMouse[cols[0]] = np.log10(dfMouse[cols[0]])
    dfMouse = dfMouse[np.isfinite(dfMouse[cols[0]])]

    hMeds = dfHela.groupby('bins')[cols[0]].median().to_frame()
    mMeds = dfMouse.groupby('bins')[cols[0]].median().to_frame()
    hMeds['136.06 ion intensity'] = hMeds['ox_136.0623_intensity']
    mMeds['136.06 ion intensity'] = mMeds['ox_136.0623_intensity']

    #drop final row due to technical artifact at urity = 1
    #hMeds = hMeds.iloc[:-1]
    #mMeds = mMeds.iloc[:-1]
    print(hMeds)
    print(mMeds)
    #print(hMeds)
    #print(mMeds)
    #hxs = list(hMeds.index[:-1])
    #hys = list(hMeds[:-1])
    #print(hMeds.index)
    #print(hMeds[0:])
    #print(hMeds[0], hMeds[1])
    #print(len(hxs), len(hys))

    #mxs = list(mMeds.index[:-1])
    #mys = list(mMeds[:-1])
    sns.regplot(data=hMeds, y='ox_136.0623_intensity', x=hMeds.index, x_estimator=np.mean, ci=None, fit_reg=True,
                color='#7DC242', order=3, scatter_kws={'s':7})
    #sns.lmplot(data=hMeds, y='ox_136.0623_intensity', x=hMeds.index, lowess=True)
    sns.regplot(data=mMeds, y='ox_136.0623_intensity', x=mMeds.index, x_estimator=np.mean, ci=None, fit_reg=True,
                color='#72CDDF', order=3, scatter_kws={'s':7})

    #sns.scatterplot(x=hxs, y=hys)
    #sns.scatterplot(x=mxs, y=mys)
    #sns.regplot(data=dfHela, x='Purity', y=cols[0], x_estimator=np.mean,x_bins=1000,ci=None, fit_reg=True, color='#7DC242')
    #sns.regplot(data=dfMouse, x='Purity', y=cols[0], x_estimator=np.mean, x_bins=1000, ci=None, fit_reg=True, color='#72CDDF')

    sns.despine()
    plt.ylim(0, 2)
    plt.title('ADPR ions in unmod spectra')
    plt.ylabel('log(136.06 ion intensity)')
    plt.xlabel('unmod MS1 purity')

    plt.savefig('ADPR_newTools\\final_ADPR\\136_purity\\136_purity_medianbinned_polyfit.png')
    plt.savefig('ADPR_newTools\\final_ADPR\\136_purity\\136_purity_medianbinned_polyfit.tiff')
    plt.savefig('ADPR_newTools\\final_ADPR\\136_purity\\136_purity_medianbinned_polyfit.pdf')
    plt.clf()

    sns.histplot(data=dfHela, x='Purity')
    plt.savefig('ADPR_newTools\\final_ADPR\\136_purity\\136_hela_purity_hit.png')
    plt.clf()


def makeRNAXLinkBaeBarChart():
    condition = ['unmod', '4SU', 'unmod', '4SU']
    psms = [12376, 12848]
    search = ['no -17', 'no -17', 'with -17', 'with -17']
    d = {'condition': condition, 'psms': psms, 'search': search}
    tinydf = pd.DataFrame(d)

    sns.barplot(data=tinydf, x='search', y='psms', hue='condition', palette={'unmod': 'darkblue', '4SU': 'darkred'})
    sns.despine()

    plt.tight_layout()
    plt.savefig('RNAXLink_Bae/psms_barchart.png')
    plt.clf()


def makeRNAXLinkHistos():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\RNAXLink\\analysis_newTools\\global.diagmine.tsv",
        sep='\t')

    peakData = peakData[peakData['peak_apex'] > 100]
    diagData = peakData[peakData['ion_type'] == 'imm']

    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    ax = sns.histplot(x='adjusted_mass', data=diagData, binwidth=0.01, binrange=(100, 800), log_scale=(False, False),
                      linewidth=1, rasterized=True, color='darkcyan', edgecolor='darkcyan')
    plt.xlim(100, 800)
    plt.ylim(000, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 3:
            print(round(x, 2))
            ax.text(x + w / 2, h + 1, str(round(x, 2)), ha='center', va='center', size=8, rotation='vertical')
    sns.despine()
    plt.savefig('RNAXLink/diagnostic_ions_histo.png')
    plt.savefig('RNAXLink/diagnostic_ions_histo.pdf')
    plt.clf()

    print('making histo zoom')
    ax = sns.histplot(x='adjusted_mass', data=diagData, binwidth=0.01, binrange=(100, 400), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(100, 400)
    plt.ylim(0, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 3:
            print(round(x, 2))
            ax.text(x + w / 2, h + 1, str(round(x, 2)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('RNAXLink/diagnostic_ions_histo_zoom.png')
    plt.clf()

    diagData = peakData[peakData['ion_type'] == 'Y']

    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    ax = sns.histplot(x='adjusted_mass', data=diagData, binwidth=0.01, binrange=(-10, 510), log_scale=(False, False),
                      linewidth=1, rasterized=True, edgecolor='darkcyan')
    plt.xlim(-10, 510)
    plt.ylim(000, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 3:
            print(round(x, 2))
            ax.text(x + w / 2, h + 1, str(round(x, 2)), ha='center', va='center', size=8, rotation='vertical')
    sns.despine()
    plt.savefig('RNAXLink/capY_ions_histo.png')
    plt.savefig('RNAXLink/capY_ions_histo.pdf')
    plt.clf()

    print('making histo zoom')
    """
    ax = sns.histplot(x='adjusted_mass', data=peakData, binwidth=0.01, binrange=(100, 400), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(100, 400)
    plt.ylim(0, 1000)
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 1:
            print(round(x, 2))
            ax.text(x + w / 2, h + 50, str(round(x, 2)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('MouseGlyco/diagnostic_ions_histo_zoom.png')
    plt.clf()
    """

    diagData = peakData[peakData['ion_type'] == 'b']

    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    ax = sns.histplot(x='adjusted_mass', data=diagData, binwidth=1, binrange=(-50, 1000), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(-50, 1000)
    plt.ylim(000, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 3:
            print(round(x, 2))
            ax.text(x + w / 2, h + 1, str(round(x)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('RNAXLink/b_ions_histo.png')
    plt.clf()

    diagData = peakData[peakData['ion_type'] == 'y']

    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    ax = sns.histplot(x='adjusted_mass', data=diagData, binwidth=1, binrange=(-50, 1000), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(-50, 1000)
    plt.ylim(000, 500)
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 5:
            print(round(x, 2))
            ax.text(x + w / 2, h + 5, str(round(x)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('RNAXLink/y_ions_histo.png')
    plt.clf()


def makeMouseKidneyHistos():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\MouseGlyco\\fixed_isotopes\\global.diagmine.tsv",
        sep='\t')

    diagData = peakData[peakData['ion_type'] == 'diagnostic']

    """
    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    ax = sns.histplot(x='adjusted_mass', data=diagData, binwidth=0.01, binrange=(100, 800), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(100, 800)
    plt.ylim(000, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 5:
            print(round(x, 2))
            ax.text(x + w / 2, h + 5, str(round(x,2)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('MouseGlyco/diagnostic_ions_histo.png')
    plt.clf()

    print('making histo zoom')
    ax = sns.histplot(x='adjusted_mass', data=diagData, binwidth=0.01, binrange=(100, 400), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(100, 400)
    plt.ylim(0, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 5:
            print(round(x, 2))
            ax.text(x + w / 2, h + 5, str(round(x,2)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('MouseGlyco/diagnostic_ions_histo_zoom.png')
    plt.clf()


    diagData = peakData[peakData['ion_type'] == 'Y']

    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    ax = sns.histplot(x='adjusted_mass', data=diagData, binwidth=0.01, binrange=(0, 2000), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(0, 2000)
    plt.ylim(-10, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 5:
            print(round(x, 2))
            ax.text(x + w / 2, h + 5, str(round(x,2)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('MouseGlyco/capY_ions_histo.png')
    plt.clf()

    print('making histo zoom')
    """
    """
    ax = sns.histplot(x='adjusted_mass', data=peakData, binwidth=0.01, binrange=(100, 400), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(100, 400)
    plt.ylim(0, 1000)
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 1:
            print(round(x, 2))
            ax.text(x + w / 2, h + 50, str(round(x, 2)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('MouseGlyco/diagnostic_ions_histo_zoom.png')
    plt.clf()
    """

    diagData = peakData[peakData['ion_type'] == 'b']

    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    ax = sns.histplot(x='adjusted_mass', data=diagData, binwidth=0.01, binrange=(-50, 500), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(-50, 1000)
    plt.ylim(000, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 5:
            print(round(x, 2))
            ax.text(x + w / 2, h + 5, str(round(x, 2)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('MouseGlyco/b_ions_histo.png')
    plt.clf()

    diagData = peakData[peakData['ion_type'] == 'y']

    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    ax = sns.histplot(x='adjusted_mass', data=diagData, binwidth=0.01, binrange=(-50, 1000), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(-50, 500)
    plt.ylim(000, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 5:
            print(round(x, 2))
            ax.text(x + w / 2, h + 5, str(round(x, 2)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('MouseGlyco/y_ions_histo.png')
    plt.clf()


def makeCCRCCYIonImpovementsBarChart():
    condition = ['normal', 'normal', 'glyco', 'glyco']
    psms = [143533, 146331, 47006, 48823]
    search = ['old', 'new', 'old', 'new']
    d = {'condition': condition, 'psms': psms, 'search': search}
    tinydf = pd.DataFrame(d)

    sns.barplot(data=tinydf, x='condition', y='psms', hue='search', palette={'old': 'darkblue', 'new': 'darkgreen'})
    sns.despine()

    plt.tight_layout()
    plt.savefig('CCRCCPhosphoGlyco/Y_ion_improvement_barchart.png')
    plt.clf()


def makeCCRCCUnfilteredGlycoPeptideIonsScatter():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\CCRCCPhosphoGlyco\\unfiltered\\global.diagmine.tsv_debug",
        sep='\t')
    filteredPeakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\CCRCCPhosphoGlyco\\global.diagmine.tsv",
        sep='\t')
    filteredPeakData = filteredPeakData[filteredPeakData['ion_type'] == 'peptide']
    filteredPeakData = filteredPeakData[filteredPeakData['peak_apex'] > 50]
    peakData = peakData[peakData['ion_type'] == 'Y']
    peakData = peakData[peakData['peak_apex'] > 50]
    peakData['pct_unmod_spectra'] = peakData['prop_unmod_spectra'] * 100.0
    peakData['ppv'] = peakData['prop_mod_spectra'] / \
                      (peakData['prop_unmod_spectra'] + peakData['prop_mod_spectra'])
    filteredPeakData['ppv'] = filteredPeakData['percent_mod'] / \
                              (filteredPeakData['percent_unmod'] + filteredPeakData['percent_mod'])
    peakData['sensitivity'] = peakData['prop_mod_spectra']
    filteredPeakData['sensitivity'] = filteredPeakData['percent_mod'] / 100.0
    peakData['F'] = 2 * peakData['ppv'] * peakData['prop_mod_spectra'] / \
                    (peakData['ppv'] + peakData['prop_mod_spectra'])
    peakData['avg_spec_diff'] = (peakData['prop_mod_spectra'] - peakData['prop_unmod_spectra']) * 100.0
    peakData['ppv_jitter'] = peakData['ppv'] + np.random.normal(0, 0.015, len(peakData['ppv']))
    peakData['F_jitter'] = peakData['F'] + np.random.normal(0, 0.015, len(peakData['F']))
    peakData['auc_jitter'] = peakData['auc'] + np.random.normal(0, 0.01, len(peakData['F']))
    peakData['avg_int'] = peakData['mod_spectra_int']
    peakData['e_value'] = peakData['e_value'] + 1e-200
    peakData['log_e_value'] = -1 * np.log(peakData['e_value'])

    filteredPeakData['log_mass'] = np.log(filteredPeakData['mass'])
    filteredPeakData['color_mass'] = filteredPeakData['mass']
    filteredPeakData[filteredPeakData['color_mass'] > 750] = 750
    filteredPeakData[filteredPeakData['color_mass'] < -25] = -25

    peakData = peakData.sort_values(by='ppv', ascending=False)

    plt.figure(figsize=(9*cm,7*cm))
    sns.scatterplot(data=filteredPeakData, x='avg_intensity_mod', y='ppv',
                    alpha=0.2, size=4, legend=False, hue='color_mass', palette='viridis')
    #sns.regplot(data=filteredPeakData, lowess=True, x='avg_intensity_mod', y='ppv', scatter=False,
    #            line_kws={'color': 'red', 'lw': 1, 'linestyle': 'dashed'})
    sns.despine()
    plt.xlabel('avg intensity (spectra with ion)', fontsize=11)
    plt.ylabel('precision',fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlim(0, 105)
    plt.ylim(0.6, 1.05)

    plt.tight_layout(pad=0)
    plt.savefig('CCRCCPhosphoGlyco/unfiltered/peptide_scatter_filtered_precision.png')
    plt.clf()

    # peakData['avg_int'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int']

    plt.figure(figsize=(6 * cm, 7 * cm))
    sns.scatterplot(data=filteredPeakData, x='avg_intensity_mod', y='auc', legend=False,
                    alpha=0.2, size=2)#, hue='color_mass', palette='viridis', rasterized=True)
    #sns.regplot(data=filteredPeakData, lowess=True, x='avg_intensity_mod', y='auc', scatter=False,
    #            line_kws={'color': 'red', 'lw': 1, 'linestyle': 'dashed'})
    sns.despine()
    plt.xlabel('avg intensity (spectra with ion)', fontsize=6)
    plt.ylabel('auc of intensity', fontsize=6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    plt.tight_layout(pad=0)
    plt.xlim(0, 105)
    plt.ylim(0.6, 1.05)

    plt.savefig('CCRCCPhosphoGlyco/unfiltered/peptide_scatter_filtered_auc_intensity.pdf')
    plt.clf()


def makeCCRCCUnfilteredGlycoDiagnosticIonsScatter():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\CCRCCPhosphoGlyco\\unfiltered\\global.diagmine.tsv_debug",
        sep='\t')
    filteredPeakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\CCRCCPhosphoGlyco\\global.diagmine.tsv",
        sep='\t')
    peakData = peakData[peakData['ion_type'] == 'diagnostic']
    peakData = peakData[peakData['peak_apex'] > 50]
    filteredPeakData = filteredPeakData[filteredPeakData['ion_type'] == 'diagnostic']
    filteredPeakData = filteredPeakData[filteredPeakData['peak_apex'] > 50]
    filteredPeakData['log_mass'] = np.log(filteredPeakData['mass'])
    filteredPeakData['color_mass'] = filteredPeakData['mass']
    filteredPeakData[filteredPeakData['color_mass'] > 300] = 300

    peakData['pct_unmod_spectra'] = peakData['prop_unmod_spectra'] * 100.0
    peakData['ppv'] = peakData['prop_mod_spectra'] / \
                      (peakData['prop_unmod_spectra'] + peakData['prop_mod_spectra'])
    filteredPeakData['ppv'] = filteredPeakData['percent_mod'] / \
                              (filteredPeakData['percent_unmod'] + filteredPeakData['percent_mod'])
    peakData['F'] = 2 * peakData['ppv'] * peakData['prop_mod_spectra'] / \
                    (peakData['ppv'] + peakData['prop_mod_spectra'])
    filteredPeakData['F'] = (2 * filteredPeakData['ppv'] * (filteredPeakData['percent_mod'] / 100.0)) / \
                            (filteredPeakData['ppv'] + (filteredPeakData['percent_mod'] / 100.0))

    peakData['sensitivity'] = peakData['prop_mod_spectra']
    filteredPeakData['sensitivity'] = filteredPeakData['percent_mod'] / 100.0
    filteredPeakData['specificity'] = (100.0 - filteredPeakData['percent_unmod']) / 100.0
    peakData['avg_spec_diff'] = (peakData['prop_mod_spectra'] - peakData['prop_unmod_spectra']) * 100.0
    peakData['avg_int'] = peakData['mod_spectra_int']
    peakData['e_value'] = peakData['e_value'] + 1e-200
    peakData['log_e_value'] = -1 * np.log(peakData['e_value'])

    print('making scatter')
    peakData['ppv_jitter'] = peakData['ppv'] + np.random.normal(0, 0.01, len(peakData['ppv']))
    peakData['F_jitter'] = peakData['F'] + np.random.normal(0, 0.01, len(peakData['F']))
    peakData['auc_jitter'] = peakData['auc'] + np.random.normal(0, 0.01, len(peakData['F']))
    print(filteredPeakData.columns)
    plt.figure(figsize=(9 * cm, 7 * cm))
    #sns.scatterplot(data=peakData, x='avg_int', y='ppv_jitter',
    #                alpha=0.25, hue='sensitivity', palette='viridis')
    ax = sns.scatterplot(data=filteredPeakData, x='avg_intensity_mod', y='ppv',
                    alpha=0.1, size=4, legend=False, c=filteredPeakData['color_mass'], cmap='viridis')
    #sns.regplot(data=filteredPeakData, lowess=True, x='avg_intensity_mod', y='ppv', scatter=False,
    #            line_kws={'color': 'red', 'lw': 2, 'linestyle': 'dashed'})

    sns.despine()
    plt.xlabel('avg intensity (spectra with ion)', fontsize=6)
    plt.ylabel('precision', fontsize=6)
    plt.xlim(0, 105)
    plt.ylim(0.6, 1.05)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    plt.tight_layout(pad=0)
    plt.savefig('CCRCCPhosphoGlyco/unfiltered/diagnostic_scatter_filtered_precision.png')
    plt.clf()

    # peakData['avg_int'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int']

    plt.figure(figsize=(9 * cm, 7 * cm))
    #sns.scatterplot(data=filteredPeakData, x='avg_int', y='auc_jitter',
                    #alpha=0.25, hue='sensitivity', palette='viridis')
    sns.scatterplot(data=filteredPeakData, x='avg_intensity_mod', y='auc', legend=False,
                    alpha=0.1, size=2, hue='color_mass', palette='viridis', rasterized=True)
    #sns.regplot(data=filteredPeakData, lowess=True, x='avg_intensity_mod', y='auc', scatter=False,
    #            line_kws={'color': 'red', 'lw': 2, 'linestyle': 'dashed'})
    sns.despine()
    plt.xlabel('avg intensity (spectra with ion)', fontsize=6)
    plt.ylabel('auc of intensity', fontsize=6)
    plt.xlim(0, 105)
    plt.ylim(0.6, 1.05)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    plt.tight_layout(pad=0)
    plt.savefig('CCRCCPhosphoGlyco/unfiltered/diagnostic_scatter_filtered_auc_intensity.pdf')
    plt.clf()

    sns.scatterplot(data=filteredPeakData, x='avg_intensity_mod', y='F',
                    alpha=0.1)
    sns.regplot(data=filteredPeakData, lowess=True, x='avg_intensity_mod', y='F', scatter=False,
                line_kws={'color': 'red', 'lw': 2, 'linestyle': 'dashed'})
    sns.despine()
    plt.xlabel('avg intensity (spectra with ion)')
    plt.ylabel('F')
    plt.xlim(0, 100)
    plt.ylim(0, 1.05)
    plt.savefig('CCRCCPhosphoGlyco/unfiltered/diagnostic_scatter_filtered_F_intensity.png')
    plt.clf()


def makeCCRCCDiagnosticHeatMap():
    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})

    diagData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\CCRCCPhosphoGlyco\\combined.diagnosticIons.tsv",
        sep='\t')
    diagData = diagData[diagData['Mass Shift'] > 50]
    diagCols = [col for col in diagData.columns if 'ox_' in col]
    knownIons = ['204.0866', '186.0761', '168.0655', '366.1395', '144.0656', '138.0550', '512.1974', '292.1027',
                 '274.0921',
                 '657.2349', '243.0264', '405.0792', '485.0456', '308.0976']
    newIons = [110.058557, 122.06008, 126.055, 145.049545, 163.060081, 166.049818, 167.034034, 172.060453, 182.044614,
               184.060287, 196.060271, 197.044747, 214.070836, 215.05489, 232.081386, 436.144679, 528.192035]
    newIons = [110.0586, 122.0601, 126.055, 145.0495, 163.0601, 166.0498, 167.034, 172.0605, 182.0446, 184.0603, 196.0603,
    197.0447, 214.0708, 215.0549, 232.0814, 436.1447,
    528.192]

    ions = list(map("{:.4f}".format,
                    [204.0866, 186.0761, 168.0655, 366.1395, 144.0656, 138.055, 512.1974, 292.1027, 274.0921, 657.2349,
                     243.0264, 405.0792, 485.0456, 308.0976, 110.0586, 112.0395, 122.0601, 123.0441, 126.055, 145.0495,
                     152.0706, 154.0498, 163.0601, 166.0498, 167.034, 172.0605, 173.0444, 182.0446, 184.0603, 196.0603,
                     197.0447, 208.0603, 214.0708, 215.0549, 226.0714, 232.0814, 246.097, 392.2228, 436.1447, 454.1553,
                     521.2656, 528.192, 569.2185, 595.302]))
    labels = ["known", "known", "known", "known", "known", "known", "known", "known", "known", "known", "known",
              "known", "known", "known", "unknown", "unknown", "unknown", "unknown", "known", "known", "unknown",
              "unknown", "known", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown",
              "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown",
              "known", "unknown", "known", "known", "unknown"]
    knownNewMapping = dict(zip(ions, labels))

    oxData = diagData[diagCols]
    # oxData.where(oxData < 99.99999, np.nan, inplace=True)
    # oxData.where(oxData > 0.00001, np.nan, inplace=True)

    diagCols = [col.split('_')[1] for col in diagCols]
    oxData.columns = diagCols
    sortedDiagCols = list(map("{:.4f}".format, sorted(map(float, knownIons + newIons))))
    print(sortedDiagCols)
    print(sortedDiagCols)
    oxData = oxData[sortedDiagCols]
    corrMat = oxData.corr(method='spearman')
    corrMat.to_csv(
        'C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\CCRCCPhosphoGlyco\\diagnostic_correlation_matrix.tsv',
        sep='\t')
    classColors = []

    for col in corrMat.columns:
        mappy = knownNewMapping[col]
        if mappy == 'known':
            classColors.append('darkblue')
        elif mappy == 'unknown':
            classColors.append('darkgreen')
        else:
            print(col, knownNewMapping.keys())
            classColors.append('purple')

    print(corrMat.columns)
    print(classColors)
    corrMat.fillna(0, inplace=True)
    plt.figure(figsize=(12 * cm, 12 * cm))
    ax = sns.clustermap(corrMat,
                   row_colors=classColors, col_colors=classColors,
                   xticklabels=True, yticklabels=True,
                   cmap='viridis',
                   dendrogram_ratio=(0.1, 0.1),
                   linewidths=0.5,
                   cbar_kws={'ticks': [-0.25, 0, 0.25, 0.5, 0.75, 1.0], 'aspect': 3.0,
                             'label': 'Spearman\ncorrelation'})
    ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xmajorticklabels(), fontsize=6)
    ax.ax_heatmap.set_yticklabels(ax.ax_heatmap.get_ymajorticklabels(), fontsize=6)
    plt.xticks(size=6)
    plt.yticks(size=6)
    plt.tight_layout(pad=0)
    plt.savefig('CCRCCPhosphoGlyco/diagnostic_correlation_heatmap.pdf')
    plt.clf()


def makeCCRCCHistos():
    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})

    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\CCRCCPhosphoGlyco\\global.diagmine.tsv",
        sep='\t')

    """
    make diag ion histogram/scatter
    """
    diagData = peakData[peakData['ion_type'] == 'diagnostic']
    # diagData['diagInt'] = (peakData['percent_mod'] / 100.0) * peakData['avg_intensity_mod']
    diagData['diagInt'] = (peakData['intensity_fold_change'])
    knownDiags = [204.086646, 186.076086, 168.065526, 366.139466, 144.0656, 138.055, 512.197375,
                  292.1026925, 274.0921325, 657.2349, 243.026426, 405.079246, 485.045576, 308.09761]

    # knownDiags = [144.0656, 138.055, 126.055, 145.067, 204.086646, 186.076086, 168.065526, 366.139466, 163.060096,
    #              292.1026925, 274.0921325, 657.2348825, 454.1555125, 512.197375, 528.192286, 690.245106, 803.2927915,
    #              893.324476, 243.026426, 405.079246, 485.045576, 284.043461, 446.096281, 811.228471, 308.097576,
    #              290.087016, 673.229766, 511.176946, 350.144555, 948.330299, 495.1820625, 1239.425716, 745.250929,
    #              583.198109, 819.287675, 470.150396, 658.255284, 407.166016, 569.218836, 553.223925, 698.2614325,
    #              964.3251825, 980.320066]

    bins = [i - 0.5 for i in range(100, 801)]
    means = [0 for i in range(100, 801)]
    ns = [0 for i in range(100, 801)]
    diagInts = [0 for i in range(100, 801)]
    colors = ['gray' for i in range(100, 801)]

    for mass, intens in zip(diagData['mass'], diagData['diagInt']):
        if mass > 800:
            continue
        for i, bin in enumerate(bins):
            if mass >= bin and mass < bins[i + 1]:
                means[i] += mass
                diagInts[i] += intens
                ns[i] += 1
                break

    for i, mean in enumerate(means):
        if ns[i] == 0:
            continue
        means[i] = mean / ns[i]
        diagInts[i] = diagInts[i] / ns[i]

    for mass in knownDiags:
        for i, bin in enumerate(bins):
            if mass > 800:
                break
            if mass >= bin and mass < bins[i + 1]:
                colors[i] = 'darkblue'
                break

    for i, diagInt in enumerate(diagInts):
        if ns[i] > 250:
            print(means[i])
            if colors[i] == 'gray':
                colors[i] = 'darkgreen'

    totPeaks = 0
    totPoints = 0
    for i,n in enumerate(ns):
        if n > 0:
            totPeaks += 1
            totPoints += n

    print(totPeaks, totPoints)

    palette = {'gray': 'gray', 'darkblue': 'darkblue', 'darkgreen': 'darkgreen'}

    scatterDf = pd.DataFrame()
    scatterDf['counts'] = ns
    scatterDf['log10_intensities'] = np.log10(diagInts)
    scatterDf['masses'] = means
    scatterDf['colors'] = colors
    scatterDf = scatterDf[scatterDf['counts'] > 1]

    # rugDf = scatterDf[scatterDf['colors'].isin(['darkblue'])]

    # print(textLabels)

    print('making diagnostic scatterplot')
    plt.figure(figsize=(4*cm,4*cm))
    ax = sns.scatterplot(data=scatterDf, x="masses", y="counts", hue="colors",
                         size="log10_intensities", sizes=(10, 50), alpha=0.75,
                         palette=palette, legend=True)
    ax.legend(loc='center left', prop={'size': 6})
    # ax.legend(bbox_to_anchor= (1.2,1), title='diag ions', labels=['unknown', 'low recurrence', 'known'])
    # sns.rugplot(data=rugDf, x="masses",
    #            hue="colors", palette=palette, legend=False)
    # for index, row in scatterDf.iterrows:
    #    if row['colors'] == 'darkblue':
    #        ax.text(row['masses']+5, row['counts']+5, round(row['masses'], 3), ha='center', va='center', size=8, rotation='45')
    plt.xlim(100, 300)
    plt.ylim(000, 500)
    plt.ylabel('n glyco ass shifts', fontsize=6)
    plt.xlabel('diagnostic ion m/z', fontsize=6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    sns.despine()
    plt.tight_layout(pad=0)
    plt.savefig('CCRCCPhosphoGlyco/diagnostic_ions_scatter_wlegend.pdf')
    plt.clf()

    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    ax = sns.histplot(x='mass', data=diagData, binwidth=1, binrange=(100, 800), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(100, 1000)
    plt.ylim(000, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 700:
            ax.text(x + w / 2, h + 5, str(round(x)), ha='center', va='center', size=5, rotation='vertical')
    sns.despine()
    plt.savefig('CCRCCPhosphoGlyco/diagnostic_ions_histo.pdf')
    plt.clf()

    print('making histo zoom')
    ax = sns.histplot(x='mass', data=diagData, binwidth=1, binrange=(100, 400), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(100, 400)
    plt.ylim(0, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 25:
            ax.text(x + w / 2, h + 5, str(round(x, 2)), ha='center', va='center', size=5, rotation='vertical')
    sns.despine()
    plt.savefig('CCRCCPhosphoGlyco/diagnostic_ions_histo_zoom.pdf')
    plt.clf()

    """
    make Y ion histogram
    """
    diagData = peakData[peakData['ion_type'] == 'peptide']
    diagData = diagData[diagData['peak_apex'] > 50.0]
    knownYs = [0, 203.07937, 406.15874, 568.21156, 730.26438, 892.3172, 349.137279]

    bins = [i - 0.5 for i in range(-200, 2001)]
    means = [0 for i in range(-200, 2001)]
    ns = [0 for i in range(-200, 2001)]
    colors = ['gray' for i in range(-200, 2001)]

    for mass in diagData['mass']:
        if mass > 2000:
            continue
        for i, bin in enumerate(bins):
            if mass >= bin and mass < bins[i + 1]:
                means[i] += mass
                ns[i] += 1
                break

    for i, mean in enumerate(means):
        if ns[i] == 0:
            continue
        means[i] = mean / ns[i]

    for mass in knownYs:
        for i, bin in enumerate(bins):
            if mass >= bin and mass < bins[i + 1]:
                colors[i] = 'darkblue'
                break

    totPeaks = 0
    totPoints = 0
    for i, n in enumerate(ns):
        if n > 0:
            totPeaks += 1
            totPoints += n

    print(totPeaks, totPoints, max(ns))

    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    plt.figure(figsize=(4 * cm, 4 * cm))
    ax = sns.histplot(x='mass', data=diagData, binwidth=1, binrange=(-200.5, 2000.5), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(-200, 1000)
    print(len(peakData['peak_apex'].unique()))
    plt.ylim(0, 500)
    for i, p in enumerate(ax.patches):
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        p.set_facecolor(colors[i])
        p.set_edgecolor(colors[i])
        if h > 50 or colors[i] == 'darkblue':
            if colors[i] == 'gray':
                p.set_edgecolor('darkgreen')
                p.set_facecolor('darkgreen')
            ax.text(x + w / 2 + 10, h + 17, round(means[i], 3), ha='center', va='center', size=5, rotation='45')

    sns.despine()
    ax.set_ylabel('n glyco mass shifts', fontsize=6)
    ax.set_xlabel('peptide remainder mass', fontsize=6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout(pad=0)
    plt.savefig('CCRCCPhosphoGlyco/capY_ions_histo.pdf')
    plt.clf()

    print('making histo zoom')
    """
    ax = sns.histplot(x='adjusted_mass', data=peakData, binwidth=0.01, binrange=(100, 400), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(100, 400)
    plt.ylim(0, 1000)
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 1:
            print(round(x, 2))
            ax.text(x + w / 2, h + 50, str(round(x, 2)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('MouseGlyco/diagnostic_ions_histo_zoom.png')
    plt.clf()
    """

    """
    diagData = peakData[peakData['ion_type'] == 'b']

    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    ax = sns.histplot(x='mass', data=diagData, binwidth=1, binrange=(-50, 1000), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(-50, 1000)
    plt.ylim(000, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 25:
            ax.text(x + w / 2, h + 5, str(round(x)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('CCRCCPhosphoGlyco/b_ions_histo.png')
    plt.clf()
    """

    diagData = peakData[peakData['ion_type'] == 'b']

    diagData = diagData[diagData['peak_apex'] > 50.0]

    bins = [i - 0.5 for i in range(-50, 501)]
    means = [0 for i in range(-50, 501)]
    ns = [0 for i in range(-50, 501)]
    colors = ['gray' for i in range(-50, 501)]

    for mass in diagData['mass']:
        if mass > 500:
            continue
        for i, bin in enumerate(bins):
            if mass >= bin and mass < bins[i + 1]:
                means[i] += mass
                ns[i] += 1
                break

    for i, mean in enumerate(means):
        if ns[i] == 0:
            continue
        means[i] = mean / ns[i]

    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    plt.figure(figsize=(4 * cm, 4 * cm))
    ax = sns.histplot(x='mass', data=diagData, binwidth=1, binrange=(-50.5, 500.5), log_scale=(False, False),
                      linewidth=1)
    sns.despine()
    plt.xlim(-50, 500)
    plt.ylim(000, 500)
    for i, p in enumerate(ax.patches):
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        p.set_facecolor(colors[i])
        p.set_edgecolor(colors[i])
        if h > 50:
            p.set_edgecolor('darkblue')
            p.set_facecolor('darkblue')
            ax.text(x + w / 2 + 10, h + 20, round(means[i], 3), ha='center', va='center', size=5, rotation='45')
    ax.set_ylabel('n glyco mass shifts', fontsize=6)
    ax.set_xlabel('b~ fragment remainder mass', fontsize=6)
    plt.tight_layout()
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout(pad=0)
    plt.savefig('CCRCCPhosphoGlyco/b_ions_histo.pdf')
    plt.clf()

    diagData = peakData[peakData['ion_type'] == 'y']

    diagData = diagData[diagData['peak_apex'] > 50.0]

    bins = [i - 0.5 for i in range(-50, 501)]
    means = [0 for i in range(-50, 501)]
    ns = [0 for i in range(-50, 501)]
    colors = ['gray' for i in range(-50, 501)]

    for mass in diagData['mass']:
        if mass > 500:
            continue
        for i, bin in enumerate(bins):
            if mass >= bin and mass < bins[i + 1]:
                means[i] += mass
                ns[i] += 1
                break

    for i, mean in enumerate(means):
        if ns[i] == 0:
            continue
        means[i] = mean / ns[i]

    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    plt.figure(figsize=(4 * cm, 4 * cm))
    ax = sns.histplot(x='mass', data=diagData, binwidth=1, binrange=(-50.5, 500.5), log_scale=(False, False),
                      linewidth=1)
    sns.despine()
    plt.xlim(-50, 500)
    plt.ylim(000, 500)
    for i, p in enumerate(ax.patches):
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        p.set_facecolor(colors[i])
        p.set_edgecolor(colors[i])
        if h > 50:
            p.set_edgecolor('darkblue')
            p.set_facecolor('darkblue')
            ax.text(x + w / 2 + 10, h + 20, round(means[i], 3), ha='center', va='center', size=5, rotation='45')
    ax.set_ylabel('n glyco mass shifts', fontsize=6)
    ax.set_xlabel('y~ fragment remainder mass', fontsize=6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout(pad=0)
    plt.savefig('CCRCCPhosphoGlyco/y_ions_histo.pdf')
    plt.clf()


def makeRNAXLinkBaeHistos():
    fig, axes = plt.subplots(2, 2, sharex='col', sharey='all')

    mzs1 = []
    knownMzs1 = [94.03, 77.0068]
    ints1 = []
    with open(
            "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\RNAXLink_Bae\\4SU\\226.0594_y.tsv") \
            as fin:
        fin.readline()
        for line in fin:
            line = line.rstrip().split('\t')
            mzs1.append(float(line[0]))
            ints1.append(float(line[1]))

    mzs2 = []
    ints2 = []
    knownMzs2 = [94.0306, 77.0064]
    with open(
            "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\RNAXLink_Bae\\4SU\\94.017_y.tsv") \
            as fin:
        fin.readline()
        for line in fin:
            line = line.rstrip().split('\t')
            mzs2.append(float(line[0]))
            ints2.append(float(line[1]))

    mzs3 = []
    ints3 = []
    knownMzs3 = [133.0502, 115.0396]
    with open(
            "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\RNAXLink_Bae\\4SU\\226.0594_diag.tsv") \
            as fin:
        fin.readline()
        for line in fin:
            line = line.rstrip().split('\t')
            mzs3.append(float(line[0]))
            ints3.append(float(line[1]))

    mzs4 = []
    ints4 = []
    knownMzs4 = [215.0582]
    with open(
            "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\RNAXLink_Bae\\4SU\\94.017_diag.tsv") \
            as fin:
        fin.readline()
        for line in fin:
            line = line.rstrip().split('\t')
            mzs4.append(float(line[0]))
            ints4.append(float(line[1]))

    minx = -150
    maxx = 450
    minx2 = 100
    maxx2 = 300

    # make plot, define bounds
    sns.scatterplot(ax=axes[0, 0], x=mzs1, y=ints1, s=0)
    sns.scatterplot(ax=axes[1, 0], x=mzs2, y=ints2, s=0)
    sns.scatterplot(ax=axes[0, 1], x=mzs3, y=ints3, s=0)
    sns.scatterplot(ax=axes[1, 1], x=mzs4, y=ints4, s=0)
    plt.gcf().set_size_inches(8.8 * cm, 8.8 * cm)
    # g.set_yticks([-150, 0, 150, 300, 450])
    plt.ylim(0, 100)
    axes[0, 0].set_xlim(minx, maxx)
    axes[1, 1].set_xlim(minx2, maxx2)
    #plt.suptitle('4SU diagnostic features')
    axes[0, 0].set_title("y~ fragments", fontsize=6)
    axes[0, 1].set_title("diagnostic ions", fontsize=6)
    axes[0, 0].set_ylabel("226 mass", fontsize=6)
    axes[1, 0].set_ylabel("94 mass", fontsize=6)
    axes[0, 0].tick_params(axis='both',which='major',labelsize=6)
    axes[0, 0].tick_params(axis='both', which='minor', labelsize=6)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=6)
    axes[0, 1].tick_params(axis='both', which='minor', labelsize=6)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=6)
    axes[1, 0].tick_params(axis='both', which='minor', labelsize=6)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=6)
    axes[1, 1].tick_params(axis='both', which='minor', labelsize=6)
    sns.despine()
    fig.supxlabel('delta m/z', fontsize=6)

    # plotting vertical lines and labels
    top100Inds1 = sorted(range(len(ints1)), key=lambda i: ints1[i], reverse=True)
    top100Inds2 = sorted(range(len(ints2)), key=lambda i: ints2[i], reverse=True)
    top100Inds3 = sorted(range(len(ints3)), key=lambda i: ints3[i], reverse=True)
    top100Inds4 = sorted(range(len(ints4)), key=lambda i: ints4[i], reverse=True)
    # top5Inds = sorted(range(len(heights)), key=lambda i: heights[i], reverse=True)[:10]
    for ind in top100Inds1:
        if mzs1[ind] in knownMzs1:
            axes[0, 0].axvline(mzs1[ind], 0, (ints1[ind] / max(ints1)), color='darkred', linewidth=1.25)
        else:
            axes[0, 0].axvline(mzs1[ind], 0, (ints1[ind] / max(ints1)), color='gray', linewidth=0.75)
    for ind in top100Inds2:
        print(mzs2[ind])
        if mzs2[ind] in knownMzs2:
            axes[1, 0].axvline(mzs2[ind], 0, (ints2[ind] / max(ints2)), color='darkred', linewidth=1.25)
        else:
            axes[1, 0].axvline(mzs2[ind], 0, (ints2[ind] / max(ints2)), color='gray', linewidth=0.75)
    for ind in top100Inds3:
        if mzs3[ind] in knownMzs3:
            axes[0, 1].axvline(mzs3[ind], 0, (ints3[ind] / max(ints3)), color='darkred', linewidth=1.25)
        else:
            axes[0, 1].axvline(mzs3[ind], 0, (ints3[ind] / max(ints3)), color='gray', linewidth=0.75)
    for ind in top100Inds4:
        if mzs4[ind] in knownMzs4:
            axes[1, 1].axvline(mzs4[ind], 0, (ints4[ind] / max(ints4)), color='darkred', linewidth=1.25)
        else:
            axes[1, 1].axvline(mzs4[ind], 0, (ints4[ind] / max(ints4)), color='gray', linewidth=0.75)
    # for ind in top5Inds:
    #    plt.text(x=bins[ind], y=heights[ind], s=str(bins[ind]))
    print("here3")
    plt.tight_layout(pad=0)
    plt.savefig('RNAXLink_Bae/226and94combined_ions.pdf')
    # plt.savefig(dmass + k + '.png')
    plt.clf()


def makePDACHistos():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\PDACGlyco\\global.diagmine.tsv",
        sep='\t')

    diagData = peakData[peakData['ion_type'] == 'diagnostic']

    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    ax = sns.histplot(x='adjusted_mass', data=diagData, binwidth=1, binrange=(100, 800), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(100, 800)
    plt.ylim(000, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 5:
            print(round(x, 2))
            ax.text(x + w / 2, h + 5, str(round(x)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('PDACGlyco/diagnostic_ions_histo.png')
    plt.clf()

    print('making histo zoom')
    ax = sns.histplot(x='adjusted_mass', data=diagData, binwidth=1, binrange=(100, 400), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(100, 400)
    plt.ylim(0, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 1:
            print(round(x, 2))
            ax.text(x + w / 2, h + 5, str(round(x, 2)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('PDACGlyco/diagnostic_ions_histo_zoom.png')
    plt.clf()

    diagData = peakData[peakData['ion_type'] == 'Y']

    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    ax = sns.histplot(x='adjusted_mass', data=diagData, binwidth=1, binrange=(0, 2000), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(0, 2000)
    plt.ylim(000, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 5:
            print(round(x, 2))
            ax.text(x + w / 2, h + 5, str(round(x)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('PDACGlyco/capY_ions_histo.png')
    plt.clf()

    print('making histo zoom')
    """
    ax = sns.histplot(x='adjusted_mass', data=peakData, binwidth=0.01, binrange=(100, 400), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(100, 400)
    plt.ylim(0, 1000)
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 1:
            print(round(x, 2))
            ax.text(x + w / 2, h + 50, str(round(x, 2)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('MouseGlyco/diagnostic_ions_histo_zoom.png')
    plt.clf()
    """

    diagData = peakData[peakData['ion_type'] == 'b']

    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    ax = sns.histplot(x='adjusted_mass', data=diagData, binwidth=1, binrange=(-50, 1000), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(-50, 1000)
    plt.ylim(000, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 5:
            print(round(x, 2))
            ax.text(x + w / 2, h + 5, str(round(x)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('PDACGlyco/b_ions_histo.png')
    plt.clf()

    diagData = peakData[peakData['ion_type'] == 'y']

    print('making histo')
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    ax = sns.histplot(x='adjusted_mass', data=diagData, binwidth=1, binrange=(-50, 1000), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(-50, 1000)
    plt.ylim(000, len(peakData['peak_apex'].unique()))
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 5:
            print(round(x))
            ax.text(x + w / 2, h + 5, str(round(x)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('PDACGlyco/y_ions_histo.png')
    plt.clf()


def calcFlrForPhospho():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\PDACGlyco\\global.diagmine.tsv",
        sep='\t')


def makeGlycoDiagnosticFeatureScatters():
    makeGlycoDiagnosticIonsScatter()
    # makeGlycoYIonsScatter()
    # makeGlyco_yIonsScatter()
    # makeGlyco_bIonsScatter()
    # makeGlycoDiagnosticIons2DScatter()
    # makeGlycoYIons2DScatter()
    # makeGlyco_yIons2DScatter()
    # makeGlyco_bIons2DScatter()
    # makeGlycoYIonsScatter()
    # makeGlyco_yIonsScatter()
    # makeGlyco_bIonsScatter()
    print()


def makeOxMetFigures():
    makeLocalizationHeatmap()


def makeLocalizationHeatmap():
    diagIonData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\OxMetProbe\\dataset01.rawglyco",
        sep='\t')
    diagIonData['nominal_mass'] = diagIonData.apply(multiMassMarker, axis=1)
    locCols = ['localization_-47.9992', 'localization_661.3660', 'localization_667.3735']
    locData = diagIonData[locCols + ['nominal_mass']]
    locData = pd.melt(locData, id_vars=['nominal_mass'], value_vars=locCols)
    locData['loc_res'] = locData.apply(multiLocMarkerOxMet, axis=1)

    fig, axes = plt.subplots(1, 3, sharey=True, gridspec_kw={'width_ratios': [2, 1, 2]})

    # plot light heatmap
    lightData = locData[locData['nominal_mass'] == 'light']
    lightHeatmap = lightData.groupby(['variable', 'loc_res']).size().reset_index(name='counts')
    lightHeatmap = lightHeatmap.pivot(index='variable', columns='loc_res')['counts'].drop(
        'none', axis=1).drop('localization_667.3735', axis=0)
    for aa in aas:
        if aa not in lightHeatmap.columns:
            lightHeatmap[aa] = 0.0
    lightHeatmap = lightHeatmap.fillna(0.0)
    lightHeatmap = lightHeatmap[aas]
    lightLabels = [r.split('_')[1][:-1] for r in lightHeatmap.index]
    print(lightHeatmap)
    seaborn.heatmap(lightHeatmap.T, ax=axes[0], annot=True, fmt='g', linewidth=1, annot_kws={"size": 8}, cbar=False,
                    cmap=truncate_colormap(plt.get_cmap('YlGnBu'), 0, lightHeatmap.values.max() / 1300, 256))

    # plot heavy heatmap
    heavyData = locData[locData['nominal_mass'] == 'heavy']
    heavyHeatmap = heavyData.groupby(['variable', 'loc_res']).size().reset_index(name='counts')
    heavyHeatmap = heavyHeatmap.pivot(index='variable', columns='loc_res')['counts'].drop(
        'none', axis=1).drop('localization_661.3660', axis=0)
    for aa in aas:
        if aa not in heavyHeatmap.columns:
            heavyHeatmap[aa] = 0.0
    print(heavyHeatmap)
    heavyHeatmap = heavyHeatmap.fillna(0.0)
    heavyHeatmap = heavyHeatmap[aas]
    heavyLabels = [r.split('_')[1][:-1] for r in heavyHeatmap.index]
    print(heavyHeatmap)
    seaborn.heatmap(heavyHeatmap.T, ax=axes[2], annot=True, fmt='g', linewidth=1, cbar=False,
                    cmap=truncate_colormap(plt.get_cmap('YlGnBu'), 0, heavyHeatmap.values.max() / 1250, 256))

    # plot none heatmap
    noneData = locData[locData['nominal_mass'] == 'none']
    noneHeatmap = noneData.groupby(['variable', 'loc_res']).size().reset_index(name='counts')
    noneHeatmap = noneHeatmap.pivot(index='variable', columns='loc_res')['counts'].drop(
        'none', axis=1).drop('localization_661.3660', axis=0).drop('localization_667.3735', axis=0)
    for aa in aas:
        if aa not in noneHeatmap.columns:
            noneHeatmap[aa] = 0.0
    noneHeatmap = noneHeatmap[aas]
    noneHeatmap = noneHeatmap.fillna(0)
    noneLabels = [r.split('_')[1][:-1] for r in noneHeatmap.index]
    seaborn.heatmap(noneHeatmap.T, ax=axes[1], annot=True, fmt='g', linewidth=1, cbar=False,
                    cmap=truncate_colormap(plt.get_cmap('YlGnBu'), 0, noneHeatmap.values.max() / 1250, 256))

    axes[0].set_xticklabels(lightLabels, rotation=45, ha="right")
    axes[1].set_xticklabels(noneLabels, rotation=45, ha="right")
    axes[2].set_xticklabels(heavyLabels, rotation=45, ha="right")

    axes[0].set_xlabel("light")
    axes[1].set_xlabel("unmodified")
    axes[2].set_xlabel("heavy")

    for ax in axes:
        for t in ax.texts:
            if float(t.get_text()) >= 1:
                t.set_text(t.get_text())  # if the value is greater than 0.4 then I set the text
            else:
                t.set_text("")  # if not it sets an empty text
        ax.tick_params(axis=u'both', which=u'both', length=0)
        ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig('OxMetProbe/loc_heatmap.png')


def multiLocMarkerOxMet(df):
    if type(df['value']) == float:
        return 'none'
    elif 2 <= len(df['value']) <= 3:
        return "".join(re.findall("[a-zA-Z]+", df['value']))
    else:
        return 'none'


def multiMassMarker(df):
    if 660.5 < df['Mass Shift'] < 663:
        return 'light'
    elif 666.5 < df['Mass Shift'] < 669:
        return 'heavy'
    else:
        return 'none'


def makeLocalizationBarCharts():
    path = "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\LUAD_ProteomeToolsPTMs\\rawglycos_byLocOnly"
    allFiles = glob.glob(path + "\\*.rawglyco")
    # df = pd.concat((pd.read_csv(f, sep='\t') for f in allFiles))
    df = pd.concat(pd.read_csv(f, sep='\t') for f in allFiles)
    df.reset_index(drop=True, inplace=True)
    df['bin'] = df.apply(multiBinMarker, axis=1)
    df = df[df['bin'] == 'Phospho']

    # Prep +80 mass shift
    df.sort_values('deltascore_79.9663', inplace=True, ascending=False)
    df['locRes'] = df.apply(multiLocMarker80, axis=1)
    phosphoRocVals = calcRocFromLocs(['S', 'T', 'Y'], 'deltascore_79.9663', df)
    phosphoF1Vals = calcF1Stat(df, ['S', 'T', 'Y'])
    df['phospho_tp'] = phosphoRocVals[0,].tolist()
    df['phospho_fp'] = phosphoRocVals[1,].tolist()
    df['phospho_f1'] = phosphoF1Vals
    # dfPhospho = df[['phospho_q', 'deltascore_79.9663']]
    # dfPhospho = dfPhospho[dfPhospho['phospho_q'] > 0.0]
    # dfPhospho.reset_index(drop=True, inplace=True)

    fig, axes = plt.subplots(2, 1, sharex=False, gridspec_kw={'height_ratios': [5, 4]})
    # +80 bar chart
    print(df['locRes'].value_counts())
    sns.countplot(y='locRes', data=df, orient='v',
                  order=['S', 'T', 'Y', 'non-STY', 'None'], color='lightskyblue', ax=axes[0])
    axes[0].set_title('+80 da localization')
    # sns.despine()
    # plt.savefig('LUAD_ProteomeToolsPTMs/phospho_loc_bar.png')
    # plt.clf()

    # Prep -18 mass shift
    df.sort_values('deltascore_-18.0106', inplace=True, ascending=False)
    df['locRes'] = df.apply(multiLocMarker18, axis=1)
    # minus18QVals = calcFdrFromLocs(['S', 'T'], 'deltascore_-18.0106', df)
    minus18RocVals = calcRocFromLocs(['S', 'T'], 'deltascore_-18.0106', df)
    minus18F1Vals = calcF1Stat(df, ['S', 'T'])
    df['minus18_tp'] = minus18RocVals[0,]
    df['minus18_fp'] = minus18RocVals[1,]
    df['minus18_f1'] = minus18F1Vals
    # dfMinus18 = df[['minus18_q', 'deltascore_-18.0106']]
    # dfMinus18 = dfMinus18[dfMinus18['minus18_q'] > 0.0]
    # dfMinus18.reset_index(drop=True, inplace=True)

    # -18 bar chart
    print(df['locRes'].value_counts())
    sns.countplot(y='locRes', data=df, orient='v',
                  order=['S', 'T', 'non-ST', 'None'], color='lightskyblue', ax=axes[1])
    sns.despine()
    axes[1].set_title('-18 da localization')
    plt.savefig('LUAD_ProteomeToolsPTMs/loc_bar.png')
    plt.clf()

    # fig, axes = plt.subplots(2, 1, sharex=True)
    sns.lineplot(data=df, x='phospho_fp', y='phospho_tp', color='lightskyblue')
    sns.lineplot(data=df, x='minus18_fp', y='minus18_tp', color='darkblue')
    # sns.lineplot(data=df, x='deltascore_79.9663', y='phospho_f1', ax=axes[1], color='lightskyblue')
    # sns.lineplot(data=df, x='deltascore_-18.0106', y='minus18_f1', ax=axes[1], color='darkblue')
    sns.despine()
    plt.savefig('LUAD_ProteomeToolsPTMs/roc.png')
    plt.clf()


def makeFdrFilteredLocalizationBarChart():
    path = "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\LUAD_ProteomeToolsPTMs"
    allFiles = glob.glob(path + "\\*.rawglyco")
    # df = pd.concat((pd.read_csv(f, sep='\t') for f in allFiles))
    df = pd.concat((pd.read_csv(f, sep='\t') for f in allFiles))
    df.reset_index(drop=True, inplace=True)

    df['bin'] = df.apply(multiBinMarker, axis=1)
    df = df[df['bin'] == 'Phospho']

    fig, axes = plt.subplots(2, 1, sharex=False, gridspec_kw={'height_ratios': [5, 4]})
    """
    Limit to phospho IDs above 3 deltascore
    """
    print('filtering +80 to >5 deltascore')
    deltaScoresArray = df[['Spectrum', 'deltascore_79.9663', 'localization_79.9663', 'bin']]
    deltaScoresArray = deltaScoresArray[deltaScoresArray['deltascore_79.9663'] > 3]
    deltaScoresArray['locRes'] = deltaScoresArray.apply(multiLocMarker80, axis=1)
    axes[0].set_xlim(0, 4000)

    sns.despine()

    print(deltaScoresArray['locRes'].value_counts())
    sns.countplot(y='locRes', data=deltaScoresArray, orient='v',
                  order=['S', 'T', 'Y', 'non-STY', 'None'], color='lightskyblue', ax=axes[0])

    """
    Limit to water loss IDs above 10 deltascore
    """
    print('calculating fdr for +80')
    deltaScoresArray = df[['Spectrum', 'deltascore_-18.0106', 'localization_-18.0106', 'bin']]
    deltaScoresArray = deltaScoresArray[deltaScoresArray['deltascore_-18.0106'] > 9]
    deltaScoresArray['locRes'] = deltaScoresArray.apply(multiLocMarker18, axis=1)

    sns.despine()
    axes[1].set_xlim(0, 4000)

    print(deltaScoresArray['locRes'].value_counts())
    sns.countplot(y='locRes', data=deltaScoresArray, orient='v',
                  order=['S', 'T', 'non-ST', 'None'], color='lightskyblue', ax=axes[1])
    plt.savefig('LUAD_ProteomeToolsPTMs/loc_bar.png')


def makeLocalizationHistos():
    path = "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\LUAD_ProteomeToolsPTMs\\rawglycos_byLocOnly"
    allFiles = glob.glob(path + "\\*.rawglyco")
    # df = pd.concat((pd.read_csv(f, sep='\t') for f in allFiles))
    df = pd.concat((pd.read_csv(f, sep='\t') for f in allFiles))
    df.reset_index(drop=True, inplace=True)

    df['bin'] = df.apply(multiBinMarker, axis=1)
    df = df[df['bin'] != 'None']

    """
    Calculate pseudo FDR, F1, and localizations for +80
    """
    print('calculating fdr for +80')
    tempDeltaScoresArray = df[['Spectrum', 'deltascore_79.9663', 'localization_79.9663', 'bin']]
    tempDeltaScoresArray = tempDeltaScoresArray[tempDeltaScoresArray['bin'].isin(['Phospho', 'Unmod'])]

    fdrArray = pd.concat([tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'Unmod'].sample(
        min(tempDeltaScoresArray['bin'].value_counts())),
        tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'Phospho']])
    fdrArray.sort_values('deltascore_79.9663', inplace=True, ascending=False)
    fdrThresh = calcFdr('Phospho', 'Unmod', 'deltascore_79.9663', fdrArray)

    tempDeltaScoresArray = tempDeltaScoresArray[tempDeltaScoresArray['deltascore_79.9663'] > fdrThresh]
    tempDeltaScoresArray = tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'Phospho']
    tempDeltaScoresArray['isAboveThresh'] = tempDeltaScoresArray['deltascore_79.9663'] > fdrThresh
    tempDeltaScoresArray['locRes'] = tempDeltaScoresArray.apply(multiLocMarker80, axis=1)

    # calculate f1 stats
    tempF1ScoresArray1 = tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'Phospho']
    tempF1ScoresArray1.reset_index(drop=True, inplace=True)
    calcF1Stat(tempF1ScoresArray1, ['S', 'T', 'Y'])
    # sns.lineplot(data=tempF1ScoresArray1, x='deltascore_79.9663', y='f1')
    # plt.savefig('LUAD_ProteomeToolsPTMs/phospho_f1.png')
    # plt.clf()

    # calculate loc-based FDR
    tempLocFdrArray1 = tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'Phospho']
    locFdrThresh = calcFdrFromLocs(['S', 'T', 'Y'], 'deltascore_79.9663', tempLocFdrArray1)
    tempLocFdrArray1 = tempLocFdrArray1[tempLocFdrArray1['deltascore_79.9663'] >= locFdrThresh]

    # plot barplot of localized residues
    tempDeltaScoresArray = tempDeltaScoresArray[tempDeltaScoresArray['isAboveThresh'] == True]
    # sns.countplot(x='locRes', data=tempDeltaScoresArray[tempDeltaScoresArray['locRes'].isin(['S', 'T', 'Y', 'non-STY'])],
    #              orient='v', color='purple')
    # plt.savefig('LUAD_ProteomeToolsPTMs/phospho_loc_bar.png')
    # plt.clf()

    """
    Make histograms for +80
    """
    print("plotting +80da histo")
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_ylim(0.95, 1)
    ax1.set_xlim(0, 50)
    ax2.axvline(3, color='black', linestyle='dashed')
    ax1.tick_params(axis='x', which='both', bottom=False)
    sns.histplot(ax=ax1, x='deltascore_79.9663', data=df, binwidth=0.5, hue='bin', palette=['red', 'blue'],
                 edgecolor='white',
                 stat='proportion', common_norm=False)
    ax2.set_ylim(0, 0.05)
    ax2.set_xlim(0, 50)
    sns.histplot(ax=ax2, x='deltascore_79.9663', data=df, binwidth=0.5, hue='bin', palette=['red', 'blue'],
                 edgecolor='white', stat='proportion', common_norm=False)
    sns.despine(ax=ax1, bottom=True)
    sns.despine(ax=ax2)
    fig.text(0.04, 0.5, 'frequency', va='center', rotation='vertical')
    plt.savefig('LUAD_ProteomeToolsPTMs/phospho_deltascore_histo.png')
    plt.clf()

    """
    Calculate pseudo FDR, F1, and localizations for -18
    """
    print('calculating fdr for -18')
    tempDeltaScoresArray = df[['Spectrum', 'deltascore_-18.0106', 'localization_-18.0106', 'bin']]
    tempDeltaScoresArray = tempDeltaScoresArray[tempDeltaScoresArray['bin'].isin(['Phospho', 'Unmod'])]

    fdrArray = pd.concat([tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'Unmod'].sample(
        min(tempDeltaScoresArray['bin'].value_counts())),
        tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'Phospho']])
    fdrArray.sort_values('deltascore_-18.0106', inplace=True, ascending=False)

    fdrThresh = calcFdr('Phospho', 'Unmod', 'deltascore_-18.0106', fdrArray)
    tempDeltaScoresArray['isAboveThresh'] = tempDeltaScoresArray['deltascore_-18.0106'] > fdrThresh
    tempDeltaScoresArray['locRes'] = tempDeltaScoresArray.apply(multiLocMarker18, axis=1)

    # save temp array for F1 stat
    tempF1ScoresArray2 = tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'Phospho']
    tempF1ScoresArray2.reset_index(drop=True, inplace=True)
    calcF1Stat(tempF1ScoresArray2, ['S', 'T'])
    # sns.lineplot(data=tempF1ScoresArray2, x='deltascore_-18.0106', y='f1')
    # plt.savefig('LUAD_ProteomeToolsPTMs/minus18_f1.png')
    # plt.clf()

    # calculate loc-based FDR
    tempLocFdrArray1 = tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'Phospho']
    locFdrThresh = calcFdrFromLocs(['S', 'T'], 'deltascore_-18.0106', tempLocFdrArray1)

    # plot piechart and barplot of localized residues
    tempDeltaScoresArray = tempDeltaScoresArray[tempDeltaScoresArray['isAboveThresh'] == True]
    # sns.countplot(x='locRes', data=tempDeltaScoresArray[tempDeltaScoresArray['locRes'].isin(['S', 'T', 'Y', 'non-STY'])],
    #              orient='v', color='purple')
    # plt.savefig('LUAD_ProteomeToolsPTMs/minus18_loc_bar.png')
    # plt.clf()

    """
    Make histograms for -18
    """
    print("plotting -18da histo")
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_ylim(0.95, 1)
    ax1.set_xlim(0, 50)
    ax2.axvline(9, color='black', linestyle='dashed')
    ax1.tick_params(axis='x', which='both', bottom=False)
    sns.histplot(ax=ax1, x='deltascore_-18.0106', data=df, binwidth=0.5, hue='bin', palette=['red', 'blue'],
                 edgecolor='white',
                 stat='proportion', common_norm=False)
    ax2.set_ylim(0, 0.05)
    ax2.set_xlim(0, 50)
    sns.histplot(ax=ax2, x='deltascore_-18.0106', data=df, binwidth=0.5, hue='bin', palette=['red', 'blue'],
                 edgecolor='white',
                 stat='proportion', common_norm=False)
    sns.despine(ax=ax1, bottom=True)
    sns.despine(ax=ax2)
    fig.text(0.04, 0.5, 'frequency', va='center', rotation='vertical')
    plt.savefig('LUAD_ProteomeToolsPTMs/minus18_deltascore_histo.png')
    plt.clf()


def makeGlycoDiagnosticIonsScatter():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\CCRCCGlyco\\global.diagmine.tsv",
        sep='\t')
    peakData = peakData[peakData['ion_type'] == 'diagnostic']
    peakData.fillna(0)
    peakData['p_value'] = peakData['p_value'] + 1e-200
    peakData['log_p_value'] = -1 * np.log(peakData['p_value'])
    peakData['avg_int_diff'] = np.log((peakData['prop_mod_spectra'] * peakData['mod_spectra_int'] + 1) /
                                      (peakData['prop_unmod_spectra'] * peakData['unmod_spectra_int'] + 1))
    peakData['avg_int'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int']
    peakData['color'] = peakData.apply(multiSignifMarker, axis=1)
    peakData['alpha'] = peakData.apply(multiSignifMarkerAlpha, axis=1)
    peakData['alpha_color'] = peakData.apply(multiSignifMarkerAlphaColor, axis=1)

    fig = plt.figure()
    ax = Axes3D(fig)
    print('making scatter')
    ax.scatter(list(peakData['auc']), list(peakData['log_p_value']), list(peakData['avg_int_diff']),
               c=list(peakData['avg_int']), s=1, cmap=plt.get_cmap('viridis'))
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 400)
    ax.set_zlim3d(0, 100)
    ax.grid(False)
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    ax.set_xlabel('AUC')
    ax.set_ylabel('log(E)')
    ax.set_zlabel('log(intensity FC')
    plt.savefig('CCRCCGlyco/3d_diagnostic_ions.png')
    plt.clf()

    print('making histo')
    peakData = peakData[peakData['alpha'] == 1]
    # peakData.to_csv('found_diagnostic_peaks.tsv', sep='\t')
    ax = sns.histplot(x='adjusted_mass', data=peakData, binwidth=0.01, binrange=(100, 800), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(100, 800)
    plt.ylim(000, 1000)
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 1:
            print(round(x, 2))
            ax.text(x + w / 2, h + 50, str(round(x, 2)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('CCRCCGlyco/diagnostic_ions_histo.png')
    plt.clf()

    print('making histo zoom')
    ax = sns.histplot(x='adjusted_mass', data=peakData, binwidth=0.01, binrange=(100, 400), log_scale=(False, False),
                      linewidth=1)
    plt.xlim(100, 400)
    plt.ylim(0, 1000)
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 1:
            print(round(x, 2))
            ax.text(x + w / 2, h + 50, str(round(x, 2)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('CCRCCGlyco/diagnostic_ions_histo_zoom.png')
    plt.clf()


def makeGlycoDiagnosticIons2DScatter():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\CCRCCGlyco\\global.diagmine.tsv",
        sep='\t')
    peakData = peakData[peakData['ion_type'] == 'diagnostic']
    peakData.fillna(0)
    peakData['p_value'] = peakData['p_value'] + 1e-300
    peakData['log_p_value'] = -1 * np.log10(peakData['p_value'])
    peakData['avg_int_diff'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int'] - \
                               peakData['prop_unmod_spectra'] * peakData['unmod_spectra_int']
    peakData['avg_int'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int']
    peakData['color'] = peakData.apply(multiSignifMarker, axis=1)
    peakData['alpha'] = peakData.apply(multiSignifMarkerAlpha, axis=1)
    peakData['alpha_color'] = peakData.apply(multiSignifMarkerAlphaColor, axis=1)

    fig = plt.figure()
    print('making scatter')

    colormap = matplotlib.cm.viridis
    divnorm = colors.TwoSlopeNorm(vmin=min(peakData['log_p_value']),
                                  vcenter=0.05,
                                  vmax=max(peakData['log_p_value']))

    ax = plt.scatter(list(peakData['avg_int_diff']), list(peakData['auc']),
                     c=list(peakData['alpha_color']), s=1,
                     cmap=plt.get_cmap('viridis'))
    plt.axvline(5.0, color='black', linestyle='dashed')
    plt.axvline(-5.0, color='black', linestyle='dashed')
    plt.axhline(0.55, color='black', linestyle='dashed')
    plt.axhline(0.45, color='black', linestyle='dashed')
    plt.xlabel('avg. intensity diff.')
    plt.ylabel('auc')
    plt.text(40, 0.41, 'auc = 0.45')
    plt.text(40, 0.58, 'auc = 0.55')
    plt.text(5, 0.2, 'diff. = 5.0', rotation=90)
    plt.text(-11, 0.6, 'diff. = -5.0', rotation=90)
    # sns.rugplot(data=peakData, x='avg_int_diff', y='auc', hue='alpha_color', lw=0.1, alpha=.005)

    plt.xlim(-25, 80)
    plt.ylim(0, 1)

    # list(peakData['log_p_value']),

    # ax = sns.scatterplot(data=peakData, x='avg_int_diff', y='auc', hue='log_p_value',
    #                     size=0.1, alpha=0.1, palette='viridis', linewidth=0, legend=False)
    # fig.colorbar(ax)
    # ax.set_xlim(-100, 100)
    # ax.set_ylim3d(0, 400)
    # ax.set_zlim3d(0, 100)
    # ax.grid(False)
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.set_xlabel('AUC')
    # ax.set_ylabel('log(E)')
    # ax.set_zlabel('avg. intensity diff.')
    plt.savefig('CCRCCGlyco/2d_diagnostic_ions.png')
    plt.clf()


def makeCysProbeScatter():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\Research\\KeriBackus\\probe2\\global.diagmine.tsv",
        sep='\t')
    peakData = peakData[peakData['ion_type'] == 'diagnostic']
    # peakData = peakData[peakData['peak_apex'] == '861.4524']
    peakData.fillna(0)
    peakData['p_value'] = peakData['p_value'] + 1e-200
    peakData['log_p_value'] = -1 * np.log10(peakData['p_value'])
    peakData['log_fold_change'] = np.log2((peakData['prop_mod_spectra'] * peakData['mod_spectra_int'] + 1) - \
                                          (peakData['prop_unmod_spectra'] * peakData['unmod_spectra_int'] + 1))
    # peakData['avg_int'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int']
    peakData['color'] = peakData.apply(multiSignifMarkerCp, axis=1)
    # peakData['alpha'] = peakData.apply(multiSignifMarkerAlpha, axis=1)
    # peakData['alpha_color'] = peakData.apply(multiSignifMarkerAlphaColor, axis=1)

    fig = plt.figure()
    print('making scatter')

    # colormap = matplotlib.cm.viridis
    # divnorm = colors.TwoSlopeNorm(vmin=min(peakData['log_p_value']),
    #                              vcenter=0.05,
    #                              vmax=max(peakData['log_p_value']))

    # ax = plt.scatter(list(peakData['log_fold_change']), list(peakData['log_p_value']),
    #                 c=list(peakData['alpha_color']), s=1,
    #                 cmap=plt.get_cmap('viridis'))
    ax = plt.scatter(list(peakData['log_fold_change']), list(peakData['log_p_value']), color=list(peakData['color']),
                     s=15, alpha=0.5)
    plt.title("Cys probe diagnostic ions")
    plt.axvline(1.6, color='black', linestyle='dashed')
    # plt.axvline(-5.0, color='black', linestyle='dashed')
    plt.axhline(2, color='black', linestyle='dashed')
    # plt.axhline(0.45, color='black', linestyle='dashed')
    plt.xlabel('log2(FC)')
    plt.ylabel('-log10(E)')

    # plt.text(40, 0.41, 'auc = 0.45')
    # plt.text(40, 0.58, 'auc = 0.55')
    # plt.text(5, 0.2, 'diff. = 5.0', rotation=90)
    # plt.text(-11, 0.6, 'diff. = -5.0', rotation=90)
    # sns.rugplot(data=peakData, x='avg_int_diff', y='auc', hue='alpha_color', lw=0.1, alpha=.005)

    plt.xlim(-2.5, 7.5)
    # plt.ylim(0, 1)

    # list(peakData['log_p_value']),

    # ax = sns.scatterplot(data=peakData, x='avg_int_diff', y='auc', hue='log_p_value',
    #                     size=0.1, alpha=0.1, palette='viridis', linewidth=0, legend=False)
    # fig.colorbar(ax)
    # ax.set_xlim(-100, 100)
    # ax.set_ylim3d(0, 400)
    # ax.set_zlim3d(0, 100)
    # ax.grid(False)
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.set_xlabel('AUC')
    # ax.set_ylabel('log(E)')
    # ax.set_zlabel('avg. intensity diff.')
    plt.savefig('CysProbe/2d_diagnostic_ions.png')
    plt.clf()


def makeMouseKidneyGlycoDiagnosticIons2DScatter():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\MouseKidneyGlyco\\global.diagmine.tsv",
        sep='\t')
    peakData = peakData[peakData['ion_type'] == 'diagnostic']
    peakData.fillna(0)
    peakData['p_value'] = peakData['p_value'] + 1e-300
    peakData['log_p_value'] = -1 * np.log10(peakData['p_value'])
    peakData['avg_int_diff'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int'] - \
                               peakData['prop_unmod_spectra'] * peakData['unmod_spectra_int']
    peakData['avg_int'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int']
    peakData['color'] = peakData.apply(multiSignifMarker, axis=1)
    peakData['alpha'] = peakData.apply(multiSignifMarkerAlpha, axis=1)
    peakData['alpha_color'] = peakData.apply(multiSignifMarkerAlphaColor, axis=1)

    fig = plt.figure()
    print('making scatter')

    colormap = matplotlib.cm.viridis
    divnorm = colors.TwoSlopeNorm(vmin=min(peakData['log_p_value']),
                                  vcenter=0.05,
                                  vmax=max(peakData['log_p_value']))

    ax = plt.scatter(list(peakData['avg_int_diff']), list(peakData['auc']),
                     c=list(peakData['alpha_color']), s=1,
                     cmap=plt.get_cmap('viridis'))
    plt.axvline(5.0, color='black', linestyle='dashed')
    plt.axvline(-5.0, color='black', linestyle='dashed')
    plt.axhline(0.55, color='black', linestyle='dashed')
    plt.axhline(0.45, color='black', linestyle='dashed')
    plt.xlabel('avg. intensity diff.')
    plt.ylabel('auc')
    plt.text(40, 0.41, 'auc = 0.45')
    plt.text(40, 0.58, 'auc = 0.55')
    plt.text(5, 0.2, 'diff. = 5.0', rotation=90)
    plt.text(-11, 0.6, 'diff. = -5.0', rotation=90)
    # sns.rugplot(data=peakData, x='avg_int_diff', y='auc', hue='alpha_color', lw=0.1, alpha=.005)

    plt.xlim(-25, 80)
    plt.ylim(0, 1)

    # list(peakData['log_p_value']),

    # ax = sns.scatterplot(data=peakData, x='avg_int_diff', y='auc', hue='log_p_value',
    #                     size=0.1, alpha=0.1, palette='viridis', linewidth=0, legend=False)
    # fig.colorbar(ax)
    # ax.set_xlim(-100, 100)
    # ax.set_ylim3d(0, 400)
    # ax.set_zlim3d(0, 100)
    # ax.grid(False)
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.set_xlabel('AUC')
    # ax.set_ylabel('log(E)')
    # ax.set_zlabel('avg. intensity diff.')
    plt.savefig('MouseKidneyGlyco/2d_diagnostic_ions.png')
    plt.clf()


def makeGlycoYIonsScatter():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\CCRCCGlyco\\global.diagmine.tsv",
        sep='\t')
    peakData = peakData[peakData['ion_type'] == 'Y']
    peakData.fillna(0)
    peakData['p_value'] = peakData['p_value'] + 1e-200
    peakData['log_p_value'] = -1 * np.log(peakData['p_value'])
    peakData['avg_int'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int']
    peakData['avg_int_diff'] = abs(
        peakData['prop_mod_spectra'] * peakData['mod_spectra_int'] -
        peakData['prop_unmod_spectra'] * peakData['unmod_spectra_int'])
    peakData['color'] = peakData.apply(multiSignifMarker, axis=1)
    peakData['alpha'] = peakData.apply(multiSignifMarkerAlpha, axis=1)
    peakData['alpha_color'] = peakData.apply(multiSignifMarkerAlphaColor, axis=1)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(list(peakData['auc']), list(peakData['log_p_value']), list(peakData['avg_int_diff']),
               c=list(peakData['avg_int']), s=list(peakData['avg_int']),
               cmap=plt.get_cmap('viridis'))
    # cmap=truncate_colormap(plt.get_cmap('viridis'), 0, max(peakData['avg_int']) / 100, 256))
    ax.set_xlim3d(0.4, 1)
    ax.set_ylim3d(0, 400)
    ax.set_zlim3d(0, 65)
    ax.grid(False)
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    ax.set_xlabel('AUC')
    ax.set_ylabel('log(E)')
    ax.set_zlabel('avg.intensity diff.')
    plt.savefig('CCRCCGlyco/3d_CapY_ions.png')
    plt.clf()

    peakData = peakData[peakData['alpha'] == 1]
    peakData = peakData[peakData['adjusted_mass'] > -1]
    ax = sns.histplot(x='adjusted_mass', data=peakData, binwidth=1, log_scale=(False, False), linewidth=1)
    plt.xlim(0, 500)
    plt.ylim(0, 1000)
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 5:
            ax.text(x + w / 2, h + 50, str(round(x)), ha='center', va='center', size=8, rotation='vertical')
    plt.savefig('CCRCCGlyco/CapY_ions_histo.png')


def makeGlycoYIons2DScatter():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\CCRCCGlyco\\global.diagmine.tsv",
        sep='\t')
    peakData = peakData[peakData['ion_type'] == 'Y']
    peakData.fillna(0)
    peakData['p_value'] = peakData['p_value'] + 1e-300
    peakData['log_p_value'] = -1 * np.log10(peakData['p_value'])
    peakData['avg_int_diff'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int'] - \
                               peakData['prop_unmod_spectra'] * peakData['unmod_spectra_int']
    peakData['avg_int'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int']
    peakData['color'] = peakData.apply(multiSignifMarker, axis=1)
    peakData['alpha'] = peakData.apply(multiSignifMarkerAlpha, axis=1)
    peakData['alpha_color'] = peakData.apply(multiSignifMarkerAlphaColor, axis=1)

    fig = plt.figure()
    print('making scatter')

    colormap = matplotlib.cm.viridis
    divnorm = colors.TwoSlopeNorm(vmin=min(peakData['log_p_value']),
                                  vcenter=0.05,
                                  vmax=max(peakData['log_p_value']))

    ax = plt.scatter(list(peakData['avg_int_diff']), list(peakData['auc']),
                     c=list(peakData['alpha_color']), s=1,
                     cmap=plt.get_cmap('viridis'))
    plt.axvline(5.0, color='black', linestyle='dashed')
    plt.axvline(-5.0, color='black', linestyle='dashed')
    plt.axhline(0.55, color='black', linestyle='dashed')
    plt.axhline(0.45, color='black', linestyle='dashed')
    plt.xlabel('avg. intensity diff.')
    plt.ylabel('auc')
    plt.text(40, 0.41, 'auc = 0.45')
    plt.text(40, 0.58, 'auc = 0.55')
    plt.text(5, 0.2, 'diff. = 5.0', rotation=90)
    plt.text(-11, 0.6, 'diff. = -5.0', rotation=90)
    # sns.rugplot(data=peakData, x='avg_int_diff', y='auc', hue='alpha_color', lw=0.1, alpha=.005)

    # list(peakData['log_p_value']),

    # ax = sns.scatterplot(data=peakData, x='avg_int_diff', y='auc', hue='log_p_value',
    #                     size=0.1, alpha=0.1, palette='viridis', linewidth=0, legend=False)
    # fig.colorbar(ax)
    plt.xlim(-25, 80)
    plt.ylim(0, 1)
    # ax.set_ylim3d(0, 400)
    # ax.set_zlim3d(0, 100)
    # ax.grid(False)
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.set_xlabel('AUC')
    # ax.set_ylabel('log(E)')
    # ax.set_zlabel('avg. intensity diff.')
    plt.savefig('CCRCCGlyco/2d_capY_ions.png')
    plt.clf()


def makeMouseKidneyGlycoYIons2DScatter():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\MouseKidneyGlyco\\global.diagmine.tsv",
        sep='\t')
    peakData = peakData[peakData['ion_type'] == 'Y']
    peakData.fillna(0)
    peakData['p_value'] = peakData['p_value'] + 1e-300
    peakData['log_p_value'] = -1 * np.log10(peakData['p_value'])
    peakData['avg_int_diff'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int'] - \
                               peakData['prop_unmod_spectra'] * peakData['unmod_spectra_int']
    peakData['avg_int'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int']
    peakData['color'] = peakData.apply(multiSignifMarker, axis=1)
    peakData['alpha'] = peakData.apply(multiSignifMarkerAlpha, axis=1)
    peakData['alpha_color'] = peakData.apply(multiSignifMarkerAlphaColor, axis=1)

    fig = plt.figure()
    print('making scatter')

    colormap = matplotlib.cm.viridis
    divnorm = colors.TwoSlopeNorm(vmin=min(peakData['log_p_value']),
                                  vcenter=0.05,
                                  vmax=max(peakData['log_p_value']))

    ax = plt.scatter(list(peakData['avg_int_diff']), list(peakData['auc']),
                     c=list(peakData['alpha_color']), s=1,
                     cmap=plt.get_cmap('viridis'))
    plt.axvline(5.0, color='black', linestyle='dashed')
    plt.axvline(-5.0, color='black', linestyle='dashed')
    plt.axhline(0.55, color='black', linestyle='dashed')
    plt.axhline(0.45, color='black', linestyle='dashed')
    plt.xlabel('avg. intensity diff.')
    plt.ylabel('auc')
    plt.text(40, 0.41, 'auc = 0.45')
    plt.text(40, 0.58, 'auc = 0.55')
    plt.text(5, 0.2, 'diff. = 5.0', rotation=90)
    plt.text(-11, 0.6, 'diff. = -5.0', rotation=90)
    # sns.rugplot(data=peakData, x='avg_int_diff', y='auc', hue='alpha_color', lw=0.1, alpha=.005)

    # list(peakData['log_p_value']),

    # ax = sns.scatterplot(data=peakData, x='avg_int_diff', y='auc', hue='log_p_value',
    #                     size=0.1, alpha=0.1, palette='viridis', linewidth=0, legend=False)
    # fig.colorbar(ax)
    plt.xlim(-25, 80)
    plt.ylim(0, 1)
    # ax.set_ylim3d(0, 400)
    # ax.set_zlim3d(0, 100)
    # ax.grid(False)
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.set_xlabel('AUC')
    # ax.set_ylabel('log(E)')
    # ax.set_zlabel('avg. intensity diff.')
    plt.savefig('MouseKidneyGlyco/2d_capY_ions.png')
    plt.clf()


def makeGlyco_bIons2DScatter():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\CCRCCGlyco\\global.diagmine.tsv",
        sep='\t')
    peakData = peakData[peakData['ion_type'] == 'b']
    peakData.fillna(0)
    peakData['p_value'] = peakData['p_value'] + 1e-300
    peakData['log_p_value'] = -1 * np.log10(peakData['p_value'])
    peakData['avg_int_diff'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int'] - \
                               peakData['prop_unmod_spectra'] * peakData['unmod_spectra_int']
    peakData['avg_int'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int']
    peakData['color'] = peakData.apply(multiSignifMarker, axis=1)
    peakData['alpha'] = peakData.apply(multiSignifMarkerAlpha, axis=1)
    peakData['alpha_color'] = peakData.apply(multiSignifMarkerAlphaColor, axis=1)

    fig = plt.figure()
    print('making scatter')

    colormap = matplotlib.cm.viridis
    divnorm = colors.TwoSlopeNorm(vmin=min(peakData['log_p_value']),
                                  vcenter=0.05,
                                  vmax=max(peakData['log_p_value']))

    ax = plt.scatter(list(peakData['avg_int_diff']), list(peakData['auc']),
                     c=list(peakData['alpha_color']), s=1,
                     cmap=plt.get_cmap('viridis'))
    plt.axvline(5.0, color='black', linestyle='dashed')
    plt.axvline(-5.0, color='black', linestyle='dashed')
    plt.axhline(0.55, color='black', linestyle='dashed')
    plt.axhline(0.45, color='black', linestyle='dashed')
    plt.xlabel('avg. intensity diff.')
    plt.ylabel('auc')
    plt.text(40, 0.41, 'auc = 0.45')
    plt.text(40, 0.58, 'auc = 0.55')
    plt.text(5, 0.2, 'diff. = 5.0', rotation=90)
    plt.text(-11, 0.6, 'diff. = -5.0', rotation=90)
    # sns.rugplot(data=peakData, x='avg_int_diff', y='auc', hue='alpha_color', lw=0.1, alpha=.005)

    # list(peakData['log_p_value']),

    # ax = sns.scatterplot(data=peakData, x='avg_int_diff', y='auc', hue='log_p_value',
    #                     size=0.1, alpha=0.1, palette='viridis', linewidth=0, legend=False)
    # fig.colorbar(ax)
    plt.xlim(-25, 80)
    plt.ylim(0, 1)
    # ax.set_ylim3d(0, 400)
    # ax.set_zlim3d(0, 100)
    # ax.grid(False)
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.set_xlabel('AUC')
    # ax.set_ylabel('log(E)')
    # ax.set_zlabel('avg. intensity diff.')
    plt.savefig('CCRCCGlyco/2d_y_ions.png')
    plt.clf()


def makeGlyco_yIons2DScatter():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\CCRCCGlyco\\global.diagmine.tsv",
        sep='\t')
    peakData = peakData[peakData['ion_type'] == 'y']
    peakData.fillna(0)
    peakData['p_value'] = peakData['p_value'] + 1e-300
    peakData['log_p_value'] = -1 * np.log10(peakData['p_value'])
    peakData['avg_int_diff'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int'] - \
                               peakData['prop_unmod_spectra'] * peakData['unmod_spectra_int']
    peakData['avg_int'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int']
    peakData['color'] = peakData.apply(multiSignifMarker, axis=1)
    peakData['alpha'] = peakData.apply(multiSignifMarkerAlpha, axis=1)
    peakData['alpha_color'] = peakData.apply(multiSignifMarkerAlphaColor, axis=1)

    fig = plt.figure()
    print('making scatter')

    colormap = matplotlib.cm.viridis
    divnorm = colors.TwoSlopeNorm(vmin=min(peakData['log_p_value']),
                                  vcenter=0.05,
                                  vmax=max(peakData['log_p_value']))

    ax = plt.scatter(list(peakData['avg_int_diff']), list(peakData['auc']),
                     c=list(peakData['alpha_color']), s=1,
                     cmap=plt.get_cmap('viridis'))
    plt.axvline(5.0, color='black', linestyle='dashed')
    plt.axvline(-5.0, color='black', linestyle='dashed')
    plt.axhline(0.55, color='black', linestyle='dashed')
    plt.axhline(0.45, color='black', linestyle='dashed')
    plt.xlabel('avg. intensity diff.')
    plt.ylabel('auc')
    plt.text(40, 0.41, 'auc = 0.45')
    plt.text(40, 0.58, 'auc = 0.55')
    plt.text(5, 0.2, 'diff. = 5.0', rotation=90)
    plt.text(-11, 0.6, 'diff. = -5.0', rotation=90)
    # sns.rugplot(data=peakData, x='avg_int_diff', y='auc', hue='alpha_color', lw=0.1, alpha=.005)

    # list(peakData['log_p_value']),

    # ax = sns.scatterplot(data=peakData, x='avg_int_diff', y='auc', hue='log_p_value',
    #                     size=0.1, alpha=0.1, palette='viridis', linewidth=0, legend=False)
    # fig.colorbar(ax)
    plt.xlim(-25, 80)
    plt.ylim(0, 1)
    # ax.set_ylim3d(0, 400)
    # ax.set_zlim3d(0, 100)
    # ax.grid(False)
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.set_xlabel('AUC')
    # ax.set_ylabel('log(E)')
    # ax.set_zlabel('avg. intensity diff.')
    plt.savefig('CCRCCGlyco/2d_b_ions.png')
    plt.clf()


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def makeGlyco_yIonsScatter():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\CCRCCGlyco\\global.diagmine.tsv",
        sep='\t')
    peakData = peakData[peakData['ion_type'] == 'y']
    peakData.fillna(0)
    peakData['p_value'] = peakData['p_value'] + 1e-200
    peakData['log_p_value'] = -1 * np.log(peakData['p_value'])
    peakData['avg_int'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int']
    peakData['avg_int_diff'] = abs(
        peakData['prop_mod_spectra'] * peakData['mod_spectra_int'] -
        peakData['prop_unmod_spectra'] * peakData['unmod_spectra_int'])
    peakData['color'] = peakData.apply(multiSignifMarker, axis=1)
    peakData['alpha'] = peakData.apply(multiSignifMarkerAlpha, axis=1)
    peakData['alpha_color'] = peakData.apply(multiSignifMarkerAlphaColor, axis=1)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(list(peakData['auc']), list(peakData['log_p_value']), list(peakData['avg_int_diff']),
               c=list(peakData['avg_int']), s=4,
               cmap=truncate_colormap(plt.get_cmap('viridis'), 0, max(peakData['avg_int']) / 100, 256))
    ax.set_xlim3d(0.2, 0.8)
    ax.set_ylim3d(0, 200)
    ax.set_zlim3d(0, 10)
    ax.grid(False)
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    ax.set_xlabel('AUC')
    ax.set_ylabel('log(E)')
    ax.set_zlabel('avg.intensity diff.')
    plt.savefig('CCRCCGlyco/3d_y_ions.png')
    plt.clf()

    peakData = peakData[peakData['alpha'] == 1]
    # peakData = peakData[peakData['adjusted_mass'] > -1]
    ax = sns.histplot(x='adjusted_mass', data=peakData, binwidth=0.001, log_scale=(False, True), linewidth=1,
                      element='step')
    for p in ax.patches:
        x, w, h = p.get_x(), p.get_width(), p.get_height()
        if h > 1:
            ax.text(x + w / 2, h, str(round(x)), ha='center', va='center', size=8)
    plt.savefig('CCRCCGlyco/y_ions_histo.png')


def makeGlyco_bIonsScatter():
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\CCRCCGlyco\\global.diagmine.tsv",
        sep='\t')
    peakData = peakData[peakData['ion_type'] == 'b']
    peakData.fillna(0)
    peakData['p_value'] = peakData['p_value'] + 1e-200
    peakData['log_p_value'] = -1 * np.log(peakData['p_value'])
    peakData['avg_int'] = peakData['prop_mod_spectra'] * peakData['mod_spectra_int']
    peakData['avg_int_diff'] = abs(
        peakData['prop_mod_spectra'] * peakData['mod_spectra_int'] -
        peakData['prop_unmod_spectra'] * peakData['unmod_spectra_int'])
    peakData['color'] = peakData.apply(multiSignifMarker, axis=1)
    peakData['alpha'] = peakData.apply(multiSignifMarkerAlpha, axis=1)
    peakData['alpha_color'] = peakData.apply(multiSignifMarkerAlphaColor, axis=1)

    fig = plt.figure()
    ax = Axes3D(fig)
    sns.set_style('white')
    ax.scatter(list(peakData['auc']), list(peakData['log_p_value']), list(peakData['avg_int_diff']),
               c=list(peakData['avg_int']), s=4,
               cmap=truncate_colormap(plt.get_cmap('viridis'), 0, max(peakData['avg_int']) / 100, 256))
    ax.set_xlim3d(0.2, 0.8)
    ax.set_ylim3d(0, 200)
    ax.set_zlim3d(0, 10)
    ax.grid(False)
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xlabel('AUC')
    ax.set_ylabel('log(E)')
    ax.set_zlabel('avg.intensity diff.')
    plt.savefig('CCRCCGlyco/3d_b_ions.png')
    plt.clf()

    peakData = peakData[peakData['alpha'] == 1]
    # peakData = peakData[peakData['adjusted_mass'] > -1]
    sns.histplot(x='adjusted_mass', data=peakData, binwidth=0.01, log_scale=(False, True), linewidth=1)

    plt.savefig('CCRCCGlyco/b_ions_histo.png')


def makeBYScatter():
    """
    IMPORTANT: run extractPairsBALines.py and (now BY lines) manually label the isValid column for inclusion in scatter
    """
    peakData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ProteomeTools\\PTScatterInput.tsv",
        sep='\t')
    print(peakData.columns)
    peakData2 = peakData[peakData['isValid'] == 1]
    # peakData2['bAvgInt'] = jitter(peakData2['bAvgInt'], 0)
    # peakData2['yAvgInt'] = jitter(peakData2['yAvgInt'], 0)
    ax = sns.scatterplot(x='bAvgInt', y='yAvgInt', data=peakData2)
    for _, row in peakData2.iterrows():
        row['yAvgInt']
        ax.text(x=row['bAvgInt'], y=row['yAvgInt'], s=round(row['yMass'], 2))
    plt.xlim(3, 15)
    plt.ylim(3, 35)
    plt.savefig('ProteomeTools/by_scatter.png')


def jitter(values, j):
    return values + np.random.normal(j, 1, values.shape)


def makeVennDiagram():
    venn2(subsets=(1, 6, 15), set_labels=('PTPTMs', 'PTMS'), set_colors=('#F8766D', '#00BFC4'), alpha=0.7)
    venn2_circles(subsets=(1, 6, 15))
    plt.savefig('ProteomeTools/diag_ions_venn.pdf')
    plt.clf()


def makeBoxPlots():
    keepDeltaMasses = []
    spectraData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ProteomeTools\\SyntheticMods.rawglyco",
        sep='\t')
    spectraData['file'] = spectraData.apply(multiFileMarker, axis=1)
    spectraData['isValid'] = spectraData.apply(isValidDMassMarker, axis=1)

    """
    Make boxplots for citrullination
    """
    spectraData2 = spectraData[spectraData['file'].isin(['Rmod_Citrullin', 'Unmod'])]
    spectraData2 = spectraData2[spectraData2['isValid'] == True]
    massCols = ['ox_130.0975_intensity', 'ox_115.0866_intensity', 'ox_113.0710_intensity']
    keepCols = [col for col in spectraData2.columns if col in massCols + ['Spectrum', 'file']]
    spectraData2 = spectraData2[keepCols]
    spectraData2 = pd.melt(spectraData2, id_vars=['Spectrum', 'file'], value_vars=massCols)

    ax = sns.boxenplot(x='variable', y='value', order=massCols, data=spectraData2,
                       hue='file', hue_order=['Rmod_Citrullin', 'Unmod'])
    sns.despine()
    ax.set_yscale('log')
    plt.savefig('ProteomeTools/citrullin_boxplot.png')
    plt.clf()

    """
    Make boxplots for hydroxyisobutyryl
    """
    spectraData2 = spectraData[spectraData['file'].isin(['Kmod_Hydroxyisobutyryl', 'Unmod'])]
    spectraData2 = spectraData2[spectraData2['isValid'] == True]
    massCols = ['ox_170.1176_intensity', 'ox_187.1440_intensity']
    keepCols = [col for col in spectraData2.columns if col in massCols + ['Spectrum', 'file']]
    spectraData2 = spectraData2[keepCols]
    spectraData2 = pd.melt(spectraData2, id_vars=['Spectrum', 'file'], value_vars=massCols)

    ax = sns.boxenplot(x='variable', y='value', data=spectraData2, hue='file',
                       hue_order=['Kmod_Hydroxyisobutyryl', 'Unmod'])
    ax.set_yscale('log')
    sns.despine()
    plt.savefig('ProteomeTools/hydroxyisobutyryl_boxplot.png')
    plt.clf()

    """
    Make boxplots for glutaryl
    """
    spectraData2 = spectraData[spectraData['file'].isin(['Kmod_Glutaryl', 'Unmod'])]
    # spectraData2.to_csv('glutaryl2.tsv', sep='\t')
    spectraData2 = spectraData2[spectraData2['isValid'] == True]
    # print(spectraData2.file.unique())
    massCols = ['ox_182.1176_intensity', 'ox_198.1124_intensity']
    keepCols = [col for col in spectraData2.columns if col in massCols + ['Spectrum', 'file']]
    spectraData2 = spectraData2[keepCols]
    spectraData2 = pd.melt(spectraData2, id_vars=['Spectrum', 'file'], value_vars=massCols)
    # spectraData2.to_csv('glutaryl.tsv', sep='\t')
    # for index, row in spectraData2.iterrows():
    #    print(row['Spectrum'], row['file'], row['variable'], row['value'])
    ax = sns.boxenplot(x='variable', y='value', data=spectraData2, hue='file', hue_order=['Kmod_Glutaryl', 'Unmod'])
    sns.despine()
    ax.set_yscale('log')
    plt.savefig('ProteomeTools/glutaryl_boxplot.png')
    plt.clf()

    """
    Make boxplots for biotinyl
    """
    spectraData2 = spectraData[spectraData['file'].isin(['Kmod_Biotinyl', 'Unmod'])]
    spectraData2 = spectraData2[spectraData2['isValid'] == True]
    massCols = ['ox_310.1580_intensity', 'ox_326.1526_intensity', 'ox_227.0848_intensity', 'ox_243.0796_intensity']
    keepCols = [col for col in spectraData2.columns if col in massCols + ['Spectrum', 'file']]
    spectraData2 = spectraData2[keepCols]
    spectraData2 = pd.melt(spectraData2, id_vars=['Spectrum', 'file'], value_vars=massCols)

    ax = sns.boxenplot(x='variable', y='value', data=spectraData2, hue='file', hue_order=['Kmod_Biotinyl', 'Unmod'])
    sns.despine()
    ax.set_yscale('log')
    plt.savefig('ProteomeTools/biotinylyl_boxplot.png')
    plt.clf()


def makeCombinedBoxPlot():
    keepDeltaMasses = []
    spectraData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of "
        "Michigan)\\DiagMiningPaper\\Data\\ProteomeTools\\SyntheticMods.rawglyco",
        sep='\t')
    spectraData['file'] = spectraData.apply(multiFileMarker, axis=1)
    spectraData['isValid'] = spectraData.apply(isValidDMassMarker, axis=1)

    """
    Make combined boxenplot
    """
    fig, axes = plt.subplots(1, 4, gridspec_kw={'width_ratios': [3, 2, 2, 4]}, figsize=(11, 3), sharey=True)
    flierprops = dict(marker='o', markerfacecolor='None', markersize=1, alpha=0.1)
    sns.set()

    """
    Make boxplots for citrullination
    """
    spectraData2 = spectraData[spectraData['file'].isin(['Rmod_Citrullin', 'Unmod'])]
    print(spectraData2.value_counts('file'))
    spectraData2 = spectraData2[spectraData2['isValid'] == True]
    spectraData2['color'] = spectraData2.apply(normalizeAlphaToDist, axis=1)
    massCols = ['ox_130.0975_intensity', 'ox_115.0866_intensity', 'ox_113.0710_intensity']
    labels = [col.split('_')[1] for col in massCols]
    keepCols = [col for col in spectraData2.columns if col in massCols + ['Spectrum', 'file']]
    spectraData2 = spectraData2[keepCols]
    spectraData2 = pd.melt(spectraData2, id_vars=['Spectrum', 'file'], value_vars=massCols)
    sns.boxplot(ax=axes[0], x='variable',
                order=['ox_130.0975_intensity', 'ox_115.0866_intensity', 'ox_113.0710_intensity'], y='value',
                data=spectraData2, hue='file',
                hue_order=['Rmod_Citrullin', 'Unmod'],
                showfliers=False, boxprops={'facecolor': 'None'})
    sns.stripplot(ax=axes[0], x='variable',
                  order=['ox_130.0975_intensity', 'ox_115.0866_intensity', 'ox_113.0710_intensity'], y='value',
                  data=spectraData2, hue='file', dodge=True,
                  hue_order=['Rmod_Citrullin', 'Unmod'],
                  linewidth=0, alpha=[0.2, 0.005], c=list(spectraData2['color']))
    # flierprops=flierprops)
    axes[0].set_xticklabels(labels, rotation=-25, ha='left')

    """
    Make boxplots for hydroxyisobutyryl
    """
    spectraData2 = spectraData[spectraData['file'].isin(['Kmod_Hydroxyisobutyryl', 'Unmod'])]
    print(spectraData2.file.unique())
    spectraData2.to_csv('tmp.tsv', '\t')
    spectraData2 = spectraData2[spectraData2['isValid'] == True]
    massCols = ['ox_170.1176_intensity', 'ox_187.1440_intensity']
    labels = [col.split('_')[1] for col in massCols]
    keepCols = [col for col in spectraData2.columns if col in massCols + ['Spectrum', 'file']]
    spectraData2 = spectraData2[keepCols]
    print(spectraData2.file.unique())
    spectraData2 = pd.melt(spectraData2, id_vars=['Spectrum', 'file'], value_vars=massCols)
    sns.boxplot(ax=axes[1], x='variable',
                order=['ox_170.1176_intensity', 'ox_187.1440_intensity'], y='value',
                data=spectraData2, hue='file',
                hue_order=['Kmod_Hydroxyisobutyryl', 'Unmod'],
                showfliers=False, boxprops={'facecolor': 'None'})
    sns.stripplot(ax=axes[1], x='variable', order=['ox_170.1176_intensity', 'ox_187.1440_intensity'], y='value',
                  data=spectraData2, hue='file', dodge=True,
                  hue_order=['Kmod_Hydroxyisobutyryl', 'Unmod'],
                  linewidth=0, alpha=0.005, )
    # flierprops=flierprops)
    axes[1].set_xticklabels(labels, rotation=-25, ha='left', fontsize=10)
    axes[1].spines['left'].set_visible(False)

    """
    Make boxplots for glutaryl
    """
    flierprops = dict(marker='o', markerfacecolor='None', markersize=1, alpha=0.01)

    spectraData2 = spectraData[spectraData['file'].isin(['Kmod_Glutaryl', 'Unmod'])]
    # spectraData2.to_csv('glutaryl2.tsv', sep='\t')
    spectraData2 = spectraData2[spectraData2['isValid'] == True]
    # print(spectraData2.file.unique())
    massCols = ['ox_182.1176_intensity', 'ox_198.1124_intensity']
    labels = [col.split('_')[1] for col in massCols]
    keepCols = [col for col in spectraData2.columns if col in massCols + ['Spectrum', 'file']]
    spectraData2 = spectraData2[keepCols]
    spectraData2 = pd.melt(spectraData2, id_vars=['Spectrum', 'file'], value_vars=massCols)
    print(spectraData2.columns)
    # spectraData2.to_csv('glutaryl.tsv', sep='\t')
    # for index, row in spectraData2.iterrows():
    #    print(row['Spectrum'], row['file'], row['variable'], row['value'])
    sns.boxplot(ax=axes[2], x='variable',
                order=['ox_182.1176_intensity', 'ox_198.1124_intensity'], y='value',
                data=spectraData2, hue='file',
                hue_order=['Kmod_Glutaryl', 'Unmod'],
                showfliers=False, boxprops={'facecolor': 'None'})
    sns.stripplot(ax=axes[2], x='variable', order=['ox_182.1176_intensity', 'ox_198.1124_intensity'],
                  y='value', data=spectraData2, hue='file', hue_order=['Kmod_Glutaryl', 'Unmod'],
                  linewidth=0, alpha=0.005,
                  dodge=True)
    # saturation=0.75)
    # flierprops=flierprops)
    axes[2].set_xticklabels(labels, rotation=-25, ha='left', fontsize=10)
    axes[2].spines['left'].set_visible(False)

    """
    Make boxplots for biotinyl
    """
    flierprops = dict(marker='o', markerfacecolor='None', markersize=1, alpha=0.1)

    spectraData2 = spectraData[spectraData['file'].isin(['Kmod_Biotinyl', 'Unmod'])]
    spectraData2 = spectraData2[spectraData2['isValid'] == True]
    massCols = ['ox_310.1580_intensity', 'ox_326.1526_intensity', 'ox_227.0848_intensity', 'ox_243.0796_intensity']
    labels = [col.split('_')[1] for col in massCols]
    keepCols = [col for col in spectraData2.columns if col in massCols + ['Spectrum', 'file']]
    spectraData2 = spectraData2[keepCols]
    spectraData2 = pd.melt(spectraData2, id_vars=['Spectrum', 'file'], value_vars=massCols)

    sns.boxplot(ax=axes[3], x='variable',
                order=['ox_310.1580_intensity', 'ox_326.1526_intensity', 'ox_227.0848_intensity',
                       'ox_243.0796_intensity'], y='value',
                data=spectraData2, hue='file',
                hue_order=['Kmod_Biotinyl', 'Unmod'],
                showfliers=False, boxprops={'facecolor': 'None'})
    sns.stripplot(ax=axes[3], x='variable', y='value',
                  order=['ox_310.1580_intensity', 'ox_326.1526_intensity', 'ox_227.0848_intensity',
                         'ox_243.0796_intensity'],
                  data=spectraData2, hue='file', hue_order=['Kmod_Biotinyl', 'Unmod'],
                  linewidth=0, alpha=0.005,
                  dodge=True)
    # flierprops=flierprops)
    axes[3].set_xticklabels(labels, rotation=-25, ha='left', fontsize=10)
    axes[3].spines['left'].set_visible(False)
    sns.despine()

    for i in range(4):
        axes[i].set(xlabel=None)
        axes[i].legend([], [], frameon=False)
        axes[i].get_xticklabels()[0].set_weight('bold')

    axes[0].set_title('citrullin', y=-0.45, fontsize=12)
    axes[1].set_title('hydroxyisobutyrlyl', y=-0.45, fontsize=12)
    axes[2].set_title('glutaryl', y=-0.45, fontsize=12)
    axes[3].set_title('biotinyl', y=-0.45, fontsize=12)
    axes[1].get_yaxis().set_visible(False)
    axes[2].get_yaxis().set_visible(False)
    axes[3].get_yaxis().set_visible(False)
    axes[0].set(ylabel='intensity')

    plt.subplots_adjust(bottom=0.3)
    plt.savefig('ProteomeTools/combined_boxplot.png')
    plt.clf()


def normalizeAlphaToDist(df):
    if df['file'] == 'Unmod':
        r, g, b = to_rgb('#ff7f0e')
        return (r, g, b, 0.005)
    else:
        r, g, b = to_rgb('#1f77b4')
        return (r, g, b, 0.2)


def makePcaPlots(printMetrics):
    printMetrics = False
    spectraData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\Combined\\combined_rawglyco2.tsv",
        sep='\t')
    spectraData['file'] = spectraData['Spectrum'].str.contains('Liver')
    spectraData['isADPR'] = spectraData['Mass Shift'] > 50
    spectraData['file_isADPR'] = spectraData.apply(multiCategoryMarker, axis=1)

    """
    Writes PCA plots containing unmodified spectra
    """
    sns.set(font_scale=2)
    colors = ['#E62125', '#070001', '#F7941D']
    sns.set_palette(sns.color_palette(colors))
    sns.set_style("white")

    spectraData2 = spectraData
    # spectraData2 = spectraData.loc[(spectraData['isADPR'] == True) & (spectraData['file'] == False)]
    # spectraData2 = spectraData[spectraData['isADPR'] == True]
    spectraData2.reset_index(drop=True, inplace=True)

    """
    # Specific diagnostic ions
    pepRemCols = [col for col in spectraData2.columns if 'ox_5' in col]  # + ['Spectrum', 'Mass Shift']
    pepRemData = spectraData2[pepRemCols]
    
    pca = PCA(n_components=len(pepRemCols))
    principalComponent = pca.fit_transform(pepRemData)
    principalDf = pd.DataFrame(data=principalComponent, columns=['PC1', 'PC2'])
    ax = sns.scatterplot(x='PC1', y='PC2', data=principalDf, alpha=0.1, hue=spectraData2['file_isADPR'])
    plt = ax.get_figure()
    sns.despine()
    plt.savefig('ADPRibose/PCA_diag_specific.png')
    plt.savefig('ADPRibose/PCA_diag_specific.pdf')
    plt.clf()

    if printMetrics == True:
        print("Diag specific:" + str(
            metrics.silhouette_score(principalDf[['PC1', 'PC2']], spectraData2['file_isADPR'], metric='euclidean')))
    """
    # Y masses
    figure(figsize=(10, 10))
    pepRemCols = [col for col in spectraData2.columns if 'Y' in col]  # + ['Spectrum', 'Mass Shift']
    pepRemData = spectraData2[pepRemCols]
    pca = PCA(n_components=len(pepRemCols))
    principalComponent = pca.fit_transform(pepRemData)
    principalDf = pd.DataFrame(data=principalComponent, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    ax = sns.scatterplot(x='PC1', y='PC2', data=principalDf, alpha=0.1, hue=spectraData2['file_isADPR'],
                         rasterized=True)
    plt = ax.get_figure()
    sns.despine()
    plt.savefig('ADPRibose/PCA_Y.png')
    plt.savefig('ADPRibose/PCA_Y.pdf')
    plt.clf()

    if printMetrics == True:
        print("Y: " + str(
            metrics.silhouette_score(principalDf[['PC1', 'PC2']], spectraData2['file_isADPR'], metric='euclidean')))

    # Y masses
    """
    pepRemCols = [col for col in spectraData2.columns if 'Y' in col]  # + ['Spectrum', 'Mass Shift']
    pepRemCols = [col for col in pepRemCols if '60.' not in col]
    pepRemData = spectraData2[pepRemCols]
    pca = PCA(n_components=len(pepRemCols))
    principalComponent = pca.fit_transform(pepRemData)
    principalDf = pd.DataFrame(data=principalComponent, columns=['PC1', 'PC2', 'PC3'])
    ax = sns.scatterplot(x='PC1', y='PC2', data=principalDf, alpha=0.1, hue=spectraData2['file_isADPR'])
    plt = ax.get_figure()
    sns.despine()
    plt.savefig('ADPRibose/PCA_Y_no60.png')
    plt.clf()
    

    if printMetrics == True:
        print("Y no 60:" + str(
            metrics.silhouette_score(principalDf[['PC1', 'PC2']], spectraData2['file_isADPR'], metric='euclidean')))

    """
    # diag masses
    figure(figsize=(10, 10))
    diagCols = [col for col in spectraData2.columns if 'ox' in col]  # + ['Spectrum', 'Mass Shift']
    diagData = spectraData2[diagCols]
    diagData = diagData - diagData.mean()
    pca = PCA(n_components=len(diagCols))
    principalComponent = pca.fit_transform(diagData)
    principalDf = pd.DataFrame(data=principalComponent,
                               columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])
    ax = sns.scatterplot(x='PC1', y='PC2', data=principalDf, alpha=0.1, hue=spectraData2['file_isADPR'], legend=False,
                         rasterized=True)
    plt = ax.get_figure()
    sns.despine()
    plt.savefig('ADPRibose/PCA_diag.png')
    plt.savefig('ADPRibose/PCA_diag.pdf')
    plt.clf()

    if printMetrics == True:
        print("Diag: " + str(
            (metrics.silhouette_score(principalDf[['PC1', 'PC2']], spectraData2['file_isADPR'], metric='euclidean'))))

    # combined
    pepRemCols = [col for col in spectraData2.columns if 'Y' in col or 'ox' in col]  # + ['Spectrum', 'Mass Shift']
    pepRemData = spectraData2[pepRemCols]
    pepRemData = pepRemData - pepRemData.mean()
    pca = PCA(n_components=len(pepRemCols))
    principalComponent = pca.fit_transform(pepRemData)
    principalDf = pd.DataFrame(data=principalComponent,
                               columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])
    ax = sns.scatterplot(x='PC1', y='PC2', data=principalDf, alpha=0.1, hue=spectraData2['file_isADPR'])
    plt = ax.get_figure()
    sns.despine()
    plt.savefig('ADPRibose/PCA_combined.png')
    plt.savefig('ADPRibose/PCA_combined.pdf')
    plt.clf()

    """
    Writes PCA plots without unmodified spectra
    """
    colors = ['#E62125', '#F7941D']
    sns.set_palette(sns.color_palette(colors))

    # spectraData2 = spectraData.loc[(spectraData['isADPR'] == True) & (spectraData['file'] == True)]
    spectraData2 = spectraData[spectraData['isADPR'] == True]
    spectraData2.reset_index(drop=True, inplace=True)

    # Specific diagnostic ions
    pepRemCols = [col for col in spectraData2.columns if 'ox_5' in col]  # + ['Spectrum', 'Mass Shift']
    pepRemData = spectraData2[pepRemCols]
    pca = PCA(n_components=len(pepRemCols))
    principalComponent = pca.fit_transform(pepRemData)
    principalDf = pd.DataFrame(data=principalComponent, columns=['PC1', 'PC2'])
    ax = sns.scatterplot(x='PC1', y='PC2', data=principalDf, alpha=0.1, hue=spectraData2['file_isADPR'])
    plt = ax.get_figure()
    sns.despine()
    plt.savefig('ADPRibose/PCA_diag_specific_nounmod.png')
    plt.savefig('ADPRibose/PCA_diag_specific_nounmod.pdf')
    plt.clf()

    # Y masses
    pepRemCols = [col for col in spectraData2.columns if 'Y' in col]  # + ['Spectrum', 'Mass Shift']
    pepRemData = spectraData2[pepRemCols]
    pca = PCA(n_components=len(pepRemCols))
    principalComponent = pca.fit_transform(pepRemData)
    principalDf = pd.DataFrame(data=principalComponent, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    ax = sns.scatterplot(x='PC1', y='PC2', data=principalDf, alpha=0.1, hue=spectraData2['file_isADPR'])
    plt = ax.get_figure()
    sns.despine()
    plt.savefig('ADPRibose/PCA_Y_nounmod.png')
    plt.savefig('ADPRibose/PCA_Y_nounmod.pdf')
    plt.clf()

    # diag masses
    diagCols = [col for col in spectraData2.columns if 'ox' in col]  # + ['Spectrum', 'Mass Shift']
    diagData = spectraData2[diagCols]
    diagData = diagData - diagData.mean()
    pca = PCA(n_components=len(diagCols))
    principalComponent = pca.fit_transform(diagData)
    principalDf = pd.DataFrame(data=principalComponent, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
    ax = sns.scatterplot(x='PC1', y='PC2', data=principalDf, alpha=0.1, hue=spectraData2['file_isADPR'])
    plt = ax.get_figure()
    sns.despine()
    plt.savefig('ADPRibose/PCA_diag_nounmod.png')
    plt.savefig('ADPRibose/PCA_diag_nounmod.pdf')
    plt.clf()

    # combined
    pepRemCols = [col for col in spectraData2.columns if 'Y' in col or 'ox' in col]  # + ['Spectrum', 'Mass Shift']
    pepRemData = spectraData2[pepRemCols]
    pepRemData = pepRemData - pepRemData.mean()
    pca = PCA(n_components=len(pepRemCols))
    principalComponent = pca.fit_transform(pepRemData)
    principalDf = pd.DataFrame(data=principalComponent,
                               columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])
    ax = sns.scatterplot(x='PC1', y='PC2', data=principalDf, alpha=0.1, hue=spectraData2['file_isADPR'])
    plt = ax.get_figure()
    sns.despine()
    plt.savefig('ADPRibose/PCA_combined_nounmod.png')
    plt.savefig('ADPRibose/PCA_combined_nounmod.pdf')
    plt.clf()


def makeADPRLocalizationHistos():
    path = "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\newTools\\Mouse"
    allFiles = glob.glob(path + "\\*.rawglyco")

    # df = pd.concat((pd.read_csv(f, sep='\t') for f in allFiles))
    df = pd.concat((pd.read_csv(f, sep='\t') for f in allFiles))
    print(len(df))
    df.reset_index(drop=True, inplace=True)

    df['bin'] = df.apply(multiADPRBinMarker, axis=1)
    df = df[df['bin'] != 'None']

    """
    Calculate pseudo FDR, F1, and localizations for +80
    """
    print('calculating fdr for +541')
    """
    tempDeltaScoresArray = df[['Spectrum', 'deltascore_541.0611', 'localization_541.0611', 'bin']]
    tempDeltaScoresArray = tempDeltaScoresArray[tempDeltaScoresArray['bin'].isin(['ADPR', 'Unmod'])]

    fdrArray = pd.concat([tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'Unmod'].sample(
        min(tempDeltaScoresArray['bin'].value_counts())),
        tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'ADPR']])
    fdrArray.sort_values('deltascore_541.0611', inplace=True, ascending=False)
    fdrThresh = calcFdr('ADPR', 'Unmod', 'deltascore_541.0611', fdrArray)

    tempDeltaScoresArray = tempDeltaScoresArray[tempDeltaScoresArray['deltascore_541.0611'] > fdrThresh]
    tempDeltaScoresArray = tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'ADPR']
    tempDeltaScoresArray['isAboveThresh'] = tempDeltaScoresArray['deltascore_541.0611'] > fdrThresh
    tempDeltaScoresArray['locRes'] = tempDeltaScoresArray.apply(multiLocMarker541, axis=1)

    # calculate f1 stats
    tempF1ScoresArray1 = tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'ADPR']
    tempF1ScoresArray1.reset_index(drop=True, inplace=True)
    calcF1Stat(tempF1ScoresArray1, ['R'])
    # sns.lineplot(data=tempF1ScoresArray1, x='deltascore_79.9663', y='f1')
    # plt.savefig('LUAD_ProteomeToolsPTMs/phospho_f1.png')
    # plt.clf()

    # calculate loc-based FDR
    tempLocFdrArray1 = tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'ADPR']
    locFdrThresh = calcFdrFromLocs(['R'], 'deltascore_541.0611', tempLocFdrArray1)
    tempLocFdrArray1 = tempLocFdrArray1[tempLocFdrArray1['deltascore_541.0611'] >= locFdrThresh]

    # plot barplot of localized residues
    tempDeltaScoresArray = tempDeltaScoresArray[tempDeltaScoresArray['isAboveThresh'] == True]
    # sns.countplot(x='locRes', data=tempDeltaScoresArray[tempDeltaScoresArray['locRes'].isin(['S', 'T', 'Y', 'non-STY'])],
    #              orient='v', color='purple')
    # plt.savefig('LUAD_ProteomeToolsPTMs/phospho_loc_bar.png')
    # plt.clf()
    """
    """
    Make histograms for +541
    """
    print("plotting +541da histo")
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_ylim(0.90, 1)
    ax1.set_xlim(0, 30)
    # ax2.axvline(3, color='black', linestyle='dashed')
    ax1.tick_params(axis='x', which='both', bottom=False)
    sns.histplot(ax=ax1, x='deltascore_541.0611', data=df, binwidth=0.5, hue='bin', palette=['red', 'blue'],
                 edgecolor='white',
                 stat='proportion', common_norm=False)
    ax2.set_ylim(0, 0.10)
    ax2.set_xlim(0, 30)
    sns.histplot(ax=ax2, x='deltascore_541.0611', data=df, binwidth=0.5, hue='bin', palette=['red', 'blue'],
                 edgecolor='white', stat='proportion', common_norm=False)
    sns.despine(ax=ax1, bottom=True)
    sns.despine(ax=ax2)
    fig.text(0.04, 0.5, 'frequency', va='center', rotation='vertical')
    plt.savefig('ADPR_newTools/ADPR_deltascore_histo.pdf')
    plt.clf()

    """
    Calculate pseudo FDR, F1, and localizations for -42
    """
    """
    print('calculating fdr for -42')
    tempDeltaScoresArray = df[['Spectrum', 'deltascore_-42.0218', 'localization_-42.0218', 'bin']]
    tempDeltaScoresArray = tempDeltaScoresArray[tempDeltaScoresArray['bin'].isin(['ADPR', 'Unmod'])]

    fdrArray = pd.concat([tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'Unmod'].sample(
        min(tempDeltaScoresArray['bin'].value_counts())),
        tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'ADPR']])
    fdrArray.sort_values('deltascore_-42.0218', inplace=True, ascending=False)

    fdrThresh = calcFdr('ADPR', 'Unmod', 'deltascore_-42.0218', fdrArray)
    tempDeltaScoresArray['isAboveThresh'] = tempDeltaScoresArray['deltascore_-42.0218'] > fdrThresh
    tempDeltaScoresArray['locRes'] = tempDeltaScoresArray.apply(multiLocMarker42, axis=1)

    # save temp array for F1 stat
    tempF1ScoresArray2 = tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'ADPR']
    tempF1ScoresArray2.reset_index(drop=True, inplace=True)
    calcF1Stat(tempF1ScoresArray2, ['R'])
    # sns.lineplot(data=tempF1ScoresArray2, x='deltascore_-18.0106', y='f1')
    # plt.savefig('LUAD_ProteomeToolsPTMs/minus18_f1.png')
    # plt.clf()

    # calculate loc-based FDR
    tempLocFdrArray1 = tempDeltaScoresArray[tempDeltaScoresArray['bin'] == 'ADPR']
    locFdrThresh = calcFdrFromLocs(['R'], 'deltascore_-42.0218', tempLocFdrArray1)

    # plot piechart and barplot of localized residues
    tempDeltaScoresArray = tempDeltaScoresArray[tempDeltaScoresArray['isAboveThresh'] == True]
    # sns.countplot(x='locRes', data=tempDeltaScoresArray[tempDeltaScoresArray['locRes'].isin(['S', 'T', 'Y', 'non-STY'])],
    #              orient='v', color='purple')
    # plt.savefig('LUAD_ProteomeToolsPTMs/minus18_loc_bar.png')
    # plt.clf()
    """
    """
    Make histograms for -42
    """
    print("plotting -42da histo")
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_ylim(0.90, 1)
    ax1.set_xlim(0, 30)
    # ax2.axvline(9, color='black', linestyle='dashed')
    ax1.tick_params(axis='x', which='both', bottom=False)
    sns.histplot(ax=ax1, x='deltascore_-42.0218', data=df, binwidth=0.5, hue='bin', palette=['red', 'blue'],
                 edgecolor='white',
                 stat='proportion', common_norm=False)
    ax2.set_ylim(0, 0.10)
    ax2.set_xlim(0, 30)
    sns.histplot(ax=ax2, x='deltascore_-42.0218', data=df, binwidth=0.5, hue='bin', palette=['red', 'blue'],
                 edgecolor='white',
                 stat='proportion', common_norm=False)
    sns.despine(ax=ax1, bottom=True)
    sns.despine(ax=ax2)
    fig.text(0.04, 0.5, 'frequency', va='center', rotation='vertical')
    plt.savefig('ADPR_newTools/ADPR_loss_deltascore_histo.pdf')
    plt.clf()


def makeNewAdprBoxplots():
    diagIonData = pd.read_excel(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\final_data\\SummaryTableDiagnosticAndPeptide_final.xlsx",
        engine='openpyxl')

    matplotlib.rc('xtick', labelsize=6)
    matplotlib.rc('ytick', labelsize=6)

    diagIonData['position'] = diagIonData['position'].astype('int')
    diagIonData['round_ion_mass'] = diagIonData['round_ion_mass'].astype('str')
    diagRows = diagIonData[diagIonData['ion_type'] == 'diagnostic']
    vals = diagRows['position'].value_counts().keys().tolist()
    counts = diagRows['position'].value_counts().tolist()
    diag_value_dict = dict(zip(vals, counts))

    colors = ['#F7941D', '#E62125', '#7DC242', '#72CDDF', '#2D2D85', '#E60F8D']
    colors.reverse()

    fig = plt.figure(figsize=(11.8*cm, 2.5*cm))

    axes = []
    cind = 0
    for k, v in diag_value_dict.items():
        ax = plt.subplot2grid((1, 10), (0, cind), colspan=v)
        axes.append(ax)
        cind += v

    for i, (axi, k) in enumerate(zip(axes, diag_value_dict.keys())):
        subDf = diagRows[diagRows['position'] == k]
        print(subDf)
        plot_order = subDf.sort_values(by='round_ion_mass', ascending=False)
        plot_indx = plot_order.index
        plot_order = plot_order['round_ion_mass']
        # axj = axi.twinx()
        # sns.lineplot(ax=axj, x=subDf['round_ion_mass'][plot_indx], y=subDf['mod_intensity'][plot_indx], linewidth=2)
        # sns.scatterplot(ax=axj, x=subDf['round_ion_mass'][plot_indx], y=subDf['mod_intensity'][plot_indx], size=5, legend=False)
        sns.boxplot(ax=axi, data=subDf, x='round_ion_mass', y='unmod_percent', linewidth=5,
                    medianprops={'color': 'gray'}, order=plot_order)
        # sns.boxplot(ax=axi, data=subDf, x='round_ion_mass', y='mod_intensity', linewidth=7,
        #            medianprops={'color': 'black'}, order=plot_order)
        sns.barplot(ax=axi, data=subDf, x='round_ion_mass', y='mod_percent', linewidth=0, color=colors[k - 1],
                    order=plot_order)
        sns.despine(top=True, right=True, bottom=True)
        axi.set(xlabel=None)
        # axj.set(xlabel=None)
        # axi.set(ylabel="pct spectra with ion")
        axi.set(ylabel="pct spectra with ion")
        axi.set_ylim(0, 100)
        # axi.set_xticklabels([])
        # axi.set_ylim(0,100)
        # axj.set_ylim(0,100)
        if i > 0:
            axi.set(ylabel=None)
            sns.despine(left=True)
            axi.set(ylabel=None)
            axi.set_yticks([])
            # axi.set(yticklabels=[])
            # axi.tick_params(left=False)


    plt.tight_layout(pad=0)
    plt.savefig('ADPR_newTools/final_ADPR/ion_barchart.pdf')
    plt.clf()

    diagIonData['position'] = diagIonData['position'].astype('int')
    pepRows = diagIonData[diagIonData['ion_type'] == 'peptide']
    vals = pepRows['position'].value_counts().keys().tolist()
    counts = pepRows['position'].value_counts().tolist()
    pep_value_dict = dict(zip(vals, counts))

    colors = ['#F7941D', '#E62125', '#7DC242', '#72CDDF', '#2D2D85', '#E60F8D']
    colors.reverse()

    fig = plt.figure(figsize=(11.8*cm, 1.75*cm))

    axes = []
    cind = 0
    for k, v in sorted(pep_value_dict.items(), reverse=True):
        ax = plt.subplot2grid((1, 10), (0, cind), colspan=v)
        axes.append(ax)
        cind += v

    print(pep_value_dict)
    for i, (axi, k) in enumerate(zip(axes, sorted(pep_value_dict.keys(), reverse=True))):
        print(subDf)
        subDf = pepRows[pepRows['position'] == k]
        plot_order = subDf.sort_values(by='round_ion_mass', ascending=False)
        plot_order = plot_order['round_ion_mass']
        # sns.boxplot(ax=axi, data=subDf, x='round_ion_mass', y='mod_intensity', linewidth=4, medianprops={'color':'red'})
        sns.boxplot(ax=axi, data=subDf, x='round_ion_mass', y='unmod_percent', linewidth=5,
                    medianprops={'color': 'gray'}, order=plot_order)
        sns.barplot(ax=axi, data=subDf, x='round_ion_mass', y='mod_percent', color=colors[k - 1], order=plot_order)
        sns.despine(top=True, right=True, bottom=True)
        axi.set(xlabel=None)
        axi.set(ylabel="pct spectra with ion")
        # axi.set_xticklabels([])
        axi.set_ylim(0, 70)
        if i > 0:
            axi.set(ylabel=None)
            sns.despine(left=True)
            axi.set(ylabel=None)
            axi.set_yticks([])
            # axi.set(yticklabels=[])
            # axi.tick_params(left=False)
        else:
            axi.set_yticks([0, 20, 40, 60])

    plt.tight_layout(pad=0)
    plt.savefig('ADPR_newTools/final_ADPR/pep_barchart.pdf')


def makeAdprBoxplots():
    spectraData = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\Combined\\combined_rawglyco.tsv",
        sep='\t')
    spectraData['file'] = spectraData['Spectrum'].str.contains('Liver')
    spectraData['isADPR'] = spectraData['Mass Shift'] > 50
    spectraData['file_isADPR'] = spectraData.apply(multiCategoryMarker, axis=1)

    # spectraData2 = spectraData.loc[(spectraData['isADPR'] == True) & (spectraData['file'] == False)]
    # spectraData2 = spectraData[spectraData['isADPR'] == True]

    # Specific diagnostic ions
    pepRemCols = [col for col in spectraData.columns if 'ox' in col]

    keepCols = [col for col in spectraData.columns if 'ox' in col] + ['isADPR']

    spectraData2 = spectraData[keepCols]
    spectraData2 = pd.melt(spectraData2, id_vars='isADPR', value_vars=pepRemCols)

    print(sorted(pepRemCols, reverse=True))

    order = ['ox_582.4014_intensity', 'ox_524.0578_intensity', 'ox_428.0366_intensity', 'ox_348.0702_intensity',
             'ox_250.0932_intensity', 'ox_136.0616_intensity']
    labels = [col.split('_')[1] for col in order]

    flierprops = dict(marker='o', markersize=1, alpha=0.01)

    fig = sns.boxplot(data=spectraData2, x='variable', y='value', hue='isADPR',
                      order=order, hue_order=[False, True],
                      fliersize=0)

    fig.set_xticklabels(labels, rotation=45, ha="right")

    # combined
    sns.despine()
    plt.tight_layout()
    plt.savefig('ADPRibose/boxplot2.pdf')
    plt.clf()


def makeADPRPurity():
    path = "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\newTools\\Hela"
    psms = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\newTools\\Hela\\psm.tsv_normalized",
        sep="\t")
    ions = pd.read_csv(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\newTools\\Hela\\dataset01.rawglyco",
        sep="\t")

    print(len(psms))
    df = psms.merge(ions, on='Spectrum')
    print(len(psms))

    cols = [col for col in df.columns if "ox_" in col]
    cols = ["ox_524.0578_intensity", "ox_428.0366_intensity", "ox_348.0702_intensity", "ox_250.0932_intensity"]

    print(df.columns)
    print(cols)

    df['bin'] = df.apply(multiADPRBinMarker, axis=1)
    df = df[df['bin'] == 'Unmod']
    # for row in df.iterrows():
    #    print(row)
    print(len(df))

    scatterData = pd.melt(df, id_vars=['Spectrum', 'Purity'], value_vars=cols)

    print(scatterData.columns)
    print(len(scatterData))
    print(scatterData['variable'], scatterData['value'])

    sns.scatterplot(data=scatterData, x='Purity', y='value', hue='variable', alpha=0.25)

    plt.savefig('ADPR_newTools/ADPR_HeLa_purity_scatter_Unmod.pdf')
    plt.clf()


def multiCategoryMarker(df):
    if df['file'] == False and df['isADPR'] == False:
        return "Unmod"
    if df['file'] == False and df['isADPR'] == True:
        return "Ser_Mod"
    if df['file'] == True and df['isADPR'] == False:
        return "Unmod"
    else:
        return "Arg_Mod"


def multiLocMarker80(df):
    locCol = 'localization_79.9663'
    if type(df[locCol]) == float:
        return 'None'
    if len(df[locCol]) <= 3:  # <=3 chars means localized to a single residue
        if 'S' in df[locCol]:
            return 'S'
        if 'T' in df[locCol]:
            return 'T'
        if 'Y' in df[locCol]:
            return 'Y'
        else:
            return 'non-STY'
    else:
        return 'None'


def multiLocMarker541(df):
    locCol = 'localization_541.0611'
    if type(df[locCol]) == float:
        return 'None'
    if len(df[locCol]) <= 3:  # <=3 chars means localized to a single residue
        if 'R' in df[locCol]:
            return 'R'
        else:
            return 'non-R'
    else:
        return 'None'


def multiLocMarker18(df):
    locCol = 'localization_-18.0106'
    if type(df[locCol]) == float:
        return 'None'
    if len(df[locCol]) <= 3:  # <=3 chars means localized to a single residue
        if 'S' in df[locCol]:
            return 'S'
        if 'T' in df[locCol]:
            return 'T'
        else:
            return 'non-ST'
    else:
        return 'None'


def multiLocMarker42(df):
    locCol = 'localization_-42.0218'
    if type(df[locCol]) == float:
        return 'None'
    if len(df[locCol]) <= 3:  # <=3 chars means localized to a single residue
        if 'R' in df[locCol]:
            return 'R'
        else:
            return 'non-R'
    else:
        return 'None'


def multiSignifMarker(df):
    # x = df['p_value'] / 400.0
    # y = df['auc']
    # z = df['avg_int'] / 100.0

    # dist = np.log((1 - x)*(1 - x) + (1 - y)*(1 - y) + (1 - z)*(1 - z))
    # return dist

    if df['p_value'] < 0.05 and df['auc'] > 0.55 and df['avg_int_diff'] > 5.0:
        return '#17517e'
    elif df['p_value'] < 0.05 and df['auc'] < 0.45 and df['avg_int_diff'] < -5.0:
        return '#bc3754'
    # elif df['p_value'] < 0.05:
    #    return '#33638DFF'
    else:
        return '#000000'


def multiSignifMarkerCp(df):
    if df['log_p_value'] > 2 and df['log_fold_change'] > 1.6:
        return '#17517e'
    # elif df['p_value'] < 0.05 and df['auc'] < 0.45 and df['avg_int_diff'] < -5.0:
    #    return '#bc3754'
    # elif df['p_value'] < 0.05:
    #    return '#33638DFF'
    else:
        return '#000000'


def multiSignifMarkerAlpha(df):
    # x = df['p_value'] / 400.0
    # y = df['auc']
    # z = df['avg_int'] / 100.0

    # dist = np.log((1 - x)*(1 - x) + (1 - y)*(1 - y) + (1 - z)*(1 - z))
    # return dist

    if df['p_value'] < 0.05 and df['auc'] > 0.55 and df['avg_int_diff'] > 5.0:
        return 0.5
    elif df['p_value'] < 0.05 and df['auc'] < 0.45 and df['avg_int_diff'] < -5.0:
        return 0.5
    # elif df['p_value'] < 0.05:
    #    return '#33638DFF'
    else:
        return 0.1

    # if df['p_value'] < 0.05 and df['auc'] > 0.55 and df['avg_int'] > 5.0:
    #    return 1
    # elif df['p_value'] < 0.05 and df['auc'] > 0.55:
    #    return 0.5
    # elif df['p_value'] < 0.05:
    #    return 0.25
    # else:
    #    return 0.1


def multiSignifMarkerAlphaColor(df):
    # x = df['p_value'] / 400.0
    # y = df['auc']
    # z = df['avg_int'] / 100.0

    # dist = np.log((1 - x)*(1 - x) + (1 - y)*(1 - y) + (1 - z)*(1 - z))
    # return dist

    r, g, b = to_rgb(df['color'])
    color_alpha = (r, g, b, df['alpha'])
    return color_alpha


def multiFileMarker(df):
    if 'Unmod' in df['Spectrum']:
        return 'Unmod'
    else:
        fname = '_'.join(df['Spectrum'].split('_')[6:8])
        return fname


def isValidDMassMarker(df):
    masses = [0.0002, 0.9846, 100.0160, 114.0326, 114.0430, 14.0160, 15.9950, 226.0784, 27.9952,
              28.0316, 42.0108, 42.0474, 44.9850, 56.0266, 68.0262, 70.0420, 79.9666, 86.0020, 86.0366]
    for mass in masses:
        if abs(mass - df['Mass Shift']) < 0.01:
            return True
    return False


def multiBinMarker(df):
    if -0.01 <= df['Mass Shift'] <= 0.01:
        return 'Unmod'
    elif 79.96140 <= df['Mass Shift'] <= 79.97440:
        return 'Phospho'
    else:
        return 'None'


def multiBinMarkerHistogramBins(df):
    bins = [i / 100.0 for i in range(102)]
    for i, leftEdge in enumerate(bins[:-1]):
        if df['Purity'] >= leftEdge and df['Purity'] < bins[i + 1]:
            return i / 100.0


def multiADPRBinMarker(df):
    if -0.01 <= df['Mass Shift'] <= 0.01:
        return 'Unmod'
    elif 541.0 <= df['Mass Shift'] <= 541.2:
        return 'ADPR'
    else:
        return 'None'


def calcFdr(tLab, dLab, dScoreBin, df):
    t = 0
    d = 0
    lastQVal = 0
    lastDScoreBin = 0
    breakPoints = []
    for i, row in df.iterrows():
        if row['bin'] == tLab:
            t += 1
        elif row['bin'] == dLab:
            d += 1
        qVal = float(d + 1) / float(t)

        if lastQVal < 0.05 and qVal >= 0.05:
            breakPoints.append(lastDScoreBin)
        lastDScoreBin = row[dScoreBin]
        lastQVal = qVal

    print(breakPoints)
    return min(breakPoints)


def calcFdrFromLocs(tLabs, dScoreBin, df):
    qvals = np.zeros(len(df))
    t = 0
    d = 0
    lastQVal = 0
    lastDScoreBin = 0
    breakPoints = []
    cind = 0
    for i, row in df.iterrows():
        # print(t, d, row['locRes'], row[5], row[6], row[7], row[8])
        if row['locRes'] in tLabs:
            # print("t")
            t += 1
        elif row['locRes'] == 'None':
            pass
        else:
            # print("d")
            d += 1
        print(d, t)
        qVal = float(d + 1) / float(t)
        # print(qVal)
        qvals[cind] = qVal
        cind += 1

    return qvals


def calcFnrFromLocs(tLabs, dScoreBin, df):
    qvals = np.zeros(len(df))
    t = 0
    d = 0

    cind = 0
    for i, row in df.iterrows():
        # print(t, d, row['locRes'], row[5], row[6], row[7], row[8])
        if row['locRes'] in tLabs:
            # print("t")
            t += 1
        elif row['locRes'] == 'None':
            pass
        else:
            # print("d")
            d += 1
        qVal = float(d + 1) / float(t)
        # print(qVal)
        qvals[cind] = qVal
        cind += 1

    return qvals


def calcRocFromLocs(tLabs, dScoreBin, df):
    rocVals = np.zeros(shape=(2, len(df)))
    counts = df['locRes'].value_counts()
    if '-18' in dScoreBin:
        fps = counts['non-ST'] + counts['None']
        print(fps)
        tps = counts['S'] + counts['T']
        print(tps)
    elif '79' in dScoreBin:
        fps = counts['non-STY'] + counts['None']
        tps = counts['S'] + counts['T'] + counts['Y']
    t = 0
    d = 0
    cind = 0
    for i, row in df.iterrows():
        print(t, d, row['locRes'], row[5], row[6], row[7], row[8])
        if row['locRes'] in tLabs:
            t += 1
        elif row['locRes'] == 'None':
            d += 1
        else:
            d += 1
        tpr = t / tps
        fpr = d / fps
        print(tpr, fpr)
        rocVals[0, cind] = tpr
        rocVals[1, cind] = fpr
        cind += 1

    return rocVals


def calcF1Stat(df, allowedAas):
    f1arr = np.zeros(len(df))
    tp = 0
    fp = 0
    fn = 0
    cInd = 0
    for i, row in df.iterrows():
        if row['locRes'] in allowedAas:
            tp += 1
        elif row['locRes'] == 'None':
            fn += 1
        else:
            fp += 1
        f1 = (2 * tp) / (2 * tp + fp + fn)
        f1arr[cInd] = f1
        cInd += 1
    return f1arr


def makeBarChart():
    diagIonData = pd.read_excel(
        "C:\\Users\\danny\\Dropbox (University of Michigan)\\DiagMiningPaper\\Data\\ADPR\\SummaryTableForFigures.xlsx",
        engine='openpyxl')
    firstBarChart = diagIonData.loc[diagIonData['included'] == 1]
    firstBarChart = pd.melt(firstBarChart, id_vars=['position', 'source'], value_vars=['prop_mod', 'prop_unmod'])
    firstBarChart['source'] = firstBarChart[['source', 'variable']].astype(str).apply('_'.join, 1)
    print(firstBarChart.columns)
    print(firstBarChart)

    ax = sns.barplot(x="position", y="value", hue="source",
                     hue_order=['mouse_prop_mod', 'mouse_prop_unmod', 'HeLa_prop_mod', 'HeLa_prop_unmod'],
                     data=firstBarChart)
    plt = ax.get_figure()
    sns.despine()
    plt.savefig('ADPRibose/diagnosticIonsBarPlot.pdf')
    plt.clf()


def makeRNAXBarChart():
    ax = sns.barplot(x=['original', 'noLos', 'noLosY'], y=[6402, 6705, 6994], palette='dark')
    plt = ax.get_figure()
    sns.despine()
    plt.savefig('RNAXLink/psm_counts.pdf')
    plt.clf()


# def makeNormalDist():


if __name__ == '__main__':
    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    matplotlib.rcParams['pdf.fonttype'] = 42
    #makeGlycoFigures()
    #makeRNAXLinkFigures()
    #makeAdprFigures()

    print(np.__version__)
    print(sns.__version__)
    print(pd.__version__)
    print(sklearn.__version__)

    """unused"""
    # makePtptmsFigures()
    # makeCysProbeFigures()
    # makeLuadFigures()
    # makeCCRCCGlycoFigures()
    # makeMouseKidneyGlycoFigures()
    # makeOxMetFigures()

    # makeNormalDist()
