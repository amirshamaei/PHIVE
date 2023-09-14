#%% import modules
import csv
import json
import random
from datetime import date
from pathlib import Path
import nibabel as nib
import hlsvdpro
import pandas as pd
import scipy.io as sio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product, permutations
from sklearn.linear_model import LinearRegression
from dateutil.utils import today
from scipy.stats import pearsonr, stats
import numpy.fft as fft
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


# important note about sklearn linear reg
#
#%% parameters
exten_path = 'clean_code/vae_new/'
analyze_name = 'exp20'
exten_path = ''
json_file_path = exten_path + 'runs/exp20.json'
test_data_root = "exp3/big_gaba_size_128_sd_3"
with open(json_file_path, 'r') as j:
    content = json.loads(j.read())
runs = content['runs']
run = runs[0]

json_file_path = exten_path + 'runs/exp21.json'
test_data_root = "exp3/big_gaba_size_128_sd_3"
with open(json_file_path, 'r') as j:
    content = json.loads(j.read())
runs.append(content['runs'][0])


saving_dir = exten_path+'analyze/' + analyze_name + '/'
Path(saving_dir).mkdir(parents=True, exist_ok=True)
save = True
t_step = run['t_step']
trnfreq = run['trnfreq']
Crfr = trnfreq * (4.7 - 3.027)
lentgh = run['sigLen']
_error = ["Error("+ i + ")" for i in run['met_name']]
_pred = ["Predicted("+ i + ")" for i in run['met_name']]
_true = ["True("+ i + ")" for i in run['met_name']]
met_name = run['met_name']
cmap = 'Reds'
t = np.expand_dims(np.arange(lentgh) * t_step, 1)
selected_met = ["Cr", "GPC", "Glu", "mIns", "NAA", "NAAG", "PCho", "PCr", "Tau"]
models = ['DeepFit','DeepFit-supervised']
selected_method = ["FiTAID"] + models
sns.set_style("white")
sns.set_palette("muted")
fontsize = 16
#%% functions
def plotppm(sig, ppm1, ppm2, rev, linewidth=0.3, linestyle='-'):
    p1 = int(ppm2p(ppm1, len(sig)))
    p2 = int(ppm2p(ppm2, len(sig)))
    n = p2 - p1
    x = np.linspace(int(ppm1), int(ppm2), abs(n))
    sig = np.squeeze(sig)
    df = pd.DataFrame({'Real Signal (a.u.)': sig[p2:p1].real})
    df['Frequency(ppm)'] = np.flip(x)
    g = sns.lineplot(x='Frequency(ppm)', y='Real Signal (a.u.)', data=df, linewidth=linewidth, linestyle=linestyle)
    plt.tick_params(axis='both', labelsize=fontsize)
    if rev:
        plt.gca().invert_xaxis()
    return g
def watrem(data, dt, n):
    npts = len(data)
    dwell = dt/0.001
    nsv_sought = n
    result = hlsvdpro.hlsvd(data, nsv_sought, dwell)
    nsv_found, singvals, freq, damp, ampl, phas = result
    idx = np.where((result[2] < (0.001 * (Crfr + 50))) & (result[2] > (0.001 * (Crfr - 50))))
    result = (len(idx),result[1],result[2][idx],result[3][idx],result[4][idx],result[5][idx])
    fid = hlsvdpro.create_hlsvd_fids(result, npts, dwell, sum_results=True, convert=False)
    return fid,result
def cal_snr_lw(signal):
    av_f = fft.fftshift(fft.fft((signal)))
    plotppm(av_f, 0, 5, False)
    lsr, rslt = watrem(signal, t_step, 8)
    lsr = fft.fftshift(fft.fft(((lsr))))
    plotppm(lsr, 0, 5, True)
    noise = np.std(signal.real[:-128])
    snr = rslt[4]/noise
    # plt.title('Linewidth: '+ str(-1000/(np.pi*res.x[2])) + "Hz" + "SNR: " + str(snr))
    plt.title('Linewidth: ' + str(-1 * 1000 / (np.pi * rslt[3])) + "Hz" + "SNR: " + str(snr))
    return snr,-1 * 1000 / (np.pi * rslt[3])


def cal_snr(data, endpoints=128, offset=0):
    return np.abs(data[0, :]) / np.std(data.real[-(offset + endpoints):-(offset + 1), :], axis=0)


def cal_snrf(data_f, endpoints=128, offset=0):
    return np.max(np.abs(data_f), 0) / (np.std(data_f.real[offset:endpoints + offset, :], axis=0))


def abs_err(a, b):
    return np.abs(a - b)

def err(a, b):
    return (a - b)

def savefig(path=str(date.today()),tight=False):
    plt.tick_params(axis='both', labelsize=fontsize)
    if tight:
        plt.tight_layout()
    if save:
        plt.savefig(saving_dir+path + ".svg", format="svg")
        plt.savefig(saving_dir+path + " .png", format="png", dpi=800)
    # plt.show()


def ppm2p(r, len):
    r = 4.7 - r
    return int(((trnfreq * r) / (1 / (t_step * len))) + len / 2)

def calib_plot(ampl_t,y_out, yerr=None,cmap=None,ident_lin=True):
    if cmap==None :
        ax = plt.scatter(x=ampl_t, y=y_out)
    else:
        ax = plt.scatter(x=ampl_t, y=y_out, c=yerr, cmap='Spectral')
        plt.set_cmap(cmap)
        cb = plt.colorbar()
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=fontsize)
    if ident_lin == True:
        plot_iden(ax)
    sns.despine()
    ax.axes.set_xlabel("True")
    ax.axes.set_ylabel('Predicted')
    plt.tick_params(axis='both', labelsize=fontsize)
    return ax
def plot_iden(ax):
    ax.axes.yaxis.set_ticks_position('left')
    ax.axes.xaxis.set_ticks_position('bottom')
    x0, x1 = ax.axes.get_xlim()
    y0, y1 = ax.axes.get_ylim()
    lims = [min(x0, x1), min(y0, y1)]
    ax.axes.axline((lims[0], lims[1]), slope=1, ls="--", zorder=0, color='silver')

def sym_mean_absolute_percentage_error(y_true,y_pred):
    nomir = np.abs(y_pred - y_true)
    demor = ((y_true) + (y_pred))
    output_errors = np.average(nomir, axis=0) / np.average(demor, axis=0)
    return np.average(output_errors)


def bland_altman_plot(data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')

def modified_bland_altman_plot(data1, data2, gt=None,c_map=None, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference
    if gt is not None:
        ax = plt.scatter(gt, diff,cmap='Spectral', *args, **kwargs)
    else:
        ax = plt.scatter(range(0, data1.shape[0]), diff ,cmap='Spectral', *args, **kwargs)
    plt.axhline(md, color='gray', linestyle='-',linewidth=3)
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--',linewidth=2)
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--',linewidth=2)
    sns.despine()
    if c_map != None:
        plt.set_cmap(c_map)
        cb = plt.colorbar()
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)

    return ax
#%% load data
# QST_MM = pd.read_csv(exten_path+'analyze/rslt_wiithout_ph_1/QST_MM.csv')
# QY_MM = pd.read_csv(exten_path+'analyze/rslt_wiithout_ph_1/QY_MM.csv')
# QY_Sub = pd.read_csv(exten_path+'analyze/rslt_wiithout_ph_1/QY_Sub.csv')

FA_T = pd.read_csv(exten_path+'analyze/rslt_wiithout_ph_1/FA_time_6.csv')
# FA_F = pd.read_csv(exten_path+'analyze/rslt_wiithout_ph_1/FA_freq.csv')

#%%
# id = "test_" + str(run['test_params'][6]) + "_" + str(run['test_params'][5]) + "_"
id = "test/" + test_data_root + "/"
data = np.load(exten_path+run['sim_order'][1]+test_data_root+'.npz')
y_test, mm_signals, ampl_t, shift_t, alpha_t, ph_t = [data[x] for x in data]
snr_t = cal_snrf(fft.fftshift(fft.fft(y_test, axis=0), axes=0))
dir_list = []
for run in runs:
    dir_list.append(exten_path+run['parent_root'] + run['child_root'] + run['version']+id)
encs_list = []
rslt_list = []
for dir in dir_list:
    try:
        encs_list.append(np.load(dir+'rslt_wiithout_ph_1.npz',allow_pickle=True))
        rslt_list.append(pd.read_csv(dir+'rslt_wiithout_ph_1.csv',index_col=[0]))
    except:
        print('cannot reach to the result at ' + dir )

#%%
met = ['NAA' , 'Cr']
df_uncer = pd.DataFrame()
errors_averaged_all = pd.DataFrame()
errors_corr_dl = pd.DataFrame(columns=met_name, index=["damping", 'frequency', 'Phase', 'SNR'])
for idy,encs in enumerate(encs_list):
    errors_averaged = pd.DataFrame(columns=['$R_2$', 'MAE', 'MSE', 'MAPE', 'r2', 'intercept', 'coef'], index=met_name)
    y_out, y_out_var, fr, damp, ph, decs, encs, epistemic_unc, aleatoric_unc, decoded_net, mm_= [encs[x] for x in encs]
    for idx, name in enumerate(met_name):
        model = LinearRegression(fit_intercept=True).fit(ampl_t[:, idx].reshape((-1, 1)), y_out[:, idx].reshape((-1, 1)))
        errors_averaged.iloc[idx] = [r2_score(ampl_t[:, idx], y_out[:, idx]),
                                     mean_absolute_error(ampl_t[:, idx], y_out[:, idx]),
                                     mean_squared_error(ampl_t[:, idx], y_out[:, idx]),
                                     sym_mean_absolute_percentage_error(ampl_t[:, idx], y_out[:, idx]) * 100,
                                     model.score(ampl_t[:, idx].reshape((-1, 1)), y_out[:, idx].reshape((-1, 1))),
                                     model.intercept_[0],
                                     model.coef_[0][0]
                                     ]
        if name in met_name:
            df_uncer['Epistemic Uncertainty ' + name + " " + models[idy]] = epistemic_unc[:, idx]
            df_uncer['Aleatoric Uncertainty ' + name + " " + models[idy]] = aleatoric_unc[:, idx]
            df_uncer['Total Uncertainty ' + name + " " + models[idy]] = aleatoric_unc[:, idx]+epistemic_unc[:, idx]
            df_uncer['Epistemic Uncertainty per' + name + " " + models[idy]] = 100*epistemic_unc[:, idx]/(y_out[:,idx])
            df_uncer['Aleatoric Uncertainty per' + name + " " + models[idy]] = 100*aleatoric_unc[:, idx]/(y_out[:,idx])

    # df_uncer_t['Method'] = models[idy]
    # df_uncer=df_uncer.append(df_uncer_t)
    errors_averaged['Method'] = models[idy]
    errors_averaged_all=errors_averaged_all.append(errors_averaged)
df_ampl = pd.DataFrame(ampl_t, columns=met_name)
for label, method in zip(["FiTAID"],[FA_T]):
    errors_averaged = pd.DataFrame(columns=['$R_2$', 'MAE', 'MSE', 'MAPE', 'r2', 'intercept', 'coef'], index=met_name)
    for idx, name in enumerate(met_name):
        model = LinearRegression(fit_intercept=True).fit(df_ampl[name].values.reshape((-1, 1)), method[name].values.reshape((-1, 1)))
        errors_averaged.iloc[idx] = [r2_score(df_ampl[name].values, method[name].values),
                                     mean_absolute_error(df_ampl[name].values, method[name].values),
                                     mean_squared_error(df_ampl[name].values, method[name].values),
                                     sym_mean_absolute_percentage_error(df_ampl[name].values, method[name].values) * 100,
                                     model.score(df_ampl[name].values.reshape((-1, 1)), method[name].values.reshape((-1, 1))),
                                     model.intercept_[0],
                                     model.coef_[0][0]
                                     ]
        if name in met_name:
            # df_uncer_t = pd.DataFrame()
            df_uncer['CRLB ' + name + " " + label] = method[name+ "_sd"]

    # df_uncer_t['Method'] = label
    # df_uncer=df_uncer.append(df_uncer_t)
    errors_averaged['Method'] = label
    errors_averaged_all=errors_averaged_all.append(errors_averaged)

errors_averaged_all.to_csv(saving_dir + "_errors_averaged_all.csv")
sns.barplot(errors_averaged_all,x= errors_averaged_all.index,y="r2",hue="Method")
savefig("bar_r2")
sns.barplot(errors_averaged_all,x= errors_averaged_all.index,y="MAPE",hue="Method")
savefig("bar_MAPE")
compar_mean=errors_averaged_all.groupby("Method").mean(0)
sns.scatterplot(x='$R_2$',y='MAPE',data=compar_mean,hue=compar_mean.index)
savefig("MAPEvsR2_mean")
compar_mean.to_csv(saving_dir + "MAPEvsR.csv")
dfm = errors_averaged_all.reset_index(level=0)
dfm["$1-R_2$"] = 1-dfm["$R_2$"]
# sns.scatterplot(x='$1-R_2$',y='MSE',data=dfm,hue='index',style='Method')
# sns.scatterplot(x='$R_2$',y='MSE',data=dfm.isin(["Single MM"]),hue='Method')
# sns.scatterplot(x='$R_2$',y='MSE',data=dfm[~dfm.isin(["QUEST 3"])],hue='Method')
ax = sns.relplot(x='$1-R_2$',y='MAPE',data=dfm[dfm["index"].isin(selected_met) &dfm["Method"].isin(selected_method)],style='index',hue='Method',alpha=0.9,s=100)
ax.set(xscale="log")
ax.set(yscale="log")
savefig("MAPEvsR2_met1")
ax = sns.relplot(x='$1-R_2$',y='MAPE',data=dfm[~dfm["index"].isin(selected_met) &dfm["Method"].isin(selected_method)],style='index',hue='Method',alpha=0.9,s=100)
ax.set(xscale="log")
ax.set(yscale="log")
savefig("MAPEvsR2_met2")
# %%
# sns.barplot(errors_averaged_all,x= )
# %%
tr = 128
df_met = pd.DataFrame()
for idx,rslt in enumerate(rslt_list):
    df_met_t = pd.DataFrame(columns=met_name)
    df_met_t[met_name] = rslt.loc[rslt['type'] == 'Predicted'][met_name].loc[0:tr]
    true_value = rslt.loc[rslt['type'] == 'True'][met_name].loc[0:tr]
    df_met_t[['True ' + i for i in met_name]] = true_value
    df_met_t['Model'] = models[idx]
    df_met_t[[x+"_error" for x in met_name]] = rslt.loc[rslt['type'] == 'Predicted'][met_name].loc[0:tr] -true_value
    df_met= df_met.append(df_met_t,ignore_index=True)

# true_value= rslt_list[0].loc[rslt_wiithout_ph_1['type'] == 'True'][met]
for label, rslt in zip(["FiTAID"],[FA_T]):
    df_met_t = pd.DataFrame(columns=met_name)
    df_met_t[met_name] = rslt[met_name].loc[0:tr]
    df_met_t[['True ' + i for i in met_name]] = true_value
    df_met_t['Model'] = label
    df_met_t[[x + "_error" for x in met_name]] = rslt[met_name].loc[0:tr] -true_value
    df_met= df_met.append(df_met_t,ignore_index=True)

#%%
# for met in met_name:
#     ax = sns.lmplot(x='True '+met, y=met, data=df_met, hue='Model', legend=True,scatter_kws={'alpha':0.6},line_kws={'lw': 1},ci=False)
#     max = df_met['True ' + met].max()
#     ax.axes[0,0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
#     savefig(met + "_compar")
def plot_altman(df_met,model,color):
    md = np.mean(df_met.loc[df_met['Model'] == model][met+"_error"])  # Mean of the difference
    sd = np.std(df_met.loc[df_met['Model'] == model][met+"_error"], axis=0)  # Standard deviation of the difference
    plt.axhline(md, color=color, linestyle='-',linewidth=3)
    plt.axhline(md + 1.96 * sd, color=color, linestyle='--',linewidth=2)
    plt.axhline(md - 1.96 * sd, color=color, linestyle='--',linewidth=2)

for met in met_name:
    ax = sns.relplot(x='True '+met,y=met+"_error", data=df_met, hue='Model', style='Model', legend=True,alpha=0.8, s=50)
    plot_altman(df_met,'DeepFit','darkslateblue')
    plot_altman(df_met, 'DeepFit-supervised', 'darkorange')
    plot_altman(df_met, 'FiTAID', 'darkgreen')
    savefig(met + "_compar")


#%%
tr = 128
df_fpds = pd.DataFrame()
df_mape = pd.DataFrame()
fpds = ['Damping', 'SNR']
for idx,rslt in enumerate(rslt_list):
    df_fpds_t = pd.DataFrame(columns=fpds)
    df_fpds_t[fpds] = rslt.loc[rslt['type'] == 'True'][fpds].loc[0:tr]
    df_fpds_t[met_name] = ((rslt.loc[rslt['type'] == 'Predicted'][met_name].loc[0:tr])-(rslt.loc[rslt['type'] == 'True'][met_name].loc[0:tr]))
    # /np.abs(rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'True'][met_name].loc[0:tr])
    df_fpds_t['Model'] = models[idx]
    df_fpds= df_fpds.append(df_fpds_t,ignore_index=True)

for label, rslt_q in zip(["FiTAID"],[FA_T]):
    df_fpds_t = pd.DataFrame(columns=fpds)
    df_fpds_t[fpds] = rslt.loc[rslt['type'] == 'True'][fpds].loc[0:tr]
    df_fpds_t[met_name] = ((rslt_q[met_name].loc[0:tr])-(rslt.loc[rslt['type'] == 'True'][met_name].loc[0:tr]))
    # /np.abs(rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'True'][met_name].loc[0:tr])
    df_fpds_t['Model'] = label
    df_fpds= df_fpds.append(df_fpds_t,ignore_index=True)

df_fpds_corr = df_fpds.groupby("Model").corr()
df_fpds_corr.to_csv(saving_dir+"damping_SNR_vs_error_corrolation")
#%%
met_ = ['NAA' , 'Cr']
for param in ['Damping', 'SNR']:
    for met in selected_met:
        ax = sns.lmplot(x=param, y=met, data=df_fpds, hue='Model', legend=True,ci=False,scatter_kws={'alpha':0.8})
        savefig(met + "_vs_" + param)
#%%




#%%
# for idx,rslt_wiithout_ph_1 in enumerate(rslt_list):
#     try:
#         dfm = pd.melt(rslt_wiithout_ph_1, id_vars=['type'])
#         sns.set_style('whitegrid')
#         sns.violinplot(x='variable', y='value', data=dfm[dfm['variable'].isin(selected_met)], hue='type',
#                        palette="Set3",
#                        linewidth=1,
#                        split=True,
#                        inner="quartile")
#         sns.despine()
#         savefig("violion_" + str(idx))
#     except:
#         print('cannot reach to the result')





#%%
df_uncer = pd.concat([df_fpds.loc[df_met['Model'] == 'DeepFit'].iloc[:,:-1].abs(),df_uncer],axis=1)
uncer_corr = df_uncer.corr()
# sns.heatmap(uncer_corr,cmap='PiYG')
# savefig( "heat_uncer")
uncer_corr.to_csv(saving_dir+"corr_uncer")
#%%
df_uncer = df_uncer.astype("float")
for metabolite in met_name:
    # x_ ='Epistemic Uncertainty '+metabolite+" DeepFit"
    # y_ = 'CRLB ' + metabolite + " " + "FiTAID"
    # ax = sns.lmplot(x=x_, y=y_, data=df_uncer, legend=True,scatter_kws={'alpha':0.6},line_kws={'lw': 1})
    # plt.title(uncer_corr.loc[x_,y_])
    # savefig( "uncer_epis" + metabolite)
    # x_ ='Total Uncertainty '+metabolite+" DeepFit"
    # y_ = 'CRLB ' + metabolite + " " + "FiTAID"
    # ax = sns.lmplot(x=x_, y=y_, data=df_uncer, legend=True,scatter_kws={'alpha':0.6},line_kws={'lw': 1})
    # plt.title(uncer_corr.loc[x_,y_])
    # savefig( "uncer_total" + metabolite)
    x_ ='Aleatoric Uncertainty '+metabolite+" DeepFit"
    y_ = 'CRLB ' + metabolite + " " + "FiTAID"
    ax = sns.lmplot(x=x_, y=y_, data=df_uncer, legend=True,scatter_kws={'alpha':0.6},line_kws={'lw': 1})
    plt.title(uncer_corr.loc[x_, y_])
    savefig("uncer_ale" + metabolite,tight=True)
    x_ ='Aleatoric Uncertainty per'+metabolite+" DeepFit"
    y_ = 'CRLB ' + metabolite + " " + "FiTAID"
    ax = sns.lmplot(x=x_, y=y_, data=df_uncer, legend=True,scatter_kws={'alpha':0.6},line_kws={'lw': 1})
    plt.title(uncer_corr.loc[x_, y_])
    savefig("per_uncer_ale" + metabolite,tight=True)


grp_df = df_fpds.groupby(['Model'])
grp_df.mean().to_csv(saving_dir+"grp_df_mean")
grp_df.std().to_csv(saving_dir+"grp_df_std")
# for metabolite in met_name:
#     x_ ='Epistemic Uncertainty '+metabolite+" DeepFit"
#     ax = sns.lmplot(x=x_, y=metabolite, data=df_uncer, legend=True,scatter_kws={'alpha':0.6},line_kws={'lw': 1})
#     plt.title(uncer_corr.loc[x_, metabolite])
#     savefig("uncer_epis_error" + metabolite,tight=True)
#%%
# df = pd.DataFrame()
# met_id_ens = 12
# for idx,encs in enumerate(encs_list):
#     df = pd.DataFrame()
#     y_out, y_out_var, fr, damp, ph, decs, encs, epistemic_unc, aleatoric_unc = [encs[x] for x in encs]
#     for i in range(0, run['ens']):
#         df[str(i)] = encs[i, :, met_id_ens]
#         # ax = calib_plot(ampl_t[:, met_id_ens], encs[i,:,met_id_ens],ident_lin=False)
#     df['True'] = ampl_t[:, met_id_ens]
#     dfm = df.melt(id_vars='True', var_name='Ensemble', value_name='Predicted')
#     ax = sns.lmplot(x='True', y='Predicted', data=dfm, hue='Ensemble', scatter_kws={'alpha': 0.6}, line_kws={'lw': 1})
#     ax.axes[0,0].axline((0.5, 0.5), slope=1, ls="--", zorder=0, color='silver')
#     savefig(saving_dir + "ens_" + run['met_name'][met_id_ens] + "_" + str(idx))
#
# #%%
# file = open(saving_dir+ '_rslt.csv', 'w')
# writer = csv.writer(file)
# df_err = pd.DataFrame()
#
# for idx,rslt_wiithout_ph_1 in enumerate(rslt_list):
#     mean_f = np.mean(abs_err(rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'True']['Frequency'], rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'Predicted']['Frequency']))
#     mean_alph = np.mean(abs_err(rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'True']['Damping'], rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'Predicted']['Damping']))
#     mean_ph = np.mean(abs_err(rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'True']['Phase'], rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'Predicted']['Phase']))
#     std_f = np.std(abs_err(rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'True']['Frequency'], rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'Predicted']['Frequency']))
#     std_alph = np.std(abs_err(rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'True']['Damping'], rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'Predicted']['Damping']))
#     std_ph = np.std(abs_err(rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'True']['Phase'], rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'Predicted']['Phase']))
#     writer.writerow(dir_list[idx])
#     writer.writerow(["freq", mean_f, std_f])
#     writer.writerow(["damp", mean_alph, std_alph])
#     writer.writerow(["ph", mean_ph, std_ph])
#     df_err_t = pd.DataFrame(columns=['Frequency', 'Damping', 'Phase', 'Method'])
#     df_err_t['Predicted Frequency'] = rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'Predicted']['Frequency']
#     df_err_t['Predicted Damping'] = rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'Predicted']['Damping']
#     df_err_t['Predicted Phase'] = rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'Predicted']['Phase']
#     df_err_t['True Frequency'] = rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'True']['Frequency']
#     df_err_t['True Damping'] = rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'True']['Damping']
#     df_err_t['True Phase'] = rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'True']['Phase']
#     df_err_t['Model'] = models[idx]
#     df_err= df_err.append(df_err_t,ignore_index=True)
#
# ax = sns.lmplot(x='True Frequency', y='Predicted Frequency', data=df_err, hue='Model', legend=True,scatter_kws={'alpha':0.6},line_kws={'lw': 1})
# ax.axes[0,0].axline((0, 0), slope=1, ls="--", zorder=0, color='silver')
# savefig(saving_dir + "Frequency" + "_dl" )
# ax = sns.lmplot(x='True Damping', y='Predicted Damping', data=df_err, hue='Model', legend=True,scatter_kws={'alpha':0.6},line_kws={'lw': 1})
# ax.axes[0,0].axline((0, 0), slope=1, ls="--", zorder=0, color='silver')
# savefig(saving_dir + "Damping" + "_dl" )
# ax = sns.lmplot(x='True Phase', y='Predicted Phase', data=df_err, hue='Model', legend=True,scatter_kws={'alpha':0.6},line_kws={'lw': 1})
# ax.axes[0,0].axline((0, 0), slope=1, ls="--", zorder=0, color='silver')
# savefig(saving_dir + "Phase" + "_dl" )
# #%%
# ids1 = [2, 12, 8, 14, 17, 9]
# ids2 = [15, 13, 7, 5, 6, 10]
# names = ["Cr+PCr", "NAA+NAAG", "Glu+Gln", "PCho+GPC", "Glc+Tau", "Ins+Gly"]
# errors_combined_all = pd.DataFrame()
# errors_combined = pd.DataFrame(columns=['$R_2$', 'MAE', 'MSE', 'MAPE', 'r2',  'coef'], index=names)
# idx = 0
# df = pd.DataFrame()
# for idx,encs in enumerate(encs_list):
#     y_out, y_out_var, fr, damp, ph, decs, encs, epistemic_unc, aleatoric_unc = [encs[x] for x in encs]
#     errors_combined = pd.DataFrame(columns=['$R_2$', 'MAE', 'MSE', 'MAPE', 'r2',  'coef'], index=names)
#     idy = 0
#     df_t = pd.DataFrame()
#     for id1, id2, name in zip(ids1, ids2, names):
#         # var = (y_out_var[:, id1]**2 + y_out_var[:, id2]**2) + (ampl_t[:, id1]**2 + ampl_t[:, id2]**2)
#         # corr, _ = pearsonr(ampl_t[:, id1], ampl_t[:, id2])
#         # warning! how we can calculate sd for two corrolated normal distribution!?
#
#         # sd = 100 * np.sqrt(y_out_var[:, id1] + y_out_var[:, id2]) / (y_out[:, id1] + y_out[:, id2])
#         # calib_plot(ampl_t[:, id1] + ampl_t[:, id2], (y_out[:, id1] + y_out[:, id2]), None, None)
#         # plt.title(name)
#         # savefig(saving_dir+ "combined_" + name)
#         # plt.show()
#         df_t['Predicted('+name+')'] = y_out[:, id1] + y_out[:, id2]
#         df_t['True('+name+')'] = ampl_t[:, id1] + ampl_t[:, id2]
#
#         model = LinearRegression(fit_intercept=False).fit((ampl_t[:, id1] + ampl_t[:, id2]).reshape((-1, 1)),
#                                        (y_out[:, id1] + y_out[:, id2]).reshape((-1, 1)))
#         errors_combined.iloc[idy] = [r2_score(ampl_t[:, id1] + ampl_t[:, id2], (y_out[:, id1] + y_out[:, id2])),
#                                      mean_absolute_error(ampl_t[:, id1] + ampl_t[:, id2],
#                                                          (y_out[:, id1] + y_out[:, id2])),
#                                      mean_squared_error(ampl_t[:, id1] + ampl_t[:, id2],
#                                                         (y_out[:, id1] + y_out[:, id2])),
#                                      mean_absolute_percentage_error(ampl_t[:, id1] + ampl_t[:, id2],
#                                                                     (y_out[:, id1] + y_out[:, id2])) * 100,
#                                      model.score((ampl_t[:, id1] + ampl_t[:, id2]).reshape((-1, 1)),
#                                                  (y_out[:, id1] + y_out[:, id2]).reshape((-1, 1))),
#                                      #model.intercept_[0],
#                                      model.coef_[0][0]
#                                      ]
#         idy += 1
#     df_t['Model'] = models[idx]
#     df = df.append(df_t)
#     errors_combined['Model'] = models[idx]
#     errors_combined_all = errors_combined_all.append(errors_combined)
# for id1, id2, name in zip(ids1, ids2, names):
#     ax = sns.lmplot(x='True('+name+')', y='Predicted('+name+')', data=df, hue='Model', legend=True, scatter_kws={'alpha': 0.6},
#                     line_kws={'lw': 1})
#     plot_iden(ax.axes[0, 0])
#     savefig(saving_dir+"name")
#     plt.show()
# errors_combined_all.to_csv(saving_dir + "_errors_combined.csv")


# ax = sns.lmplot(x='True Cr', y='Cr', data=df_met, hue='Model', legend=True,scatter_kws={'alpha':0.6},line_kws={'lw': 1})
# ax.axes[0,0].axline((0.5, 0.5), slope=1, ls="--", zorder=0, color='silver')
# plt.show()





# sns.violinplot(x='Model', y='value', data=df_met, hue='type',
#                palette="Set3",
#                linewidth=1,
#                split=True,
#                inner="quartile")










# df = pd.DataFrame()
# df_t = pd.DataFrame()
# loi = met_name + ['Frequency', 'Damping', 'Phase']
# # loi_pred = _pred + ['Predicted Frequency', 'Predicted Damping', 'Predicted Phase']
# # loi_true = _true + ['True Frequency', 'True Damping', 'True Phase']
# loi_error = _error + ['Error(Frequency)', 'Error(Damping)', 'Error(Phase)']
# for idx,rslt_wiithout_ph_1 in enumerate(rslt_list):
#     df_t[loi] = rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type']=='Predicted'][loi] - rslt_wiithout_ph_1.loc[rslt_wiithout_ph_1['type'] == 'True'][loi]
#     df_t['Model'] = models[idx]
#     df = df.append(df_t)
# df['SNR'] = rslt_wiithout_ph_1['SNR']



