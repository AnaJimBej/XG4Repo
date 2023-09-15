import sys

import numpy as np
import pandas as pd
import json
from scipy import stats
import matplotlib.pyplot as plt
import operator
import statistics
from statsmodels.stats.proportion import proportion_confint
import seaborn as sns
import Data_management_code.analyse_kg as analyse_kg


def get_train(train_file):
    train_df = pd.read_csv(train_file,sep='\t',names=['head','relation','tail'])

    df_relation = train_df.groupby('relation').count()
    df_relation = df_relation.reset_index()
    df_relation = df_relation.sort_values(by=['head'])
    df_relation['head_norm'] = df_relation['head']/len(train_df)
    #head_tail_df = get_head_tail_ratio(train_df)
    head_tail_df = []
    return df_relation,head_tail_df
def weighted_harmonic_mean(a, weights = None):
    """
    Calculate weighted harmonic mean.

    :param a:
        the array
    :param weights:
        the weight for individual array members

    :return:
        the weighted harmonic mean over the array

    .. seealso::
        https://en.wikipedia.org/wiki/Harmonic_mean#Weighted_harmonic_mean
    """
    if weights is None:
        return stats.hmean(a)

    # normalize weights
    weights = weights.astype(float)
    weights = weights / weights.sum()
    # calculate weighted harmonic mean
    return np.reciprocal(np.average(np.reciprocal(a.astype(float)), weights=weights))

def get_relation_metrics(df_test, relation):

    # Computes the metrics of an experiment given the ranks
    df = df_test
    rslt_df = df[df['relation'] == relation]
    ranks = rslt_df['rank'] + 1
    mrr2 = np.reciprocal(weighted_harmonic_mean(a=ranks.fillna(sys.maxsize))).item() # Pessimistic
    h1 = sum(ranks<=1)/len(ranks)
    h3 = sum(ranks <= 3) / len(ranks)
    h10 = sum(ranks <= 10) / len(ranks)
    h20 = sum(ranks <= 20) / len(ranks)
    h50 = sum(ranks <= 50) / len(ranks)
    h100 = sum(ranks <= 50) / len(ranks)
    return [mrr2,h1, h3, h10, h20, h50,h100]

def plot_bar1(df,x,y,title,kind='bar'):
    # Plots a dataframe
    ax = df.plot(x, y,kind, rot=45)
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.title(title)
    plt.yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.ylim((0, 0.5))
    plt.show()
def plot_relation_result(df_test,title,df_relation,head_tail_df):
    """
    Plots the results for a relation
    :param df_test:
    :param title:
    :param df_relation:
    :param head_tail_df:
    :return:
    """
    results = []
    for relation in df_test['relation'].unique():
        mrri = get_relation_metrics(df_test,relation)
        results.append([relation]+mrri)
    results = sorted(results, key=operator.itemgetter(1),reverse= True)
    result = pd.DataFrame(results,columns=['relation','MRR','Hits@1','Hits@3','Hits@10','Hits@20','Hits@50','Hits@100'])
    x = 'relation'
    y = ['MRR','Hits@1','Hits@3','Hits@10','Hits@100']
    plot_bar(result, x, y, title)
    #plot_relation(result, x, y, title + ' and Relations',df_relation,head_tail_df)




def read_minerva(result_file):
    """
    Reads a file of Minerva
    :param result_file:
    :return:
    """
    with open(result_file, 'r') as f:
        df_test = pd.DataFrame(data=json.load(f))
    df_test = df_test.transpose()


    return df_test

def read_rnnlogic(result_file):
    """
    Reads a file with RNNLogic results
    :param result_file:
    :return:
    """
    df = pd.read_csv(result_file, sep=';',names = ['split','head_norm','relation','tail','Lowrank','rank'])
    df = df.loc[df['split'] =='test']
    df['rank'] = df['rank']-2
    df['start'] = df['head_norm']


    return df

def get_mrrs(df_test):
    """
    Computes the MRR for three different cases
    :param df_test:
    :return:
    """
    ranks = df_test['rank'] + 1
    #ranks_i = 1/ranks
    mrr1 = np.reciprocal(weighted_harmonic_mean(a=ranks.fillna(2400))).item()
    mrr2 = np.reciprocal(weighted_harmonic_mean(a=ranks.fillna(sys.maxsize))).item()
    mrr3 = np.reciprocal(weighted_harmonic_mean(a=ranks.fillna(100))).item()
    print(mrr1)
    print(mrr2)
    print(mrr3)

def plot_relation(df,x,y,title,df_relation,head_tail_df):
    """
    Plots the head tail ratio and the number of training samples for each relation
    :param df:
    :param x:
    :param y:
    :param title:
    :param df_relation:
    :param head_tail_df:
    :return:
    """
    df_merge = pd.merge(df, df_relation, how='inner')

    #df_merge = df_merge.merge(head_tail_df, on='relation')


    ax1 = df_merge[[x] + ['head_norm']].plot(
        x='relation', linestyle='-', marker='o')
    ax1.set_yscale("log")


    df_merge[['relation']+ y ].plot( kind='bar',ax=ax1)
    df_merge[['relation', 'ratio_norm']].plot(x='relation', linestyle='-', marker='s', ax=ax1)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.ylim((0, 1.1))
    plt.show()

def different_axis(df_merge,y,x='relation'):
    """
    Creates a plot where left and right axis mean different things
    :param df_merge:
    :param y:
    :param x:
    :return:
    """
    # Creating plot with dataset_1
    fig, ax1 = plt.subplots()

    ax1 = df_merge[[x] + ['head_norm']].plot(
        x='relation', linestyle='-', marker='o')

    # Adding Twin Axes to plot using dataset_2
    ax2 = ax1.twinx()

    ax2 = df_merge[['relation']+ y ].plot(x='relation', kind='bar',ax=ax1)
    ax2.set_yscale("log")

    # Adding title
    plt.title('Use different y-axes on the left and right of a Matplotlib plot', fontweight="bold")
    plt.ylim((0, 1.1))
    # Show plot
    plt.show()

def get_head_tail_ratio(df):
    """
    Obtains the average number of heads and tails and plots it
    :param df:
    :return:
    """
    relation_heads = df.groupby('relation')['head'].unique()
    relation_heads_count = {}
    for r, nodes in relation_heads.items():
        relation_heads_count[r] = len(nodes)

    relation_tails = df.groupby('relation')['tail'].unique()
    relation_tails_count = {}
    for r, nodes in relation_tails.items():

        relation_tails_count[r] = len(nodes)


# Heads per tail
    relation_ratio_dict = {}
    for r,nodes in relation_tails.items():
        list_r = []
        for tail in relation_tails[r]:
            rslt_df = df.loc[(df['relation'] == r) & (df['tail'] == tail)]
            list_r.append(len(rslt_df))
        relation_ratio = np.average(list_r)
        relation_ratio_dict[r] = relation_ratio
        ratio_df = pd.DataFrame.from_dict(relation_ratio_dict, orient='index')
    ratio_df.rename(columns={0: 'heads_per_tail'}, inplace=True)
    # apply normalization techniques by Column 1
    column ='heads_per_tail'
    ratio_df['tail_norm'] = (ratio_df[column] - ratio_df[column].min()) / (
                ratio_df[column].max() - ratio_df[column].min())

# Tails per head
    relation_ratio_dict = {}
    for r,nodes in relation_heads.items():
        list_r = []
        for head in relation_heads[r]:
            rslt_df = df.loc[(df['relation'] == r) & (df['head'] == head)]
            list_r.append(len(rslt_df))
        relation_ratio = np.average(list_r)
        relation_ratio_dict[r] = relation_ratio
        ratio_df_2 = pd.DataFrame.from_dict(relation_ratio_dict, orient='index')
    ratio_df_2.rename(columns={0: 'tail_per_head'}, inplace=True)
    ratio_df = ratio_df.join(ratio_df_2)
    # apply normalization techniques by Column 1
    column = 'tail_per_head'
    # ratio_df['head_norm'] = (ratio_df[column] - ratio_df[column].min()) / (
    #             ratio_df[column].max() - ratio_df[column].min())
    # ratio_df['ratio_tail_head_avg'] = ratio_df['tail_avg'] / ratio_df['head_avg']
    ratio_df = ratio_df.reset_index()
    ratio_df.rename(columns={'index': 'relation'}, inplace=True)
    #ratio_df = ratio_df.set_index('relation')

    # plt.plot(ratio_df.index,ratio_df['tail_per_head'],'.')
    # plt.xticks(rotation=45, ha='right')
    # #plt.yscale('log')
    # plt.tight_layout()
    # plt.show()
    # head_tail_df = pd.DataFrame.from_dict(relation_heads_count, orient='index',columns=['head'])
    # head_tail_df = head_tail_df.reset_index()
    # head_tail_df.rename(columns={head_tail_df.columns[0]: 'relation'}, inplace=True)
    # head_tail_df['tail'] = head_tail_df['relation'].map(relation_tails_count)
    # head_tail_df['ratio'] = head_tail_df['head'] / head_tail_df['tail']

    # # copy the data
    # df_min_max_scaled = head_tail_df.copy()
    #
    # # apply normalization techniques by Column 1
    # column = 'ratio'
    # df_min_max_scaled['ratio_norm'] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (
    #             df_min_max_scaled[column].max() - df_min_max_scaled[column].min())

    return ratio_df #df_min_max_scaled

def plot_ratio_avg(head_tail_df):
    """
    Classifies the relations according to the average heads per tail and tails per head
    :param head_tail_df:
    :return:
    """
    head_tail_df['diff'] = np.abs(head_tail_df['heads_per_tail'] - head_tail_df['tail_per_head'])
    x = np.array(head_tail_df['relation'])
    h_t = np.array(head_tail_df['heads_per_tail'])
    t_h = np.array((head_tail_df['tail_per_head']))
    diff = np.array((head_tail_df['diff']))

    upper = 3.2

    MM = np.ma.masked_where((h_t > upper) & (t_h > upper) & (diff<2), x)
    oneM = np.ma.masked_where((h_t < upper) & (t_h > upper)& (diff>2), x)
    Mone = np.ma.masked_where((h_t > upper) & (t_h < upper)& (diff>2), x)
    oneone = np.ma.masked_where((h_t < upper) & (t_h < upper)& (diff<2), x)

    MM_rel = x[MM.mask]
    oneM_rel = x[oneM.mask]
    Mone_rel = x[Mone.mask]
    oneone_rel = x[oneone.mask]
    print(MM_rel)
    print(oneM_rel)
    print(Mone_rel)
    print(oneone_rel)

    fig, ax = plt.subplots()
    y = ['heads_per_tail','tail_per_head']
    g = sns.scatterplot(data=head_tail_df, hue='relation', x='tail_per_head', y='heads_per_tail', palette="deep", size='diff')
    h, l = g.get_legend_handles_labels()
    plt.legend(h[0:20],l[0:20],loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.savefig('scatter.png')
    #plt.ylim((-0.02, 1.1))
    #plt.xlim((-0.05, 0.9))
    plt.title('Heads per tail and tails per head. Size is the difference between both.')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def plot_ratio(head_tail_df):
    """
    Plots the heads tail ratio when it was computed as the division of both concepts
    :param head_tail_df:
    :return:
    """
    x = np.array(head_tail_df['relation'])
    yy = np.array(head_tail_df['ratio'])
    y = np.array((head_tail_df['ratio_norm']))
    upper = 1.5
    lower = 0.67

    supper = np.ma.masked_where(yy < upper, 5)
    slower = np.ma.masked_where(yy > lower, 2)
    smiddle = np.ma.masked_where((yy < lower) | (yy > upper), 1)

    fig, ax = plt.subplots()
    ax.plot(x, smiddle, x, slower, x, supper,kind = 'bar')
    label = np.repeat('1 to 1',len(yy))
    aa = yy*0+5
    aa[yy >= upper] = 10
    label[yy >= upper] ='M to 1'
    aa[yy <= lower] = 0
    label[yy <= lower] = '1 to M'
    plt.scatter(x,y,c=aa,label = label)
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def merge2compare(df_1,df_2=None,y=None,df_3 = None):
    """
    Mereges the dataframes that include the results for different models
    :param df_1:
    :param df_2:
    :param y:
    :param df_3:
    :return:
    """
    results1 = []
    results2 = []
    results3 = []
    for relation in df_1['relation'].unique():
        mrr_i1 = get_relation_metrics(df_1, relation)
        results1.append([relation, mrr_i1[0]])
        if df_2 is not None:
            if relation in df_2['relation'].unique():
                mrr_i2 = get_relation_metrics(df_2, relation)
                results2.append([relation, mrr_i2[0]])
            else:
                results2.append([relation, 0])
        if df_3 is not None:
            if relation in df_3['relation'].unique():
                mrr_i3 = get_relation_metrics(df_3, relation)
                results3.append([relation, mrr_i3[0]])
            else:
                results3.append([relation, 0])



    results1 = sorted(results1, key=operator.itemgetter(1), reverse=True)
    result1 = pd.DataFrame(results1, columns=['relation', y[0]])
    n1 = pd.DataFrame(df_1.groupby('relation').count())['start']
    result1 = result1.merge(n1, on='relation')
    result1 = result1.rename(columns={"start": "n1"})

    if df_2 is not None:
        results2 = sorted(results2, key=operator.itemgetter(1), reverse=True)
        result2 = pd.DataFrame(results2, columns=['relation', y[1]])
        n2 = pd.DataFrame(df_2.groupby('relation').count())['start']
        result2 = result2.merge(n2, on='relation')
        result2 = result2.rename(columns={"start": "n2"})
        df_merge = pd.merge(result1, result2, how='inner')
    else:
        df_merge = result1
    if df_3 is not None:
        results3 = sorted(results3, key=operator.itemgetter(1), reverse=True)
        result3 = pd.DataFrame(results3, columns=['relation', y[2]])
        n3 = pd.DataFrame(df_3.groupby('relation').count())['start']
        result3 = result3.merge(n2, on='relation')
        result3 = result3.rename(columns={"start": "n3"})
        df_merge = pd.merge(df_merge, result3, how='inner')



    return df_merge


def compare_mrr(df_1,df_2,title,df_relation,head_tail_df,y):
    """
    Compare the MRR of different models
    :param df_1:
    :param df_2:
    :param title:
    :param df_relation:
    :param head_tail_df:
    :param y:
    :return:
    """
    results1 = []
    results2 = []
    for relation in df_1['relation'].unique():
        mrr_i1 = get_relation_metrics(df_1,relation)
        results1.append([relation,mrr_i1[0]])
        mrr_i2 = get_relation_metrics(df_2,relation)
        results2.append([relation,mrr_i2[0]])


    results1 = sorted(results1, key=operator.itemgetter(1), reverse=True)
    result1 = pd.DataFrame(results1,columns=['relation',y[0]])

    results2 = sorted(results2, key=operator.itemgetter(1), reverse=True)
    result2 = pd.DataFrame(results2, columns=['relation', y[1]])

    df_merge = pd.merge(result1, result2, how='inner')
    x = 'relation'
    plot_bar(df_merge,x,y,title,kind='bar')
    plot_relation(df_merge,x,y,title,df_relation,head_tail_df)


def confidence_interval(p, n):
    """
    Computes the confidence interval of the binomial using Pearson Clopper approximation
    :param p:
    :param n:
    :return:
    """
    l_bound, h_bound = proportion_confint(count=(n*p).astype('int'), nobs=n, alpha=0.1, method='beta')

    # if isinstance(l_bound, pd.Series):
    #     l_bound = l_bound[0]
    #     h_bound = h_bound[0]
    return l_bound, h_bound

def assymetric_ci(df, n_test,name_list,title,legend):
    """
    Plots assymetric confidence interval
    :param df:
    :param n_test:
    :param name_list:
    :param title:
    :param legend:
    :return:
    """
    model_array_list = []
    low = []
    high = []
    mrr = []
    relation_array_list = []
    i = 0
    for column in df.columns:
        [lowi, upi] = confidence_interval(df[column], n_test[i])
        i +=1
        model_array_list.append(np.repeat(column, len(df), axis=None))
        low.append(lowi)
        high.append(upi)
        mrr.append(df[column])
        relation_array_list.append(list(df.index))

    relation_array = np.concatenate(relation_array_list)
    model_array = np.concatenate(model_array_list)

    ix3 = pd.MultiIndex.from_arrays([model_array, relation_array], names=['Model', 'Relation'])
    df3 = pd.DataFrame({'MRR': np.concatenate(mrr), 'low': np.concatenate(low),
                        'up': np.concatenate(high)}, index=ix3)
    df_multi = df3.groupby(level=['Model', 'Relation'],sort=False).sum()
    df_multi.columns = ['mrr', 'errlo', 'errhi']

    my_index = df_multi.index
    #new_index = [my_index[0],my_index[2],my_index[4],my_index[6],my_index[1],my_index[3],my_index[5]]
    #df_multi = df_multi.reindex(new_index)
    df_multi['errlo2'] = df_multi['mrr'] - df_multi['errlo']
    df_multi['errhi2'] = df_multi['errhi'] - df_multi['mrr']

    data = df_multi.unstack(level='Model')
    data = data.reindex(relation_array_list[0])
    error_list = []
    for i in range(len(df.columns)):
        err_i = data[['errlo2', 'errhi2']].T.values[[i, i+len(df.columns)]]
        error_list.append(err_i.T)

    # err1 = data[['errlo2', 'errhi2']].T.values[[0, 2]]
    # err2 = data[['errlo2', 'errhi2']].T.values[[1, 3]]
    err = np.dstack(error_list).T
    ax = data['mrr'].plot(kind='line', yerr=err)
    #ax.legend(legend)
    plt.xticks(rotation=45, ha='right',fontsize='7')
    plt.title(title)
    plt.tight_layout()
    plt.ylim((0, 0.9))
    plt.xlabel('metric')
    plt.tight_layout()
    #tikzplotlib.save('test.tex')
    #mlp.rcParams.update(mlp.rcParamsDeFault)
    plt.show()

def scatter_result_ratio(df,y):
    """
    Plots the scatterplot of the metrics and the averaged number of tails per head
    :param df:
    :param y:
    :return:
    """
    fig, ax = plt.subplots()
    g = sns.scatterplot(data=df, hue='relation', x=y, y='tail_per_head', palette="deep", size='heads_per_tail')
    h, l = g.get_legend_handles_labels()
    plt.legend(h[0:20],l[0:20],loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.savefig('scatter.png')
    #plt.ylim((-0.02, 1.1))
    plt.xlim((-0.05, 0.9))
    plt.xlabel('MRR: '+ y)
    plt.ylabel('Average number of tails per head' )
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()
def compare_with_errors(df_1, df_2,y=['MRR - Minerva', 'MRR - RNNLogic'],df_3= None):
    """
    Compares the results of two models (or configurations) including the assymetric confidence intervals
    :param df_1:
    :param df_2:
    :param y:
    :param df_3:
    :return:
    """
    df = merge2compare(df_1, df_2,y,df_3)
    df = df.set_index('relation')
    df = df.sort_values(by = y[0])
    if df_3 is not None:
        n_list = (df['n1'],df['n2'],df['n3'])
    elif df_2 is not None:
        n_list = (df['n1'], df['n2'])
    else:
        n_list = [df['n1']]
    assymetric_ci(df[y], n_list, 'name_list', 'title', y)

    if True:
        df_scatter = df[y].join(head_tail_df.set_index('relation'))
        df_scatter = df_scatter.join(df_relation.set_index('relation'))
        df_scatter = df_scatter.reset_index()
        scatter_result_ratio(df_scatter, y[0])
        if df_2 is not None:
            scatter_result_ratio(df_scatter, y[1])
        if df_3 is not None:
            scatter_result_ratio(df_scatter, y[2])


def read_some_rnn(file,n):
    """
    Reads the files of RNNLogic when there are several executions of the same configuration (for variance reduction)
    :param file:
    :param n:
    :return:
    """
    df_list = []
    for i in range(n):
        file_i = file.split('.')[0] +'_'+str(i+1)+'.csv'
        df = read_rnnlogic(file_i)
        df_list.append(df)
    df_final = pd.concat(df_list)
    return df_final

def read_some_minerva(file,n):
    """
    Reads the files of Minerva when there are several executions of the same configuration (for variance reduction)
    :param file:
    :param n:
    :return:
    """
    df_list = []
    for i in range(n):
        file_i = file.split('.')[0] +str(i+1)+'.json'
        df = read_minerva(file_i)
        df_list.append(df)
    df_final = pd.concat(df_list)
    return df_final

def get_results(split, predictor, all_results):
    return all_results.loc[(all_results['split'] == split) & (all_results['predictor'] == predictor)]

def read_several_trials_minerva(folder, file):
    result = pd.read_csv(folder + file, sep=';',
                         names=['model_name', 'iteration', 'hits1', 'hits3', 'hits10', 'hits20', 'mrr'])
    return result.groupby('iteration').mean()

def read_trials_rnnlogic2(folder, files, predictor,n=2):
    result_list = {}
    name_list = []
    legend = []
    for file in files:
        name1 = 'N : ' + file.split('_')[2] + ' K : ' + file.split('_')[3]
        name = int(file.split('_')[2]) + int(file.split('_')[3])
        # name = (int(file.split('_')[2]))
        # name = file.split(' ')[-1]

        legend.append((int(file.split('_')[2]), int(file.split('_')[3])))
        # legend.append((name1))
        mrr_list = []
        for i in range(n):
            file_i = file.split('.')[0] + str(i + 1) + '.csv'

            result = pd.read_csv(folder + file_i, sep=';',
                                 names=['model_name', 'predictor', 'iteration', 'split', 'hits1', 'hits3', 'hits10',
                                        'mr',
                                        'mrr'])
            test = get_results('test', predictor, result)
            metrics  = ['hits1', 'hits3', 'hits10','mrr']
            mrr_list.append(test[metrics].iloc[-1])
            mrr = np.average(np.array(mrr_list), axis=0)
        result_list[name] = mrr

    legend.sort()
    # legend = ['Train direct, test direct', 'Train inverse, test inverse', 'Train direct, test inverse']
    result_df = pd.DataFrame(result_list)
    if predictor == 'EM':
        pred_name = 'RNNLogic'
    else:
        pred_name = 'RNNLogic+'
    result_df = result_df.rename(columns={result_df.columns[0]: pred_name})
    result_df['metric']=metrics
    result_df = result_df.set_index('metric')
    return result_df, name_list, legend




def compare_metrics_model(folderM, fileM, folderR, fileR,Plus=None):
    result_m = read_several_trials_minerva(folderM, fileM)
    def_result_m = pd.DataFrame(result_m.iloc[-1])
    def_result_m = def_result_m.T
    def_result_m = def_result_m[['hits1', 'hits3', 'hits10','mrr']]
    def_result_m = def_result_m.T
    def_result_m = def_result_m.rename(columns={def_result_m.columns[0]: 'Minerva'})

    result_df_r, name_list,legend = read_trials_rnnlogic(folderR, fileR, 'EM', n=4) # The function was renamed
    result_df_plus, name_list,legend = read_trials_rnnlogic(folderR, fileR, 'PLUS', n=4)
    df_merge2 = result_df_r.join(def_result_m)
    if Plus is not None:
        df_merge3 = df_merge2.join(result_df_plus)
        assymetric_ci(df_merge3, [1283*4,1283*4,1283*4], name_list, 'title', 'legend')
    else:
        assymetric_ci(df_merge2, [1283 * 4, 1283 * 4, 1283 * 4], name_list, 'title', 'legend')

def get_results(split,predictor,all_results):
    return all_results.loc[(all_results['split']==split) & (all_results['predictor']==predictor)]

def plot_bar(df,x,y,title,kind='bar'):
    # plt.style.use("ggplot")


    ax = df.plot(x, y, kind, rot=45)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    title = title.replace(' - ','_')
    title = title.replace(' : ','_')
    title = title.replace(' ','_')
    plt.ylim((0,0.5))
    #plt.savefig(title + '.png')

    plt.show()

def plot_evolution_from_filtered_file(result_file,title,predictor,n_test=85):
    """
    Esta función se utiliza para plotear la evolución cuando el fichero de entrada es uno de ficheros que he
    modificado para filtrar las tripletas de test en las que no aparece la head en el subgrafo.

    :param result_file:
    :param title:
    :param predictor:
    :param n_test:
    :return:
    """
    result = pd.read_csv(result_file, sep=';',index_col=0,names = ['model_name','predictor', 'iteration', 'split', '1hits1', '2hits3', '3hits10', '4mr', '5mrr','n_test'])

    test_em = get_results('test', predictor, result)
    n_test = list(result['n_test'])[0]
    x = 'iteration'
    y = ['5mrr','1hits1','2hits3','3hits10']
    legend = ['hits@1','hits@3','hits@10','MRR']

    test_em['iters'] = range(len(test_em))
    test_em = test_em.set_index('iters')

    assymetric_ci(test_em[y], n_test, y, title, legend)

def plot_evolution_RNNLogic(result_file,title,n_test=85):
    result = pd.read_csv(result_file, sep=';',index_col=0,names = ['model_name','predictor', 'iteration', 'split', '1hits1', '2hits3', '3hits10', '4mr', '5mrr'])

    test_em = get_results('test', 'EM', result)
    test_plus = get_results('test', 'PLUS', result)
    valid_em = get_results('valid', 'EM', result)
    valid_plus = get_results('valid', 'PLUS', result)
    x = 'iteration'
    y = ['5mrr','1hits1','2hits3','3hits10']
    legend = ['hits@1','hits@3','hits@10','MRR']
    #plot_bar(valid_em,x, y, title + ' EM validation results')
    #plot_bar(test_em, x, y, title +' EM test results')
    #plot_bar(valid_plus, x, y, title +' Predictor + validation results')
    #plot_bar(test_plus, x, y,title + ' Predictor + test results')
    test_em['iters'] = range(len(test_em))
    test_em = test_em.set_index('iters')
    test_plus['iters'] = range(len(test_plus))
    test_plus = test_plus.set_index('iters')
    assymetric_ci(test_em[y], n_test, y, title, legend)
    assymetric_ci(test_plus[y], n_test, y, title, legend)

def compare_metric_trials_rnnlogic(folder, files, title, n_test, n=2):
    result_list = []
    result_dict = {}
    name_list = []
    legend = []
    x = 'model_name'
    y = ['mrr','hits1','hits3','hits10']
    for file in files:
        name1 = 'N : ' + file.split('_')[2] + ' K : ' + file.split('_')[3]
        name = int(file.split('_')[2]) + int(file.split('_')[3])
        # name = (int(file.split('_')[2]))
        # name = file.split(' ')[-1]

        #name = file.split('_')[-3] +file.split('_')[-2]
        #legend.append(name)
        legend.append((int(file.split('_')[2]), int(file.split('_')[3])))
        # legend.append((name1))
        mrr_list = []
        result_list = []
        for i in range(n):
            file_i = file.split('.')[0] + str(i + 1) + '.csv'

            result = pd.read_csv(folder + file_i, sep=';',
                                 names=['model_name', 'predictor', 'iteration', 'split', 'hits1', 'hits3', 'hits10',
                                        'mr',
                                        'mrr'])
            test = get_results('test', 'EM', result)
            def_result = test.iloc[-1]
            result_list.append(def_result[y])
            print(file)
            print(def_result[y])

        results = np.average(np.array(result_list), axis=0)
        result_dict[name] = results

    legend.sort()
    result_df = pd.DataFrame(result_dict)

    result_df['metrics'] = y
    result_df = result_df.set_index('metrics')
    result_df= result_df.T
    result_df = result_df.reset_index()
    result_df = result_df.rename(columns={"index": "model_name"})
    result_df = result_df.set_index('model_name')
    # result_df = result_df.reset_index()
    # result_df = result_df.reset_index()
    # result_df.rename(columns={result_df.columns[0]: 'iteration'}, inplace=True)
    assymetric_ci_model(result_df[y].T, n_test, y, title,legend)

def compare_models_RNNLogic(folder,files,predictor,title,ntest=1283):
    result_list = []
    for file in files:
        result = pd.read_csv(folder + file, sep=';',
                             names=['model_name','predictor', 'iteration', 'split', 'hits1', 'hits3', 'hits10', 'mr', 'mrr'])
        test = get_results('test', predictor, result)
        def_result = test.iloc[-1]
        result_list.append(def_result)
    result_df = pd.DataFrame(result_list)
    x = 'model_name'
    y = ['mrr','hits1','hits3','hits10']
    result_df = result_df.set_index('model_name')
    #plot_bar(result_df,x,y,title)
    legend = list(result_df.index)
    assymetric_ci_model(result_df[y].T, ntest, y, title,legend)


def compare_models_RNNLogic_evolution(folder,files,predictor,title,ntest=1283):
    """
    Compares the evolution of the mrr of different models. I want to see if the model converges
    :param folder:
    :param files:
    :param predictor:
    :param title:
    :param ntest:
    :return:
    """
    result_list = []
    for file in files:
        print(file)
        result = pd.read_csv(folder + file, sep=';',
                             names=['model_name','predictor', 'iteration', 'split', 'hits1', 'hits3', 'hits10', 'mr', 'mrr'])
        test = get_results('valid', predictor, result)
        test = test.set_index('iteration')
        def_result = test['mrr']
        def_result = def_result.append(pd.Series(list(test['model_name'])[0], index=['model_name']))
        result_list.append(def_result)
    result_df = pd.DataFrame(result_list)
    result_df = result_df.fillna(0)
    x = 'model_name'
    y = list(np.arange(1,len(def_result)))
    result_df = result_df.set_index('model_name')
    #plot_bar(result_df,x,y,title)
    legend = list(result_df.index)
    assymetric_ci_model(result_df[y].T, ntest, y, title,legend)

def assymetric_ci_model(df, n_test,name_list,title,legend):
    model_array_list = []
    low = []
    high = []
    mrr = []
    relation_array_list = []
    for column in df.columns:
        [lowi, upi] = confidence_interval(df[column], n_test)
        model_array_list.append(np.repeat(column, len(df), axis=None))
        low.append(lowi)
        high.append(upi)
        mrr.append(df[column].astype('float'))
        relation_array_list.append(list(df.index))

    relation_array = np.concatenate(relation_array_list)
    model_array = np.concatenate(model_array_list)

    ix3 = pd.MultiIndex.from_arrays([model_array, relation_array], names=['Color (columna)', 'Bloque (filas)'])
    df3 = pd.DataFrame({'MRR': np.concatenate(mrr), 'low': np.concatenate(low),
                        'up': np.concatenate(high)}, index=ix3)
    df_multi = df3.groupby(level=['Color (columna)', 'Bloque (filas)'], sort=False).sum()
    df_multi.columns = ['mrr', 'errlo', 'errhi']

    df_multi['errlo2'] = df_multi['mrr'] - df_multi['errlo']
    df_multi['errhi2'] = df_multi['errhi'] - df_multi['mrr']

    data = df_multi.unstack(level='Color (columna)')
    data = data.reindex(np.array(name_list))
    error_list = []
    for i in range(len(df.columns)):
        err_i = data[['errlo2', 'errhi2']].T.values[[i, i+len(df.columns)]]
        error_list.append(err_i.T)

    err1 = data[['errlo2', 'errhi2']].T.values[[0, 2]]
    err2 = data[['errlo2', 'errhi2']].T.values[[1, 3]]
    err = np.dstack(error_list).T
    ax = data['mrr'].plot(kind='bar', yerr=err, legend = False)
    ax.legend(legend,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.xlabel('metrics')
    plt.ylabel('value')
    plt.ylim((0, 0.4))
    plt.tight_layout()
    # plt.savefig(title)
    # tikzplotlib.save('test.tex')
    plt.show()
    plt.close()

def assymetric_ci(df, n_test,name_list,title,legend):
    model_array_list = []
    low = []
    high = []
    mrr = []
    relation_array_list = []
    for column in df.columns:
        [lowi, upi] = confidence_interval(df[column], n_test)
        model_array_list.append(np.repeat(column, len(df), axis=None))
        low.append(lowi)
        high.append(upi)
        mrr.append(df[column])
        relation_array_list.append(list(df.index))

    relation_array = np.concatenate(relation_array_list)
    model_array = np.concatenate(model_array_list)

    ix3 = pd.MultiIndex.from_arrays([model_array, relation_array], names=['Model', 'Iteration'])
    df3 = pd.DataFrame({'MRR': np.concatenate(mrr), 'low': np.concatenate(low),
                        'up': np.concatenate(high)}, index=ix3)
    df_multi = df3.groupby(level=['Model', 'Iteration'],sort = False).sum()

    df_multi.columns = ['mrr', 'errlo', 'errhi']

    my_index = df_multi.index
    #new_index = [my_index[0],my_index[2],my_index[4],my_index[6],my_index[1],my_index[3],my_index[5]]
    #df_multi = df_multi.reindex(new_index)
    df_multi['errlo2'] = df_multi['mrr'] - df_multi['errlo']
    df_multi['errhi2'] = df_multi['errhi'] - df_multi['mrr']

    data = df_multi.unstack(level='Model')
    data2 = df_multi.unstack(level='Model').reindex(df_multi.index)
    error_list = []
    for i in range(len(df.columns)):
        err_i = data[['errlo2', 'errhi2']].T.values[[i, i+len(df.columns)]]
        error_list.append(err_i.T)

    err1 = data[['errlo2', 'errhi2']].T.values[[0, 2]]
    err2 = data[['errlo2', 'errhi2']].T.values[[1, 3]]
    err = np.dstack(error_list).T
    #fig, ax = plt.subplots(figsize=(15, 5))
    fig, ax = plt.subplots()

    ax = data['mrr'].plot(kind='bar', yerr=err,ax=ax)
    ax.legend(legend,loc='center left', bbox_to_anchor=(1, 0.5),title="Metrics")
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.ylim((0, 0.5))
    plt.tight_layout()
    #tikzplotlib.save('test.tex')
    #mlp.rcParams.update(mlp.rcParamsDeFault)
    plt.show()


def read_trials_rnnlogic1(folder,files,title, n_test,n=2):
    result_list = {}
    name_list = []
    legend = []
    for file in files:
        #name1 = 'N : ' + file.split('_')[2] + ' K : ' + file.split('_')[3]
        #name = int(file.split('_')[2]) + int(file.split('_')[3])
        #name = (int(file.split('_')[2]))
        name = int(file.split('_')[-2])

        legend.append(name)
        #legend.append((name1))
        mrr_list = []
        for i in range(n):
            file_i = file.split('.')[0]  + str(i+1) + '.csv'

            result = pd.read_csv(folder + file_i, sep=';',
                                 names=['model_name', 'predictor', 'iteration', 'split', 'hits1', 'hits3', 'hits10', 'mr',
                                        'mrr'])
            test = get_results('test', 'PLUS', result)
            mrr_list.append(test['mrr'])
            mrr = np.average(np.array(mrr_list),axis=0)
        result_list[name] = mrr

    legend.sort()
    #legend = ['Train direct, test direct', 'Train inverse, test inverse', 'Train direct, test inverse']
    result_df = pd.DataFrame(result_list)
    return result_df, name_list,legend


def compare_metric_trials(folder,files,title, n_test,n=2,iterations = None):
    result_df, name_list,legend = read_trials_rnnlogic1(folder, files, title, n_test, n=2)

    #result_df = result_df.reset_index()
    #result_df = result_df.reset_index()
    #result_df.rename(columns={result_df.columns[0]: 'iteration'}, inplace=True)

    if iterations is not None:
        result_df = result_df[iterations-1:iterations]
    assymetric_ci(result_df, n_test,name_list,title,legend)

def compare_metric(folder,files,title):
    result_list = {}
    name_list = []
    legend = []
    for file in files:
        name1 = 'N rules : ' + file.split('_')[2] + ' Top rules : ' + file.split('_')[3]
        name = int(file.split('_')[2])
        # name = file.split(' ')[-1]
        legend.append(int(file.split('_')[2]))
        name_list.append(name)

        mrr_list = []
        file_i = file.split('.')[0]  + '.csv'

        result = pd.read_csv(folder + file_i, sep=';',
                             names=['model_name', 'predictor', 'iteration', 'split', 'hits1', 'hits3', 'hits10', 'mr',
                                    'mrr'])
        result = result.set_index('model_name')
        test = get_results('test', 'PLUS', result)
        mrr = (test['mrr'])
        result_list[(name)] = [mrr[0]]

    legend.sort()
    result_df = pd.DataFrame(result_list)
    result_df = result_df.reset_index()
    result_df = result_df.reset_index()
    result_df.rename(columns={result_df.columns[0]: 'iteration'}, inplace=True)
    x = 'iteration'
    y = ['mrr']
    #plot_bar(result_df, x, name_list,title)
    assymetric_ci(result_df[name_list], 1283,name_list,title,legend)


def filter_test_by_heads(dataset_dir,result_folder,last_iter_rank,outfile=''):
    # Get the nodes that are not included in train
    heads2filter = analyse_kg.not_common(dataset_dir)
    e_dict =  pd.read_csv(dataset_dir+'entities.dict', sep='\t',names=['id','relation'])
    e2id = e_dict.set_index('relation')
    e2id = e2id.to_dict()['id']
    heads2filter_id = [e2id[item] for item in heads2filter]

    # Para cada iteración
    ranks_file_name = last_iter_rank.split('_')
    niters = ranks_file_name[1]
    result_list = []
    for i in range(int(niters)):
        file_iter_1 = ranks_file_name[0]+'_'+str(i+1)+'_'+('_').join(ranks_file_name[2:])
        df_i = read_rnnlogic(result_folder+file_iter_1)
        # Filtrar las tripletas
        df_i_filtered = df_i[~df_i['head_norm'].isin(heads2filter_id)]

        # Calcular las métricas

        [mrr2,h1, h3, h10, h20, h50,h100]= get_relation_metrics(df_i_filtered,'CtD')
        result_list.append(('model_name',ranks_file_name[0],(i+1),list(df_i['split'])[0],h1,h3,h10,0,mrr2,len(df_i_filtered)))



    # Añadir a un dataframe igual que el de metrics
    metrics_df = pd.DataFrame(result_list)

    outfile = ('_').join(['metrics_results',ranks_file_name[0]]+ranks_file_name[2:])

    with open(result_folder+outfile, 'w') as f:
        metrics_df.to_csv(f,index = True, header=None, sep=';', mode='w', lineterminator='\n')