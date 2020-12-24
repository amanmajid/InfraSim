'''
    utils.py
        Series of functions to make life a bit easier

    @amanmajid
'''

import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['font.family']      = 'Arial'

def results_to_graph(results,**kwargs):

    # define graph
    G = nx.DiGraph()

    # define edges
    for i in results.index:
        # add edge if it has flow going through it
        if results.Value.loc[i] > 0:

            G.add_edge(results.Start.loc[i],
                       results.End.loc[i],
                       weight=results.Value.loc[i]+0.2,
                       label="{:.2f}".format(results.Value.loc[i]*100)+'%')

    # position graph in circular layout
    pos = nx.circular_layout(G)

    # get edge attributes
    edge_widths = nx.get_edge_attributes(G, 'weight')
    edge_labels = nx.get_edge_attributes(G, 'label')

    # draw nodes
    nx.draw_networkx_nodes(G,
                           pos,
                           nodelist=G.nodes(),
                           node_size=300,
                           node_color=kwargs.get('edge_color', 'black'),
                           alpha=1)

    # draw edges
    nx.draw_networkx_edges(G,
                           pos,
                           edgelist = edge_widths.keys(),
                           width=list(edge_widths.values()),
                           arrowsize=20,
                           edge_color=kwargs.get('edge_color', 'black'),
                           alpha=1)

    # draw node labels
    nx.draw_networkx_labels(G,
                            pos=pos,
                            labels=dict(zip(G.nodes(),G.nodes())),
                            font_color=kwargs.get('node_font_color', 'black'),
                            font_size=kwargs.get('node_font_size', 10))

    # draw edge labels
    nx.draw_networkx_edge_labels(G,
                                 pos,
                                 edge_labels=edge_labels,
                                 label_pos=0.5,
                                 #rot=90,
                                 font_size=kwargs.get('edge_font_size', 10))

    return G


def plot_storage_volume(storage_results,node,**kwargs):
    '''Plot storage volume at a given node
    '''
    v = storage_results[storage_results.Node==node].reset_index(drop=True)
    ax = v.Value.plot(color='blue')
    ax.set_xlabel('Time')
    ax.set_ylabel('Storage Volume')
    ax.set_xlim([kwargs.get("xlim", None)])
    ax.set_ylim([kwargs.get("ylim", None)])
    plt.axhline(y=v.Value.max(), color='gray', linestyle=':')


def plot_inflow_outflow(arcflow_result,node,**kwargs):
    '''Plot nodal inflow and outflows
    '''
    inflow  = arcflow_result[arcflow_result.End==node].groupby(by='Timestep').sum().reset_index(drop=True)
    outflow = arcflow_result[arcflow_result.Start==node].groupby(by='Timestep').sum().reset_index(drop=True)

    grouped = kwargs.get("grouped", True)

    if grouped==True:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (5,5)))
        ax.plot(inflow.Value,color='blue',label='Inflow')
        ax.plot(outflow.Value,color='red',label='Outflow')
        ax.set_xlabel('Time')
        ax.set_ylabel('Inflow-Outflow')
        plt.legend(loc='upper right')

    elif grouped==False:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (5,5)),nrows=2,ncols=1)
        plt.subplot(2,1,1)
        plt.plot(inflow.Value,color='blue',label='Inflow')
        plt.ylabel('Inflow-Outflow')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(outflow.Value,color='red',label='Outflow')
        plt.ylabel('Inflow-Outflow')
        plt.legend()
        plt.xlabel('Time')


def plot_average_production(region,results,ax,resolution='Hour',mismatch=False):
    
    """

    Parameters
    ----------
    region : str
        Region to plot (e.g. Israel, West Bank, Gaza, Jordan).
    results : pandas dataframe
        Dataframe of results from model run.
    resolution : str
        Time resolution to visualise [Hour,Day,Month]. The default is Hour.

    Returns
    -------
    None.

    """
    production = results
    if production.empty:
        print('No generation data to plot for ' + region)
    else:
        production = production.groupby(by=['Start',resolution]).mean().reset_index()
        production = production[['Start',resolution,'Value']]
        production.loc[production['Start'].str.contains('ccgt'), 'Start']         = 'CCGT'
        production.loc[production['Start'].str.contains('diesel'), 'Start']       = 'Diesel'
        production.loc[production['Start'].str.contains('natural gas'), 'Start']  = 'Natural Gas'
        production.loc[production['Start'].str.contains('coal'), 'Start']         = 'Coal'
        production.loc[production['Start'].str.contains('super_source'), 'Start'] = 'Shortage'

        production.loc[production['Start'].str.contains('new solar',case=False), 'Start']   = 'New Solar'
        production.loc[production['Start'].str.contains('solar',case=True), 'Start']        = 'Solar'
        production.loc[production['Start'].str.contains('new wind',case=False), 'Start']         = 'New Wind'
        production.loc[production['Start'].str.contains('wind',case=True), 'Start']         = 'Wind'

        production = production.groupby(by=['Start',resolution]).mean().reset_index()
        production = production.pivot(index=resolution, columns='Start', values=['Value'])
        production.columns = production.columns.get_level_values(1)
        production = production.drop(production[production.columns[production.sum()==0]].columns,axis=1)

        # mismatch
        if mismatch==True:
            try:
                production['Shortage'] = -production['Shortage']
            except:
                pass

        color_dict = {'Shortage'       : 'red',
                      'Natural Gas'    : 'peru',
                      'Diesel'         : 'dimgrey',
                      'Solar'          : 'teal',
                      'New Solar'      : 'magenta',
                      'Coal'           : 'darkgray',
                      'CCGT'           : 'silver',
                      'Wind'           : 'orange'}
        
        ax = production.plot(kind='bar',
                 stacked=True,
                 figsize=(8,6),
                 edgecolor='white',
                 ax=ax,
                 color=[color_dict.get(x, 'green') for x in production.columns])
        
        ax.set_axisbelow(True)
        ax.set_title(region,loc='left',fontweight='bold')
        #ax.set_xticks(rotation=0)
        ax.grid(True,linestyle=':')
        ax.set_ylabel('Average Daily Production (MWh)')
        #plt.legend(loc='upper left')

        if mismatch==True:
            ax.axhline(0,linewidth=1,linestyle='-',color='black')

    
def plot_energy_deficit(results,method='maximum'):
    
    """
    
    Parameters
    ----------
    results : pandas dataframe
        Dataframe of results from model run.
    method : str, optional
        Maximum or total deficit. The default is 'maximum'.

    Returns
    -------
    None.

    """
    
    production = results
    # get super source outflows 
    production = production[production.Start=='super_source'].reset_index(drop=True)
    # replace 
    production.loc[production['End'].str.contains('Israel'), 'End']     = 'Israel'
    production.loc[production['End'].str.contains('West Bank'), 'End']  = 'West Bank'
    production.loc[production['End'].str.contains('Jordan'), 'End']     = 'Jordan'
    production.loc[production['End'].str.contains('Gaza'), 'End']       = 'Gaza'
    # groupby method
    if method=='total':
        production = production.groupby(by='End').sum().Value
        ax = production.plot(kind='barh',color='coral',edgecolor='navy',)
        ax.set_xlabel('Total Shortage [MWh]')
    elif method=='maximum':
        production = production.groupby(by='End').max().Value
        ax = production.plot(kind='barh',color='coral',edgecolor='navy',)
        ax.set_xlabel('Maximum Shortage [MWh]')
    # plt formatting    
    ax.set_ylabel('Region')
    ax.set_axisbelow(True)
    plt.grid(True,linestyle=':')


def plot_ecdf(arcflow_result,country,ax):

    def ecdf(df,linestyle):
        x = np.sort(df.dropna().to_numpy())
        n = x.size
        y = np.arange(1, n+1) / n
        ax.plot(x,y,label=country,linestyle=linestyle)
        ax.set_xlabel('Shortage hours per day')
        ax.set_ylabel('P(x)')

    d = arcflow_result[arcflow_result.Start=='super_source'].reset_index(drop=True)
    d = d[d.End.str.contains(country)].reset_index(drop=True)
    d['Counter'] = 0
    d.loc[d.Value > 0,'Counter'] = 1
    d = d.groupby(by=['Day','Month','Year']).sum().reset_index(drop=True)
    ecdf(d.Counter,linestyle='-')


def plot_emissions_ts(run,emission_factors,label=None,color='blue'):
    run = run.results_arcflows
    run = run.drop(run[run.Start.str.contains('super_source')].index,axis=0)
    run = run.drop(run[run.Start.str.contains('generation')].index,axis=0)
    run.loc[run['Start'].str.contains('ccgt',case=False), 'Start']         = 'CCGT'
    run.loc[run['Start'].str.contains('diesel',case=False), 'Start']       = 'Diesel'
    run.loc[run['Start'].str.contains('natural gas',case=False), 'Start']  = 'Natural Gas'
    run.loc[run['Start'].str.contains('wind',case=False), 'Start']         = 'Wind'
    run.loc[run['Start'].str.contains('solar',case=False), 'Start']        = 'Solar'
    run.loc[run['Start'].str.contains('coal',case=False), 'Start']         = 'Coal'
    run = run.groupby(by=['Start','Timestep','Hour','Day','Month','Year']).sum().reset_index(drop=False)
    run.index = run.Start
    run['CO2'] = run['Value'].mul(pd.Series(emission_factors), axis=0)
    run = run.groupby(by=['Year']).mean().reset_index(drop=False)
    run['CO2'] = run['CO2'].divide(1000)
    run['CO2'].plot(label=label,color=color)
