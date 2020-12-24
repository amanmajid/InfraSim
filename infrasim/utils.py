'''
    utils.py
        Series of functions to make life a bit easier

    @amanmajid
'''

import os
import numpy as np
import pandas as pd
import geopandas as gpd

from . import spatial
from .metainfo import *
from .params import *


#---
# Conversions
#---

def kwh_to_gwh(v):
    '''Kilowatt hours to Gigawatt hours
    '''
    return v*0.000001

def gwh_to_kwh(v):
    '''Gigawatt hours to Kilowatt hours
    '''
    return v/0.000001

def cmd_to_mld(v):
    '''Cubic meters per day to megalitres per day
    '''
    return v/0.001

def mld_to_cmd(v):
    '''Megalitres per day to cubic meters per day
    '''
    return v*0.001

def lps_to_cmps(v):
    '''Litres per second to cubic meters per second
    '''
    return v*0.001

def seconds_to_hours(v):
    '''Convert seconds to hours
    '''
    return v*3600

#---
# Functions
#---

def create_dir(path):
    ''' Create dir if it doesn't exist '''
    if not os.path.exists(path):
        os.makedirs(path)

def tidy(flows):
    ''' Convert supply/demand data to tidy format '''
    flows['Timestep'] = [i for i in range(1,len(flows)+1)]
    id_vars = metainfo['flow_header'] + ['Timestep']
    tidy = flows.melt(id_vars=id_vars,var_name='Node',value_name='Value')
    return tidy

def map_timesteps_to_date(flows,mappable):
    ''' Function to map dates to timesteps '''
    # get time reference table
    id_vars = metainfo['flow_header'] + ['Timestep']
    time_ref = flows[id_vars].groupby(by='Timestep').max().reset_index()
    # perform merge
    mapped = pd.merge(mappable,time_ref,on='Timestep',how='right')
    return mapped

def add_time_to_edges(flows,edges):
    ''' Function to add time indices to edge data'''
    # Number of timesteps
    timesteps = flows.Timestep.max()
    # add time
    edges['Timestep'] = 1
    #repeat for each timestep
    new_edges = edges.append( [edges]*(timesteps-1) )
    #create time indices in loop
    tt = []
    for i in range(0,timesteps):
        t = edges.Timestep.to_numpy() + i
        tt.append(t)
    tt = np.concatenate(tt,axis=0)
    #add time to pandas datafram
    new_edges['Timestep'] = tt
    # reset index
    new_edges = new_edges.reset_index(drop=True)
    # add dates
    new_edges = map_timesteps_to_date(flows,new_edges)
    # reorder
    col_order = metainfo['edges_header'] + ['Timestep'] + metainfo['flow_header']
    new_edges = new_edges[col_order]
    return new_edges

def check_integrity():
    ''' Check integrity of nodes and edges data'''
    # nodes not in edge data
    # edges not in node data
    # repeating node IDs
    # check types
    # check subtypes
    # check status
    #print('> Integrity check completed with ' + int(errors) ' errors')
    errors = []

def arc_indicies_as_dict(self,var_name):
    ''' Function to convert edge indices dataframe to dict '''
    asDict = self.edge_indices[self.indices+[var_name]].set_index(keys=self.indices)[var_name].to_dict()
    return asDict

def flows_as_dict(flows):
    ''' '''
    flows_dict = flows[['Node','Timestep','Value']]
    flows_dict = flows_dict.set_index(keys=['Node','Timestep']).to_dict()['Value']
    return flows_dict

def add_super_source(nodes,edges,commodities,**kwargs):
    ''' Add super source to edges '''
    new_edges = []
    for commodity in commodities:
        tmp_edges = pd.DataFrame({'Start'       : 'super_source',
                                  'End'         : kwargs.get("sink_nodes", nodes.Name.unique()),
                                  'Commodity'   : commodity,
                                  'Cost'        : constants['super_source_maximum'],
                                  'Minimum'     : 0,
                                  'Maximum'     : constants['super_source_maximum']})
        new_edges.append(tmp_edges)
    new_edges = pd.concat(new_edges,ignore_index=True)
    return edges.append(new_edges, ignore_index=True)

def add_super_sink(nodes,edges,commodities,**kwargs):
    ''' Add super sinks to edges '''
    new_edges = []
    for commodity in commodities:
        tmp_edges = pd.DataFrame({'Start'       : kwargs.get("start_nodes", nodes.Name.unique()),
                                  'End'         : 'super_sink',
                                  'Commodity'   : commodity,
                                  'Cost'        : constants['super_source_maximum'],
                                  'Minimum'     : 0,
                                  'Maximum'     : constants['super_source_maximum']})
        new_edges.append(tmp_edges)
    new_edges = pd.concat(new_edges,ignore_index=True)
    return edges.append(new_edges, ignore_index=True)

def normalise_column(df_column):
    normalised_column = (df_column-df_column.min()) / (df_column.max()-df_column.min())
    return normalised_column

#---
# Look ups
#---

def get_nodes(nodes,index_column,lookup,index_column2=None,lookup2=None,index_column3=None,lookup3=None,index_column4=None,lookup4=None):
    ''' Get nodes of particular type and sector '''
    if index_column2 is not None and index_column3 is not None and index_column4 is not None:
        idx_nodes = nodes.loc[(nodes[index_column]==lookup) & \
                              (nodes[index_column2]==lookup2) & \
                              (nodes[index_column2]==lookup2) & \
                              (nodes[index_column4]==lookup4)]
    elif index_column2 is not None and index_column3 is not None:
        idx_nodes = nodes.loc[(nodes[index_column]==lookup) & \
                              (nodes[index_column2]==lookup2) & \
                              (nodes[index_column3]==lookup3)]
    elif index_column2 is not None:
        idx_nodes = nodes.loc[(nodes[index_column]==lookup) & \
                              (nodes[index_column2]==lookup2)]
    else:
        idx_nodes = nodes.loc[(nodes[index_column]==lookup)]
    return idx_nodes

def get_node_names(nodes,index_column,lookup,index_column2=None,lookup2=None,index_column3=None,lookup3=None,index_column4=None,lookup4=None):
    ''' Get nodes of particular type and sector '''
    if index_column2 is not None and index_column3 is not None and index_column4 is not None:
        idx_nodes = nodes.loc[(nodes[index_column]==lookup) & \
                              (nodes[index_column2]==lookup2) & \
                              (nodes[index_column2]==lookup2) & \
                              (nodes[index_column4]==lookup4)].Name.to_list()
    elif index_column2 is not None and index_column3 is not None:
        idx_nodes = nodes.loc[(nodes[index_column]==lookup) & \
                              (nodes[index_column2]==lookup2) & \
                              (nodes[index_column3]==lookup3)].Name.to_list()
    elif index_column2 is not None:
        idx_nodes = nodes.loc[(nodes[index_column]==lookup) & \
                              (nodes[index_column2]==lookup2)].Name.to_list()
    else:
        idx_nodes = nodes.loc[(nodes[index_column]==lookup)].Name.to_list()
    return idx_nodes

def get_flow_at_nodes(flows,list_of_nodes):
    ''' Get flows of specific nodes '''
    idx_flows = flows.loc[flows.Node.isin(list_of_nodes)].reset_index(drop=True)
    return idx_flows

def update_scenario(edges,connectivity_dict):
    ''' Update edges dataframe to match scenario '''
    edges.loc[(edges.Start.isin(['Jordan generation'])) \
              & (edges.End.isin(['West Bank energy demand'])),'Maximum']    = connectivity_dict['jordan_to_westbank']
    edges.loc[(edges.Start.isin(['Jordan generation'])) \
              & (edges.End.isin(['Israel energy demand'])),'Maximum']       = connectivity_dict['jordan_to_israel']
    # Israel --> West Bank, Gaza, Jordan
    edges.loc[(edges.Start.isin(['Israel generation'])) \
              & (edges.End.isin(['West Bank energy demand'])),'Maximum']    = connectivity_dict['israel_to_westbank']
    edges.loc[(edges.Start.isin(['Israel generation'])) \
              & (edges.End.isin(['Gaza energy demand'])),'Maximum']         = connectivity_dict['israel_to_gaza']
    edges.loc[(edges.Start.isin(['Israel generation'])) \
              & (edges.End.isin(['Jordan energy demand'])),'Maximum']       = connectivity_dict['israel_to_jordan']
    # Egypt --> Gaza
    edges.loc[(edges.Start.isin(['Egypt generation'])) \
              & (edges.End.isin(['Gaza energy demand'])),'Maximum']         = connectivity_dict['egypt_to_gaza']

    return edges
