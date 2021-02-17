'''
    infrasim.py
        A multi-commodity flow simulation model

    @amanmajid
'''

import gurobi as grb
import gurobipy as gp
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import time
import numpy as np

from . import submodels
from . import utils
from . import spatial
from . import plotting

from .metainfo import *
from .params import *


class Model():






    def __init__(self,nodes,edges,flows,**kwargs):

        #---
        # Read data

        # read shape files of nodes and edges
        if '.shp' in nodes and '.shp' in edges:
            nodes = gpd.read_file(nodes)
            edges = gpd.read_file(edges)
            # add topology to edge data
            edges = spatial.add_graph_topology(nodes,edges,id_attribute='Name')
            # drop geometry
            # nodes,edges = spatial.drop_geom(nodes,edges)

        # else read csv files of nodes and edges
        elif '.csv' in nodes and '.csv' in edges:
            nodes = pd.read_csv(nodes)
            edges = pd.read_csv(edges)

        # read flow data
        flows = pd.read_csv(flows)
        #flows['Hour'] = flows['Hour'] + 1

        # restrict replicates
        replicate_restriction = kwargs.get("replicate", None)
        if replicate_restriction is not None:
            flows = flows.loc[(flows.Replicate == replicate_restriction)]

        # restrict year
        years_restriction = kwargs.get("year", None)
        if years_restriction is not None:
            flows = flows.loc[(flows.Year == years_restriction)]

        # restrict timesteps
        timesteps = kwargs.get("timesteps", None)
        if timesteps is not None:
            flows = flows.loc[(flows.Timestep >= flows.Timestep.min()) & \
                              (flows.Timestep <= flows.Timestep.min() + timesteps)]

        #---
        # Add super source
        super_source = kwargs.get("super_source", True)
        if super_source==True:
            sinks = ['Three Valleys','SWOX','London','Lee_environmental_flow',
                     'Essex and Suffolk','Affinity and Surrey','Affinity Iver']
            edges = utils.add_super_source(nodes,edges,commodities=edges.Commodity.unique(),sink_nodes=sinks)
            edges.loc[edges.Start=='super_source','CAPEX'] = constants['super_source_maximum']

        #---
        # Add super sink
        super_sink = kwargs.get("super_sink", False)
        if super_sink==True:
            edges = utils.add_super_sink(nodes,edges,commodities=edges.Commodity.unique())
            edges.loc[edges.End=='super_sink','CAPEX'] = constants['super_source_maximum']

        #---
        # Tidy flow data
        flows = utils.tidy(flows)
        # filter negatives out
        flows.loc[flows.Value < 0, 'Value'] = np.nan
        flows = flows.dropna(axis=0)

        #---
        # Create infrasim cache files
        utils.create_dir(path=metainfo['infrasim_cache'])
        spatial.graph_to_csv(nodes,edges,output_dir=metainfo['infrasim_cache'])
        flows.to_csv(metainfo['infrasim_cache']+'flows.csv')

        #---
        # add time indices to edge data
        self.edge_indices   = utils.add_time_to_edges(flows,edges)

        #---
        # Define sets
        self.flows          = flows
        self.edges          = edges
        self.nodes          = nodes
        self.indices        = metainfo['edge_index_variables']
        self.constants      = constants
        self.commodities    = self.edges.Commodity.unique().tolist()
        self.node_types     = self.nodes.Type.unique().tolist()
        self.technologies   = self.nodes.Subtype.unique().tolist()
        self.functions      = self.nodes.Function.unique().tolist()
        self.timesteps      = self.flows.Timestep.unique().tolist()
        self.time_ref       = self.flows[['Timestep','Month','Year']]
        self.days           = self.flows.Day.unique().tolist()
        self.months         = self.flows.Month.unique().tolist()
        self.years          = self.flows.Year.unique().tolist()

        #---
        # define gurobi model
        model_name          = 'infrasim'
        self.model          = gp.Model(model_name)

        #---
        # define model temporal resolution
        self.temporal_resolution = metainfo['temporal_resolution']






    def build(self):
        '''

        Contents:
        -------

            1. Variables
            2. Objective Function
            3. Generic Constraints


        Returns
        -------
        None.


        '''

        # start timing
        start_time = time.clock()

        #======================================================================
        # VARIABLES
        #======================================================================

        #---
        # arcflows
        arc_indicies      = self.edge_indices[self.indices].set_index(keys=self.indices).index.to_list()
        self.arcFlows     = self.model.addVars(arc_indicies,name="arcflow")

        #---
        # storage volumes
        storage_nodes         = utils.get_nodes(nodes=self.nodes,index_column='Type',lookup='storage')

        storage_indices       = [(n, storage_nodes.loc[storage_nodes.Name==n,'Commodity'].values[0], t)
                                 for n in storage_nodes.Name
                                 for t in self.timesteps]

        self.storage_volume   = self.model.addVars(storage_indices,lb=0,name="storage_volume")
        
        #---
        # capacity of new options
        option_nodes         = utils.get_nodes(nodes=self.nodes,index_column='Status',lookup='option')

        option_indices       = [(n, option_nodes.loc[option_nodes.Name==n,'Commodity'].values[0])
                                 for n in option_nodes.Name]

        self.option_indices  = self.model.addVars(option_indices,lb=0,name="option_capacities")


        #======================================================================
        # OBJECTIVE FUNCTION
        #======================================================================

        #---
        # Minimise cost of flow + capex

        # create cost dict
        self.costDict   = utils.arc_indicies_as_dict(self,metainfo['cost_column'])
        # create capex dict
        self.capexDict  = self.nodes[['Name','Commodity','CAPEX']].set_index(['Name','Commodity'])['CAPEX'].to_dict()
        # set objective
        self.model.setObjectiveN(
            gp.quicksum( \
                self.arcFlows[i,j,k,t] * self.costDict[i,j,k,t] \
                    for i,j,k,t in self.arcFlows),0,weight=1)

        #---
        # Maximise storage
        self.model.setObjectiveN( \
            gp.quicksum( \
                self.storage_volume[n,k,t] \
                    for n,k,t in self.storage_volume),1,weight=-1)

        #======================================================================
        # CONSTRAINTS
        #======================================================================

        #---
        # SUPER NODES
        #---

        if 'super_source' in self.edges.Start.unique():
            # constrain
            self.model.addConstrs(( \
                self.arcFlows.sum('super_source','*',k,t)  \
                    <= constants['super_source_maximum'] \
                        for t in self.timesteps \
                            for k in self.commodities),'super_source_supply')

        if 'super_sink' in self.edges.End.unique():
            # constrain
            self.model.addConstrs(( \
                self.arcFlows.sum('*','super_sink',k,t)  >= 0 \
                    for t in self.timesteps \
                        for k in self.commodities),'super_sink_demand')

        #----------------------------------------------------------------------
        # ARC FLOW BOUNDS
        #----------------------------------------------------------------------

        # Flows must be below upper bounds
        upper_bound = utils.arc_indicies_as_dict(self,metainfo['upper_bound'])
        self.model.addConstrs(( \
            self.arcFlows[i,j,k,t] <= upper_bound[i,j,k,t] \
                for i,j,k,t in self.arcFlows),'upper_bound')

        # Flows must be above lower bounds
        lower_bound = utils.arc_indicies_as_dict(self,metainfo['lower_bound'])
        self.model.addConstrs(( \
            lower_bound[i,j,k,t] <= self.arcFlows[i,j,k,t] \
                for i,j,k,t in self.arcFlows),'lower_bound')

        #----------------------------------------------------------------------
        # WATER SUPPLY
        #----------------------------------------------------------------------

        #---
        # SUPPLY FROM SOURCES
        #---
        if 'water' in self.commodities:
            # get water supply nodes
            water_nodes = utils.get_node_names(nodes=self.nodes,
                                               index_column='Type',lookup='source',
                                               index_column2='Nodal_Flow',lookup2='True',
                                               index_column3='Commodity',lookup3='water')

            # get flow at water supply nodes
            water_flows  = utils.get_flow_at_nodes(flows=self.flows,list_of_nodes=water_nodes)
            supply_dict  = utils.flows_as_dict(flows=water_flows)

            # constrain
            self.model.addConstrs((
                self.arcFlows.sum(i,'*','water',t)  == supply_dict[i,t] \
                    for t in self.timesteps \
                        for i in water_nodes),'water_supply')

        #---
        # DEMAND AT SINKS
        #---
        if 'water' in self.commodities:
            # get water demand nodes
            water_nodes = utils.get_node_names(nodes=self.nodes,
                                               index_column='Type',lookup='sink',
                                               index_column2='Nodal_Flow',lookup2='True',
                                               index_column3='Commodity',lookup3='water')

            # get flow at water supply nodes
            water_flows  = utils.get_flow_at_nodes(flows=self.flows,list_of_nodes=water_nodes)
            demand_dict  = utils.flows_as_dict(flows=water_flows)

            # constrain
            self.model.addConstrs((
                self.arcFlows.sum('*',j,'water',t)  == demand_dict[j,t] \
                    for t in self.timesteps \
                        for j in water_nodes),'water_demand')

        #----
        # FLOWS AROUND RIVER LEE
        #----

        #-
        # Sum of Lee abstraction
        for y in self.years:
            ref_timesteps = self.flows.loc[self.flows.Year==y,'Timestep'].to_list()
            self.model.addConstr( \
                gp.quicksum(self.arcFlows[i,j,k,t] \
                    for i in ['Lee abstraction'] \
                        for j in ['London Storages'] \
                            for k in ['water'] \
                                for t in ref_timesteps) \
                                    <= 200937,'SurrAnnual')

        #-
        # Total pumping from River Lee
        self.model.addConstrs((\
            self.arcFlows[i,j,k,t] <= constants['river_lee_pumping'] \
                for i in ['Lee abstraction'] \
                    for j in ['London Storages'] \
                        for k in ['water'] \
                            for t in self.timesteps),'LeeMax')

        #-
        # River Lee Environmental Flow constraint
        self.model.addConstrs((
            self.arcFlows.sum('*',j,k,t) == constants['river_lee_env_constraint'] \
                for k in ['water'] \
                    for t in self.timesteps \
                        for j in ['Lee environmental flow']),'demand')

        #-
        # London ground water license
        for y in self.years:
            ref_timesteps = self.flows.loc[self.flows.Year==y,'Timestep'].to_list()
            self.model.addConstr( \
                gp.quicksum(self.arcFlows['Aggregated London GW','London','water',t] \
                    for t in ref_timesteps) \
                        <= constants['annual_london_groundwater'],'GWCap')

        self.model.addConstrs(( \
            self.arcFlows.sum('Aggregated London GW','*','water',t) <= constants['london_gw_pumping'] \
                for t in self.timesteps),'LeeSS')

        #----
        # TEDDINGTON ENVIRONMENTAL FLOWS
        #----

        #-
        # Minimum flow to Teddington
        self.model.addConstrs((
            self.arcFlows.sum('*',j,k,t) >= constants['min_flow_teddington']
                for j in ['Teddington Environmental Flow']
                    for k in ['water']
                        for t in self.timesteps),'demand')

        #-
        # Min flow to Teddington
        self.model.addConstrs((
            self.arcFlows[i,j,k,t] >= 500 \
                for i in ['J3']
                    for j in ['Teddington Environmental Flow']
                        for k in ['water']
                            for t in self.timesteps),'demand')

        #----------------------------------------------------------------------
        # STORAGES
        #----------------------------------------------------------------------

        if 'storage' in self.node_types:

            #---
            # Storage node volume must be below capacity
            storage_nodes = utils.get_nodes(nodes=self.nodes,index_column='Type',lookup='storage')
            storage_caps  = storage_nodes.set_index(keys=['Name','Commodity']).to_dict()['Capacity']

            self.model.addConstrs((
                self.storage_volume.sum(n,k,t) \
                    <= storage_caps[n,k] \
                        for n,k,t in self.storage_volume \
                            if (n,k) in storage_caps),'Capacity')

            #---
            # Storage node balance
            storage_nodes = utils.get_nodes(nodes=self.nodes,index_column='Type',lookup='storage')
            storage_caps  = storage_nodes.set_index(keys=['Name','Commodity']).to_dict()['Capacity']
            storage_nodes = utils.get_node_names(nodes=self.nodes,index_column='Type',lookup='storage')

            # t=1
            self.model.addConstrs((
                self.storage_volume.sum(j,k,t) == \
                    storage_caps[j,k] \
                        + self.arcFlows.sum('*',j,k,t) \
                            - self.arcFlows.sum(j,'*',k,t) \
                                for k in self.commodities
                                    for t in self.timesteps if t==1
                                        for j in storage_nodes
                                            if (j,k) in storage_caps),'storage_init')
            # t>1
            self.model.addConstrs((
                self.storage_volume.sum(j,k,t) == \
                    self.storage_volume.sum(j,k,t-1) \
                        + self.arcFlows.sum('*',j,k,t) \
                            - self.arcFlows.sum(j,'*',k,t) \
                                for k in self.commodities
                                    for t in self.timesteps if t>1
                                        for j in storage_nodes
                                            if (j,k) in storage_caps),'storage_balance')

            #----
            # FLOWS AROUND LONDON STORAGES
            #----

            #-
            # Sum of flow into London Storages from Upper Thames
            self.model.addConstrs((
                self.arcFlows[i,j,k,t] <= 3500 \
                    for i in ['J3'] \
                        for j in ['London Storages'] \
                            for k in ['water'] \
                                for t in self.timesteps),'LonStorCap')

            #-
            # SWOX Discharge
            self.model.addConstrs((
                self.arcFlows['SWOX Discharge','J1','water',t] \
                    <= 0.95 * (self.arcFlows['SWOX GW','SWOX','water',t] \
                        + self.arcFlows['Farmoor','SWOX','water',t])\
                            for t in self.timesteps),'swoxDischarge')

            self.model.addConstrs((
                self.arcFlows['SWOX Discharge','J1','water',t] \
                    <= 90 for t in self.timesteps),'swoxDischarge')

            #-
            # WB Groundwater
            self.model.addConstrs((
                self.arcFlows.sum(i,'*',k,t)  == 0 \
                    for i in ['WB GW'] \
                        for k in ['water'] \
                            for t in self.timesteps),'demand')



        #----------------------------------------------------------------------
        # JUNCTIONS
        #----------------------------------------------------------------------

        #---
        # Junction node balance
        if 'junction' in self.node_types:
            junction_nodes = utils.get_node_names(nodes=self.nodes,index_column='Type',lookup='junction')

            for k in self.commodities:
                self.model.addConstrs((
                    self.arcFlows.sum('*',j,k,t) \
                        == self.arcFlows.sum(j,'*',k,t) \
                            for t in self.timesteps \
                                for j in junction_nodes),'junction_balance')


        #----------------------------------------------------------------------
        # WATER SUPPLY TO WASTEWATER
        #----------------------------------------------------------------------

        if 'wastewater' in self.commodities:

            #---
            # WATER DEMANDS -> WWTPs
            water_nodes = utils.get_node_names(nodes=self.nodes,
                                               index_column='Type',lookup='sink',
                                               index_column2='Nodal_Flow',lookup2='True',
                                               index_column3='Commodity',lookup3='water')

            self.model.addConstrs(( \
                constants['water_to_wastewater_capture'] \
                    *self.arcFlows.sum('*',j,'water',t) \
                        == self.arcFlows.sum(j,'*','wastewater',t) \
                            for j in water_nodes \
                                for t in self.timesteps),'demand')

            #---
            # WWTP inflow should be less than capacity
            wwtp_nodes = utils.get_nodes(nodes=self.nodes,
                                         index_column='Subtype',lookup='wwtp')

            self.model.addConstrs(( \
                self.arcFlows.sum('*',j,'wastewater',t) \
                    <= wwtp_nodes.loc[wwtp_nodes.Name==j,'Capacity'] \
                        for j in wwtp_nodes.Name \
                            for t in self.timesteps),'demand')

        #----------------------------------------------------------------------
        # ENERGY SUPPLY
        #----------------------------------------------------------------------

        if 'electricity' in self.commodities:

            #---
            # SUPPLY FROM SOURCES
            #---

            #---
            # solar
            if 'solar' in self.technologies:
                # get solar nodes
                solar_nodes = utils.get_nodes(nodes=self.nodes,
                                        index_column='Subtype',lookup='solar')
                # get irradiance data
                solar_irradiance  = utils.get_flow_at_nodes(flows=self.flows,list_of_nodes=['rsds'])
                solar_irradiance  = utils.flows_as_dict(flows=solar_irradiance)
                # get monthly timestep ref
                month_ref = self.time_ref.set_index('Timestep')['Month'].to_dict()
                # constrain
                self.model.addConstrs((
                    self.arcFlows.sum(i,'*','electricity',t)  \
                        <= submodels.solar_pv(area=solar_nodes.loc[solar_nodes.Name==i,'Capacity'],
                            month=month_ref[t],efficiency=constants['solar_efficiency'],\
                                irradiance=solar_irradiance['rsds',t]) \
                                    for t in self.timesteps \
                                        for i in solar_nodes.Name),'solar_supply')

            #---
            # wind
            if 'wind' in self.technologies:
                # get wind nodes
                wind_nodes = utils.get_nodes(nodes=self.nodes,
                                        index_column='Subtype',lookup='wind')
                # get irradiance data
                wind_speed  = utils.get_flow_at_nodes(flows=self.flows,list_of_nodes=['wss'])
                wind_speed  = utils.flows_as_dict(flows=wind_speed)
                # constrain
                self.model.addConstrs((
                    self.arcFlows.sum(i,'*','electricity',t)  \
                        <= constants['wind_efficiency'] * submodels.wind_vestas2(wind_speed['wss',t])
                            for t in self.timesteps
                                for i in wind_nodes.Name),'wind_supply')

            #---
            # biogas
            if 'biogas' in self.technologies:
                # get biogas nodes
                biogas_nodes = utils.get_nodes(nodes=self.nodes,
                                        index_column='Subtype',lookup='biogas')
                # constrain
                self.model.addConstrs((
                    self.arcFlows.sum(i,'*','electricity',t)  \
                        <= biogas_nodes.loc[biogas_nodes.Name==i,'Capacity'] * 1000 * 24
                            for t in self.timesteps
                                for i in biogas_nodes.Name),'biogas_supply')

        #----------------------------------------------------------------------
        # ENERGY FOR WATER SUPPLY
        #----------------------------------------------------------------------

        #---
        # WATER SUPPLY AGGREGATED
        #---

        demand_nodes = utils.get_nodes(nodes=self.nodes,
                                       index_column='Subtype',lookup='demand')

        self.model.addConstrs((
            self.arcFlows.sum('*',j,'electricity',t)  >= \
                #----
                # WATER SUPPLY
                # water treatment + distribution
                (constants['water_treatment_ei']+constants['water_network_coverage']*constants['surface_water_pumping_ei']) \
                    * self.arcFlows.sum('*',j,'water',t) \
                #----
                # WATER END USE
                + variables['pctg_demand_personal_washing']*constants['natural_gas_water_heating']*self.arcFlows.sum('*',j,'water',t) \
                    + variables['pctg_demand_clothes_washing']*constants['clothes_washing_ei']*self.arcFlows.sum('*',j,'water',t) \
                    + variables['pctg_demand_wc_flushing']*constants['auxillary_water_use_ei']*self.arcFlows.sum('*',j,'water',t) \
                    + variables['pctg_demand_external']*constants['auxillary_water_use_ei']*self.arcFlows.sum('*',j,'water',t) \
                    + variables['pctg_demand_dish_washing']*constants['dish_washing_ei']*self.arcFlows.sum('*',j,'water',t) \
                        for j in demand_nodes.Name \
                            for t in self.timesteps),'ws_energy')

        #---
        # GROUNDWATER
        #---

        if 'groundwater' in self.technologies:

            gw_nodes = utils.get_nodes(nodes=self.nodes,
                                       index_column='Subtype',lookup='groundwater')

            self.model.addConstrs((
                self.arcFlows.sum('*',j,'electricity',t)  == \
                    constants['groundwater_pumping_ei']\
                        *constants['groundwater_pumping_height']\
                            *self.arcFlows.sum(j,'*','water',t)\
                                for j in gw_nodes.Name \
                                    for t in self.timesteps),'gw_energy')

        #---
        # DESALINATION
        #---

        if 'desalination' in self.technologies:

            desal_nodes = utils.get_nodes(nodes=self.nodes,
                                          index_column='Subtype',lookup='groundwater')

            self.model.addConstrs((
                self.arcFlows.sum('*',j,'electricity',t)  == \
                    constants['desalination_brackish_ei']\
                        *self.arcFlows.sum(j,'*','water',t)\
                            for j in desal_nodes.Name \
                                for t in self.timesteps),'desal_energy')

        #----------------------------------------------------------------------
        # ENERGY FOR WASTEWATER
        #----------------------------------------------------------------------

        #---
        # WWTP AGGREGATED
        #---

        if 'wastewater' in self.commodities:

            wwtp_nodes = utils.get_nodes(nodes=self.nodes,
                                         index_column='Subtype',lookup='wwtp')

            self.model.addConstrs((
                self.arcFlows.sum('*',j,'electricity',t)  == \
                    # wastewater treatment
                    (constants['wastewater_treatment_ei']\
                        # wastewater network
                        + constants['wastewater_network_coverage']\
                            *constants['wastewater_pumping_ei']) \
                                * self.arcFlows.sum('*',j,'wastewater',t) \
                                    for j in wwtp_nodes.Name \
                                        for t in self.timesteps),'ww_energy')
        
        self.build_time = time.clock() - start_time

        #print(self.build_time, "seconds")
        #print('------------- MODEL BUILD COMPLETE -------------')






    def run(self,pprint=True,write=True):
        ''' Function to solve GurobiPy model'''
        # write model to LP
        if write==True:
            self.model.write(metainfo['infrasim_cache']+self.model.ModelName+'.lp')
        # set output flag
        if pprint==True:
            self.model.setParam('OutputFlag', 1)
        else:
            self.model.setParam('OutputFlag', 0)
        # optimise
        self.model.optimize()

        # WRITE RESULTS
        utils.create_dir(path=metainfo['outputs_data'])

        if self.model.Status == 2:
            # arcFlows
            arcFlows            = self.model.getAttr('x', self.arcFlows)
            keys                = pd.DataFrame(arcFlows.keys(),columns=['Start','End','Commodity','Timestep'])
            vals                = pd.DataFrame(arcFlows.items(),columns=['key','Value'])
            results_arcflows    = pd.concat([keys,vals],axis=1)
            results_arcflows    = self.flows[['Day','Month','Year','Timestep']].merge(results_arcflows, on='Timestep')
            results_arcflows    = results_arcflows[['Start','End','Commodity','Day','Month','Year','Timestep','Value']]
            results_arcflows    = results_arcflows.drop_duplicates()
            # write csv
            results_arcflows.to_csv(metainfo['outputs_data']+'results_arcflows.csv',index=False)
            self.results_arcflows = results_arcflows

            # storageVolumes
            storage_volumes              = self.model.getAttr('x', self.storage_volume)
            keys                         = pd.DataFrame(storage_volumes.keys(),columns=['Node','Commodity','Timestep'])
            vals                         = pd.DataFrame(storage_volumes.items(),columns=['key','Value'])
            results_storage_volumes      = pd.concat([keys,vals],axis=1)
            results_storage_volumes      = self.flows[['Day','Month','Year','Timestep']].merge(results_storage_volumes, on='Timestep')
            results_storage_volumes      = results_storage_volumes[['Node','Commodity','Day','Month','Year','Timestep','Value']]
            results_storage_volumes      = results_storage_volumes.drop_duplicates()
            self.results_storage_volumes = results_storage_volumes
            # write csv
            results_storage_volumes.to_csv(metainfo['outputs_data']+'results_storage_volumes.csv',index=False)

            # optionIndicies
            option_indices              = self.model.getAttr('x', self.option_indices)
            keys                        = pd.DataFrame(option_indices.keys(),columns=['Option','Commodity'])
            vals                        = pd.DataFrame(option_indices.items(),columns=['key','Value'])
            results_option_indices      = pd.concat([keys,vals],axis=1)
            results_option_indices      = results_option_indices[['Option','Commodity','Value']]
            self.results_option_indices = results_option_indices
            # write csv
            results_option_indices.to_csv(metainfo['outputs_data']+'results_option_indices.csv',index=False)






    def debug(self,output_path=''):
        '''
        Compute model Irreducible Inconsistent Subsystem (IIS) to help deal with infeasibilies
        '''
        self.model.computeIIS()
        self.model.write(metainfo['infrasim_cache']+'model-debug-report.ilp')






    def postprocess(self):
        ''' Post processing of results '''

        utils.create_dir(path=metainfo['outputs_figures'])

        #----------------------------------------------------------------------
        # GRAPH ANALYSIS
        #----------------------------------------------------------------------

        #---
        # Arc utilisation
        for k in self.commodities:
            plt.figure(figsize=(10,8))
            w = self.results_arcflows.loc[self.results_arcflows.Commodity==k].reset_index(drop=True)
            w = w.groupby(by=['Start','End']).sum().reset_index()
            w.Value = w.Value / w.Value.sum()

            if k=='water':
                edge_color='blue'
            elif k=='electricity':
                edge_color='red'
            else:
                edge_color='black'

            G = plotting.results_to_graph(w,edge_color=edge_color)
            plt.savefig(metainfo['outputs_figures']+'arc_utilisation_'+k+'.pdf')
