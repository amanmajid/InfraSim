'''
    infrasim.py
        A multi-commodity flow simulation model

    @amanmajid
'''

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


class infrasim():






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
        # Get scenario
        self.scenario       = kwargs.get("scenario", metainfo['scenario'])
        self.connectivity   = connectivity
        if self.scenario == 'baseline':
            edges = utils.update_scenario(edges,connectivity)
        elif self.scenario == 'cooperation':
            # update connectivity
            self.connectivity['jordan_to_israel'] = 99999
            self.connectivity['israel_to_jordan'] = 99999
            edges = utils.update_scenario(edges,self.connectivity)
        elif self.scenario == 'no_cooperation':
            # update connectivity
            #self.connectivity['jordan_to_westbank']     = 0
            self.connectivity['israel_to_jordan']       = 0
            #self.connectivity['israel_to_westbank']     = 0
            #self.connectivity['israel_to_gaza']         = 0
            #self.connectivity['westbank_to_israel']     = 0
            #self.connectivity['egypt_to_gaza']          = 0
            edges = utils.update_scenario(edges,self.connectivity)

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
        self.time_ref       = self.flows[['Timestep','Year']]
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
            # COST OF FLOW
            gp.quicksum(
                self.arcFlows[i,j,k,t] * self.costDict[i,j,k,t]
                for i,j,k,t in self.arcFlows),0,weight=1)

        #---
        # Maximise storage
        self.model.setObjectiveN( gp.quicksum(self.storage_volume[n,k,t]
                                              for n,k,t in self.storage_volume),1,weight=-0.5)

        #======================================================================
        # CONSTRAINTS
        #======================================================================

        #----------------------------------------------------------------------
        # SUPER NODES
        #----------------------------------------------------------------------

        if 'super_source' in self.edges.Start.unique():
            # constrain
            self.model.addConstrs((
                            self.arcFlows.sum('super_source','*',k,t)  <= constants['super_source_maximum']
                            for t in self.timesteps
                            for k in self.commodities),'super_source_supply')

        if 'super_sink' in self.edges.End.unique():
            # constrain
            self.model.addConstrs((
                            self.arcFlows.sum('*','super_sink',k,t)  >= 0
                            for t in self.timesteps
                            for k in self.commodities),'super_sink_demand')

        #----------------------------------------------------------------------
        # ARC FLOW BOUNDS
        #----------------------------------------------------------------------

        # Flows must be below upper bounds
        upper_bound = utils.arc_indicies_as_dict(self,metainfo['upper_bound'])
        self.model.addConstrs((self.arcFlows[i,j,k,t] <= upper_bound[i,j,k,t]
                               for i,j,k,t in self.arcFlows),'upper_bound')

        # Flows must be above lower bounds
        lower_bound = utils.arc_indicies_as_dict(self,metainfo['lower_bound'])
        self.model.addConstrs((lower_bound[i,j,k,t] <= self.arcFlows[i,j,k,t]
                               for i,j,k,t in self.arcFlows),'lower_bound')

        #----------------------------------------------------------------------
        # WATER SUPPLY
        #----------------------------------------------------------------------

        #---
        # Supply from water source nodes
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
                            self.arcFlows.sum(i,'*','water',t)  == supply_dict[i,t]
                            for t in self.timesteps
                            for i in water_nodes),'water_supply')

        #---
        # Demand at sink nodes
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
                            self.arcFlows.sum('*',j,'water',t)  == demand_dict[j,t]
                            for t in self.timesteps
                            for j in water_nodes),'water_demand')

        # #----------------------------------------------------------------------
        # # WATER TO WASTEWATER
        # #----------------------------------------------------------------------
        #
        # if 'wastewater' in self.commodities:
        #
        #     #---
        #     # Water to wastewater rule
        #     demand_nodes = utils.get_node_names(self.nodes,index_column='Subtype',lookup='demand')
        #
        #     self.model.addConstrs((
        #              self.constants['water_to_wastewater_capture'] \
        #                  *self.arcFlows.sum('*',i,'water',t)==self.arcFlows.sum(i,'*','wastewater',t)
        #                     for t in self.timesteps
        #                     for i in demand_nodes),'water_to_wastewater')
        #
        #     #---
        #     # Flow into WWTPs must be less than capacity
        #     wastewater_nodes      = utils.get_nodes(self.nodes,index_column='Subtype',lookup='wwtp')
        #     wastewater_capacities = wastewater_nodes.set_index(keys=['Name']).to_dict()['Capacity']
        #
        #     self.model.addConstrs((
        #               self.arcFlows.sum('*',j,'wastewater',t) <= wastewater_capacities[j]
        #               for t in self.timesteps
        #               for j in wastewater_nodes.Name),'wastewater_capacity')
        #
        #     #---
        #     # Flow into WWTPs must be more than minimum flow fraction
        #     self.model.addConstrs((
        #               self.arcFlows.sum('*',j,'wastewater',t) \
        #                   >= constants['wastewater_plant_minimum_flow'] * wastewater_capacities[j]
        #               for t in self.timesteps
        #               for j in wastewater_nodes.Name),'wastewater_plant_minimum_flow')

        # #----------------------------------------------------------------------
        # # ENERGY SUPPLY
        # #----------------------------------------------------------------------
        #
        # #---
        # # Energy demand
        # if 'electricity' in self.commodities:
        #     # get water demand nodes
        #     energy_nodes = utils.get_node_names(nodes=self.nodes,
        #                                         index_column='Type',lookup='sink',
        #                                         index_column2='Nodal_Flow',lookup2='True',
        #                                         index_column3='Commodity',lookup3='electricity')
        #
        #     # get flow at water supply nodes
        #     energy_flows  = utils.get_flow_at_nodes(flows=self.flows,list_of_nodes=energy_nodes)
        #     demand_dict   = utils.flows_as_dict(flows=energy_flows)
        #
        #     # constrain
        #     self.model.addConstrs((
        #                     self.arcFlows.sum('*',j,'electricity',t)  == demand_dict[j,t]
        #                     for t in self.timesteps
        #                     for j in energy_nodes),'energy_demand')
        #
        # #---
        # # Supply from electricity source nodes
        # if 'electricity' in self.commodities:
        #
        #     #---
        #     # get energy supply nodes
        #     elec_nodes = utils.get_node_names(nodes=self.nodes,
        #                                       index_column='Type',lookup='source',
        #                                       index_column2='Nodal_Flow',lookup2='True',
        #                                       index_column3='Commodity',lookup3='electricity',
        #                                       index_column4='Status',lookup4='online')
        #
        #     # get flow at energy supply nodes
        #     elec_flows   = utils.get_flow_at_nodes(flows=self.flows,list_of_nodes=elec_nodes)
        #     supply_dict  = utils.flows_as_dict(flows=elec_flows)
        #
        #     # constrain
        #     self.model.addConstrs((
        #                     self.arcFlows.sum(i,'*','electricity',t)  <= supply_dict[i,t]
        #                     for t in self.timesteps
        #                     for i in elec_nodes),'electricity_supply')
        #
        #     #---
        #     # Baseload supplies
        #     def baseload_supply(technology,ramping_rate):
        #         if technology in self.technologies:
        #             # index nodes
        #             baseload_nodes = utils.get_nodes(self.nodes,index_column='Subtype',lookup=technology)
        #
        #             # constrain supply
        #             self.model.addConstrs((
        #                         self.arcFlows.sum(i,'*','electricity',t)  <= \
        #                             baseload_nodes.loc[baseload_nodes.Name==i,'Capacity'].values[0]
        #                         for t in self.timesteps
        #                         for i in baseload_nodes.Name),technology+'_baseload')
        #
        #             # constrain ramping rate
        #             if self.temporal_resolution == 'hourly':
        #                  self.model.addConstrs((
        #                         self.arcFlows.sum(i,'*','electricity',t) -
        #                             self.arcFlows.sum(i,'*','electricity',t-1) <= \
        #                                 ramping_rate
        #                                 for t in self.timesteps if t>1
        #                                 for i in baseload_nodes.Name),technology+'_supply')
        #
        #     # open-cycle gas turbine (OCGT) generation
        #     baseload_supply(technology='ocgt',ramping_rate=constants['ocgt_ramping_rate'])
        #     # closed-cycle gas turbine (ccgt) generation
        #     baseload_supply(technology='ccgt',ramping_rate=constants['ccgt_ramping_rate'])
        #     # Coal generation
        #     baseload_supply(technology='coal',ramping_rate=constants['coal_ramping_rate'])
        #     # Diesel generation
        #     baseload_supply(technology='diesel',ramping_rate=constants['diesel_ramping_rate'])
        #     # Bio-gas generation
        #     baseload_supply(technology='biogas',ramping_rate=constants['ccgt_ramping_rate'])
        #     # Natural gas generation
        #     baseload_supply(technology='natural gas',ramping_rate=constants['nat_gas_ramping_rate'])

        # #----------------------------------------------------------------------
        # # ENERGY FOR WATER SUPPLY
        # #----------------------------------------------------------------------
        #
        # if 'electricity' in self.commodities:
        #
        #     def energy_for_water(technology,ei_factor):
        #         if technology in self.technologies:
        #             # get nodes
        #             technology_nodes = utils.get_node_names(self.nodes,index_column='Subtype',lookup=technology)
        #             # create constraint
        #             self.model.addConstrs((
        #                 ei_factor * self.arcFlows.sum(i,'*','water',t) \
        #                     == self.arcFlows.sum('*',i,'electricity',t)
        #                 for i in technology_nodes
        #                 for t in self.timesteps),'energy_for_'+technology[0])

            # #---
            # # Electricity for groundwater
            # technology  = 'groundwater'
            # ei_factor   = constants['groundwater_pumping_ei'] * constants['groundwater_pumping_height']
            # energy_for_water(technology=technology,ei_factor=ei_factor)
            #
            # #---
            # # Electricity for water pumping
            # technology  = 'surface water'
            # ei_factor   = constants['surface_water_pumping_ei'] * constants['water_network_coverage']
            # energy_for_water(technology=technology,ei_factor=ei_factor)
            #
            # #---
            # # Electricity for water treatment
            # technology  = 'water treatment'
            # ei_factor   = constants['water_treatment_ei']
            # energy_for_water(technology=technology,ei_factor=ei_factor)
            #
            # #---
            # # Electricity for brackish desalination
            # technology  = 'brackish desalination'
            # ei_factor   = constants['desalination_brackish_ei']
            # energy_for_water(technology=technology,ei_factor=ei_factor)
            #
            # #---
            # # Electricity for ocean desalination
            # technology  = 'ocean desalination'
            # ei_factor   = constants['desalination_ocean_ei']
            # energy_for_water(technology=technology,ei_factor=ei_factor)

        # #----------------------------------------------------------------------
        # # ENERGY FOR WATER END-USE
        # #----------------------------------------------------------------------
        #
        # if 'electricity' in self.commodities:
        #
        #     #---
        #     # Electricity for household water end-use
        #     if 'household' in self.functions:
        #         consumer_nodes = utils.get_node_names(self.nodes,index_column='Function',lookup='household')
        #         # get ei_factor
        #         ei_factor = submodels.household_water_demand()
        #         # create constraint
        #         self.model.addConstrs((
        #                 ei_factor * self.arcFlows.sum('*',i,'water',t) \
        #                     == self.arcFlows.sum('*',i,'electricity',t)
        #                 for i in consumer_nodes
        #                 for t in self.timesteps),'energy_for_household')



        # #----------------------------------------------------------------------
        # # ENERGY FOR WASTEWATER
        # #----------------------------------------------------------------------
        #
        # if 'electricity' in self.commodities and 'wastewater' in self.commodities:
        #
        #     def energy_for_wastewater(technology,ei_factor):
        #         if technology in self.technologies:
        #             # get nodes
        #             technology_nodes = utils.get_node_names(self.nodes,index_column='Subtype',lookup=technology)
        #             # create constraint
        #             self.model.addConstrs((
        #                 ei_factor * self.arcFlows.sum('*',i,'wastewater',t) \
        #                     == self.arcFlows.sum('*',i,'electricity',t)
        #                 for i in technology_nodes
        #                 for t in self.timesteps),'energy_for_'+technology[0])
        #
        #     #---
        #     # Electricity for wastewater treatment
        #     technology  = 'wwtp'
        #     ei_factor   = constants['wastewater_treatment_ei']
        #     energy_for_wastewater(technology=technology,ei_factor=ei_factor)
        #
        #     #---
        #     # Electricity for wastewater pumping
        #     technology  = 'wwtp'
        #     ei_factor   = constants['wastewater_pumping_ei'] * constants['wastewater_network_coverage']
        #     energy_for_wastewater(technology=technology,ei_factor=ei_factor)


        #----------------------------------------------------------------------
        # STORAGES
        #----------------------------------------------------------------------

        if 'storage' in self.node_types:

            #---
            # Storage node volume must be below capacity
            storage_nodes = utils.get_nodes(nodes=self.nodes,index_column='Type',lookup='storage')
            storage_caps  = storage_nodes.set_index(keys=['Name','Commodity']).to_dict()['Capacity']

            self.model.addConstrs((
                            self.storage_volume.sum(n,k,t) <= storage_caps[n,k]
                            for n,k,t in self.storage_volume
                            if (n,k) in storage_caps),'Capacity')

            #---
            # Storage node balance
            storage_nodes = utils.get_nodes(nodes=self.nodes,index_column='Type',lookup='storage')
            storage_caps  = storage_nodes.set_index(keys=['Name','Commodity']).to_dict()['Capacity']
            storage_nodes = utils.get_node_names(nodes=self.nodes,index_column='Type',lookup='storage')

            # t=1
            self.model.addConstrs((
                         self.storage_volume.sum(j,k,t) == \
                         storage_caps[j,k] + self.arcFlows.sum('*',j,k,t) - self.arcFlows.sum(j,'*',k,t) \
                         for k in self.commodities
                         for t in self.timesteps if t==1
                         for j in storage_nodes
                         if (j,k) in storage_caps),'storage_init')
            # t>1
            self.model.addConstrs((
                         self.storage_volume.sum(j,k,t) == \
                         self.storage_volume.sum(j,k,t-1) + self.arcFlows.sum('*',j,k,t) - self.arcFlows.sum(j,'*',k,t) \
                         for k in self.commodities
                         for t in self.timesteps if t>1
                         for j in storage_nodes
                         if (j,k) in storage_caps),'storage_balance')


        #----------------------------------------------------------------------
        # JUNCTIONS
        #----------------------------------------------------------------------

        #---
        # Junction node balance
        if 'junction' in self.node_types:
            junction_nodes = utils.get_node_names(nodes=self.nodes,index_column='Type',lookup='junction')

            for k in self.commodities:
                self.model.addConstrs((
                         self.arcFlows.sum('*',j,k,t)  == self.arcFlows.sum(j,'*',k,t)
                                for t in self.timesteps
                                for j in junction_nodes),'junction_balance')
            

        #----------------------------------------------------------------------
        # THAMES SPECIFIC CONSTRAINTS
        #----------------------------------------------------------------------

        #-
        # London Storage Pumping: Refill or Draw down



        # for j in ['Brent','Queen Mother','Island Barn','Hilfield Park','Walthamstow No.4']:
        #     for t in self.timesteps:
        #         if t>1:
        #             storage_pctg_before = (self.storage_volume.sum(j,'water',t-1) / storage_caps[j,'water'])*100
        #             storage_pctg_after  = (self.storage_volume.sum(j,'water',t) / storage_caps[j,'water'])*100
        #             if storage_pctg_before <= storage_pctg_after:

        #----
        # FLOWS TO AFFINITY
        #----

        # #-
        # # Sum of flow to Affinity Iver
        # for y in self.years:
        #     ref_timesteps = self.flows.loc[self.flows.Year==y,'Timestep'].to_list()
        #     self.model.addConstr(
        #                 gp.quicksum(self.arcFlows['J3','Affinity Iver','water',t]
        #                             for t in ref_timesteps) \
        #                 <= constants['annual_flow_to_affinity_iver'],'IverAnnual')
        #
        # #-
        # # Sum of flow to Affinity and Surrey
        # for y in self.years:
        #     ref_timesteps = self.flows.loc[self.flows.Year==y,'Timestep'].to_list()
        #     self.model.addConstr(
        #                 gp.quicksum(self.arcFlows['J3','Affinity and Surrey','water',t]
        #                             for t in ref_timesteps) \
        #                 <= constants['annual_flow_to_affinity_surrey'],'SurrAnnual')

        #----
        # FLOWS TO AFFINITY
        #----

        #-
        # Sum of flow into London Storages from Upper Thames
        self.model.addConstrs((
                    self.arcFlows['J3','Brent',k,t] + \
                    self.arcFlows['J3','Queen Mother',k,t] + \
                    self.arcFlows['J3','Island Barn',k,t] + \
                    self.arcFlows['J3','Hilfield Park',k,t] + \
                    self.arcFlows['J3','Walthamstow No.4',k,t] <= 3500
                    for k in ['water']
                    for t in self.timesteps),'LonStorCap')

        #-
        # Sum of Lee abstraction
        for y in self.years:
            ref_timesteps = self.flows.loc[self.flows.Year==y,'Timestep'].to_list()
            self.model.addConstr(
                        gp.quicksum(self.arcFlows['Lee abstraction',j,'water',t]
                                    for j in ['Brent','Hilfield Park','Walthamstow No.4','King George V']
                                    for t in ref_timesteps) \
                        <= 200937,'SurrAnnual')

        # self.model.addConstr(
        #         gp.quicksum(self.arcFlows['Lee abstraction',j,'water',t]
        #                     for j in ['Brent','Hilfield Park','Walthamstow No.4','King George V']
        #                     for t in self.timesteps) <= 1200, 'LeeCap')

        #-
        # Super Sink
        self.model.addConstrs((
                    self.arcFlows.sum('*',j,'water',t) <= constants['super_source_maximum']
                    for j in ['SS1']
                    for t in self.timesteps),'LeeSS')

        #-
        # London ground water license
        for y in self.years:
            ref_timesteps = self.flows.loc[self.flows.Year==y,'Timestep'].to_list()
            self.model.addConstr(
                        gp.quicksum(self.arcFlows['Aggregated London GW','London','water',t]
                                    for t in ref_timesteps) \
                        <= constants['annual_london_groundwater'],'GWCap')

        #-
        # Teddington Environmental Flow constraint
        # get flow at water supply nodes
        water_flows  = utils.get_flow_at_nodes(flows=self.flows,list_of_nodes=['Upper Thames'])
        supply_dict  = utils.flows_as_dict(flows=water_flows)
        for t in self.timesteps:
            if supply_dict['Upper Thames',t] <= 2300:
                self.model.addConstr(
                    self.arcFlows.sum('*','Teddington Environmental Flow','water',t) \
                    == 168,'TedFlow')

        # #-
        # # River Lee Environmental Flow constraint
        # self.model.addConstrs((
        #                 self.arcFlows.sum('*',j,k,t)  == 45
        #                 for k in ['water']
        #                 for t in self.timesteps
        #                 for j in ['Lee environmental flow']),'demand')

        #-
        # Farmoor inflow constraints
        water_flows  = utils.get_flow_at_nodes(flows=self.flows,list_of_nodes=['Upper Thames'])
        supply_dict  = utils.flows_as_dict(flows=water_flows)

        self.model.addConstrs((
                        self.arcFlows.sum('Upper Thames','Farmoor','water',t)  <= 300
                        for t in self.timesteps
                        if supply_dict['Upper Thames',t] > 436.4),'demand')

        #-
        # Farmoor annual abstraction
        for y in self.years:
            ref_timesteps = self.flows.loc[self.flows.Year==y,'Timestep'].to_list()
            self.model.addConstr(
                        gp.quicksum(
                            self.arcFlows[i,j,k,t]
                            for i in ['Upper Thames']
                            for j in ['Farmoor']
                            for k in ['water']
                            for t in ref_timesteps) \
                                <= 55312,'FarmCap')

        #-
        # SWOX Discharge
        self.model.addConstrs((
                    self.arcFlows.sum('SWOX Discharge','J1','water',t) \
                    == 0.95*self.arcFlows.sum('*','SWOX','water',t)
                    for t in self.timesteps),'swoxDischarge')

        #-
        # WB Groundwater
        self.model.addConstrs((
                        self.arcFlows.sum(i,'*',k,t)  == 0
                        for i in ['WB GW']
                        for k in ['water']
                        for t in self.timesteps),'demand')

        # #----------------------------------------------------------------------
        # # MENA SPECIFIC CONSTRAINTS
        # #----------------------------------------------------------------------
        #
        # #---
        # # Existing solar energy output
        # for territory in self.nodes.Territory.unique():
        #     solar_nodes = utils.get_nodes(nodes=self.nodes,
        #                                   index_column='Subtype',lookup='solar',
        #                                   index_column2='Territory',lookup2=territory,
        #                                   index_column3='Status',lookup3='online')
        #
        #     # get capacity data
        #     solar_caps  = solar_nodes.set_index(keys=['Name']).to_dict()['Capacity']
        #
        #     # get irradiance data
        #     irradiance = utils.get_flow_at_nodes(flows=self.flows,list_of_nodes=[territory + ' solar'])
        #     irradiance_dict = utils.flows_as_dict(flows=irradiance)
        #
        #     # constrain
        #     self.model.addConstrs((
        #         self.arcFlows.sum(i,'*','electricity',t)  <= solar_caps[i] * irradiance_dict[i,t]
        #         for t in self.timesteps
        #         for i in solar_nodes.Name),'solar_supply')
        #
        # #---
        # # New Israel solar energy output
        # irradiance = utils.get_flow_at_nodes(flows=self.flows,list_of_nodes=['Israel solar'])
        # irradiance_dict = utils.flows_as_dict(flows=irradiance)
        # self.model.addConstrs((self.arcFlows[i,j,k,t] \
        #     <= irradiance_dict['Israel solar',t] * self.option_indices[i,k]
        #                        for i in ['Israel new solar']
        #                        for j in ['Israel generation']
        #                        for k in ['electricity']
        #                        for t in self.timesteps),'new_isr_solar')
        # # self.model.addConstrs((
        # #     self.arcFlows.sum('Israel new solar','*','electricity',t) \
        # #     <= irradiance_dict['Israel solar',t] * self.option_indices.sum('Israel new solar','electricity')
        # #     for t in self.timesteps),'new_isr_solar')
        #
        # #---
        # # New Jordan solar energy output
        # irradiance = utils.get_flow_at_nodes(flows=self.flows,list_of_nodes=['Jordan solar'])
        # irradiance_dict = utils.flows_as_dict(flows=irradiance)
        # self.model.addConstrs((self.arcFlows[i,j,k,t] \
        #     <= irradiance_dict['Jordan solar',t] * self.option_indices[i,k]
        #                        for i in ['Jordan new solar']
        #                        for j in ['Jordan generation']
        #                        for k in ['electricity']
        #                        for t in self.timesteps),'new_jor_solar')
        #
        # # self.model.addConstrs((
        # #     self.arcFlows.sum('Jordan new solar','*','electricity',t) \
        # #     <= irradiance_dict['Jordan solar',t] * self.option_indices.sum('Jordan new solar','electricity')
        # #     for t in self.timesteps),'new_jor_solar')
        #
        # #---
        # # Existing wind energy output
        # for territory in self.nodes.Territory.unique():
        #     wind_nodes = utils.get_node_names(nodes=self.nodes,
        #                                       index_column='Subtype',lookup='wind',
        #                                       index_column2='Territory',lookup2=territory)
        #
        #     # get capacity data
        #     wind_caps  = self.nodes.set_index(keys=['Name']).to_dict()['Capacity']
        #
        #     # get irradiance data
        #     irradiance = utils.get_flow_at_nodes(flows=self.flows,list_of_nodes=[territory + ' wind'])
        #     irradiance_dict = utils.flows_as_dict(flows=irradiance)
        #
        #     # constrain
        #     self.model.addConstrs((
        #         self.arcFlows.sum(i,'*','electricity',t)  <= wind_caps[i] * irradiance_dict[i,t]
        #         for t in self.timesteps
        #         for i in wind_nodes),'wind_supply')
        #
        # #---
        # # Egypt export to Gaza
        # self.model.addConstrs((
        #     self.arcFlows.sum('Egypt generation','Gaza energy demand','electricity',t)  <= 99999999
        #     for t in self.timesteps),'egypt_import')
        #
        # #---
        # # Israel's coal output must be half of capacity
        # coal_capacity = self.nodes.loc[self.nodes.Name=='Israel coal','Capacity'].values[0]
        # self.model.addConstrs((
        #     self.arcFlows.sum('Israel coal','Israel generation','electricity',t)  >= 0.5 * coal_capacity
        #     for t in self.timesteps),'coal_cap')
        #
        # ccgt_capacity = self.nodes.loc[self.nodes.Name=='Israel ccgt','Capacity'].values[0]
        # self.model.addConstrs((
        #     self.arcFlows.sum('Israel ccgt','Israel generation','electricity',t)  >= 0.5 * ccgt_capacity
        #     for t in self.timesteps),'ccgt_cap')
        #
        # #---
        # # RES constraint: Israel's total annual supply must be 17% renewable by 2030
        # def isr_res_target(target,year):
        #     reference_timesteps = self.time_ref[self.time_ref.Year==year].Timestep
        #     self.model.addConstr(
        #     gp.quicksum(
        #         target*(self.arcFlows['Israel solar','Israel generation','electricity',t] \
        #                 + self.arcFlows['Israel wind','Israel generation','electricity',t] \
        #                 + self.arcFlows['Israel ccgt','Israel generation','electricity',t] \
        #                 + self.arcFlows['Israel coal','Israel generation','electricity',t] \
        #                 + self.arcFlows['Israel new solar','Israel generation','electricity',t] \
        #                 + self.arcFlows['Israel natural gas','Israel generation','electricity',t]) \
        #         for t in reference_timesteps) ==
        #     gp.quicksum(
        #         self.arcFlows['Israel solar','Israel generation','electricity',t] \
        #         + self.arcFlows['Israel wind','Israel generation','electricity',t]
        #         + self.arcFlows['Israel new solar','Israel generation','electricity',t]
        #         for t in reference_timesteps),'isr_res') \
        # # target 1
        # isr_res_target(target=0.3,year=2030)
        #
        # #---
        # # RES constraint: Jordan's total annual supply must be 20% renewable by 2030
        # def jor_res_target(target,year):
        #     reference_timesteps = self.time_ref[self.time_ref.Year==year].Timestep
        #     self.model.addConstr(
        #     gp.quicksum(
        #         target*(self.arcFlows['Jordan solar','Jordan generation','electricity',t] \
        #                 + self.arcFlows['Jordan wind','Jordan generation','electricity',t] \
        #                 + self.arcFlows['Jordan ccgt','Jordan generation','electricity',t] \
        #                 + self.arcFlows['Jordan coal','Jordan generation','electricity',t] \
        #                 + self.arcFlows['Jordan new solar','Jordan generation','electricity',t] \
        #                 + self.arcFlows['Jordan natural gas','Jordan generation','electricity',t])
        #         for t in reference_timesteps) ==
        #     gp.quicksum(
        #         self.arcFlows['Jordan solar','Jordan generation','electricity',t] \
        #         + self.arcFlows['Jordan wind','Jordan generation','electricity',t]
        #         + self.arcFlows['Jordan new solar','Jordan generation','electricity',t]
        #         for t in reference_timesteps),'isr_res') \
        # # target 1
        # jor_res_target(target=0.3,year=2030)
        #
        # #---
        # # Random demand at Gaza
        # self.model.addConstrs((
        #                     self.arcFlows.sum('*',j,'electricity',t)  == 52
        #                     for t in self.timesteps
        #                     for j in ['Gaza energy demand']),'energy_demand')
        

        print(time.clock() - start_time, "seconds")
        print('------------- MODEL BUILD COMPLETE -------------')






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
