'''
    metainfo.py
        Meta-data associated with the INFRASIM model

    @amanmajid
'''

metainfo = {
            'i_field'               : 'Start',
            'j_field'               : 'End',
            'cost_column'           : 'Cost',
            'upper_bound'           : 'Maximum',
            'lower_bound'           : 'Minimum',
            'capex_column'          : 'CAPEX',
            'nodes_header'          : ['ID','Type','Subtype','Name','Status','Sector','k','Capacity','initCap','Cap_units','Built','Comments'],
            'edges_header'          : ['Start','End','Commodity','Cost','Minimum','Maximum'],
            'flow_header'           : ['Day','Month','Year'],
            'edge_index_variables'  : ['Start','End','Commodity','Timestep'],
            'infrasim_cache'        : '../data/demo/__infrasim__/',
            'outputs_figures'       : '../outputs/demo/figures/',
            'outputs_data'          : '../outputs/demo/statistics/',
            'temporal_resolution'   : 'daily',
            'scenario'              : 'baseline'
            }
