'''
    constants.py
        Assumed constants in the INFRASIM model

    @amanmajid
'''

variables = {
            # -THAMES
            'res_control_curve_l1'              : [0.76,0.84,0.91,0.95,0.95,0.94,0.90,0.85,0.80,0.76,0.74,0.73],
            'res_control_curve_l2'              : [0.62,0.74,0.85,0.91,0.92,0.90,0.82,0.72,0.62,0.57,0.56,0.56],
            'res_control_curve_l3'              : [0.37,0.43,0.50,0.54,0.55,0.56,0.56,0.56,0.53,0.47,0.41,0.36],
            # -HOUSEHOLD
            'pctg_demand_personal_washing'      : 0.39,         # % total demand https://www.ofwat.gov.uk/wp-content/uploads/2018/05/The-long-term-potential-for-deep-reductions-in-household-water-demand-report-by-Artesia-Consulting.pdf
            'pctg_demand_dish_washing'          : 0.10,         # % total demand
            'pctg_demand_clothes_washing'       : 0.12,         # % total demand
            'pctg_demand_wc_flushing'           : 0.24,         # % total demand
            'pctg_demand_external'              : 0.04,         # % total demand
            'household_water_demand_dish_wash'  : 0.11,         # % total demand
            }


constants = {
            # -WATER SUPPLY
            'water_network_coverage'            : 32000,        # km
            'surface_water_pumping_ei'          : 0.009,        # kWh/ML.km
            'groundwater_pumping_ei'            : 5,            # kWh/ML.m
            'groundwater_pumping_height'        : 10,           # m
            'desalination_brackish_ei'          : 2000,         # kWh/ML
            'desalination_ocean_ei'             : 3890,         # kWh/ML
            'water_treatment_ei'                : 250,          # kWh/ML
            'river_lee_pumping'                 : 1200,         # ML/day
            'river_lee_env_constraint'          : 45,           # ML/day
            'london_gw_pumping'                 : 355,          # ML/day
            'annual_flow_to_affinity_iver'      : 82964,        # ML/year
            'annual_flow_to_affinity_surrey'    : 65000,        # ML/year
            'annual_london_groundwater'         : 166970,       # ML/year
            'min_flow_teddington'               : 300,          # ML/day
            'std_flow_teddington'               : 800,          # ML/day
            # -WASTEWATER
            'water_to_wastewater_capture'       : 0.95,         # %
            'wastewater_treatment_ei'           : 1000,         # kWh/ML
            'wastewater_network_coverage'       : 108838,       # km
            'wastewater_network_capacity'       : 0,            # ML
            'wastewater_pumping_ei'             : 0.006,        # kWh/ML.km
            'wastewater_BOD_standard'           : 22,           # mg/L
            'wastewater_COD_standard'           : 50,           # mg/L
            'wastewater_SSs_standard'           : 25,           # mg/L
            'wastewater_NH3_standard'           : 8,            # mg/L
            'wastewater_plant_minimum_flow'     : 0,            # % of plant capacity
            'wwtp_energy_output'                : 49.6,         # kWh/ML
            # -ELECTRICITY
            'baseload_coefficient'              : 0.5,
            'storage_loss_coefficient'          : 0.1,
            'peak_demand_reserve'               : 0.2,
            'ocgt_ramping_rate'                 : 1200,         # MW/h
            'ccgt_ramping_rate'                 : 600,          # MW/h
            'coal_ramping_rate'                 : 260,          # MW/h
            'solar_ramping_rate'                : 12000,        # MW/h
            'wind_ramping_rate'                 : 3000,         # MW/h
            'nuclear_ramping_rate'              : 1200,         # MW/h
            'nat_gas_ramping_rate'              : 4000,         # MW/h
            'pumped_hydro_ramping_rate'         : 12000,        # MW/h
            'diesel_ramping_rate'               : 420,          # MW/h
            'solar_efficiency'                  : 0.48,         # %
            'wind_efficiency'                   : 0.33,         # %
            # -HOUSEHOLD
            'natural_gas_water_heating'         : 12000,        # kWh/ML
            'dish_washing_ei'                   : 22000,        # kWh/ML
            'clothes_washing_ei'                : 9450,         # kWh/ML
            'auxillary_water_use_ei'            : 0,            # kWh/ML
            # -OTHER
            'super_source_maximum'              : 10**12,
            'mask_value'                        : -999,
            }
