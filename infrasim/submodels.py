'''
    models.py
        submodels used within infrasim

    @amanmajid
'''

import pandas as pd
from tqdm import tqdm

from . import utils
from .params import *

def solar_pv(area=10,month=7,efficiency=0.86,irradiance=20):
    '''
    Solar PV model

    Arguments:
    ----------

        *area* : float, default 10
            PV array area

        *month* : int, default 7
            Current month (important for seasonal variability)

        *efficiency* : float, default 0.86
            Panel efficiency

        *irradiance* : float, default 20
            Incoming irradiance

    Returns:
    --------

        *output* : electricity output (kWh)

    '''
    # seasonal factors
    seasonal_factors = {1:0.0149, 2:0.0394, 3:0.0707, 4:0.1237,
                        5:0.1597, 6:0.1767, 7:0.1863, 8:0.1728,
                        9:0.1498, 10:0.1003, 11:0.0522, 12:0.0217}
    # compute outputs
    output = area * seasonal_factors[month] * (irradiance*24*0.001) * efficiency
    return output

def wind_vestas2(wss=10):
    '''
    Wind Turbine of the Vestas 2.2 MW turbine

    Arguments:
    ----------

        *wss* : float, default 10
            Wind speed

    Returns:
    --------

        *output* : electricity output (kWh)

    '''

    # compute GWh per annum
    if wss>9:
        output = 11.5
    else:
        output = 1.3328 * wss - 0.0939
    # convert to kWh per day
    output = utils.gwh_to_kwh(output/365)
    return output

def compute_elec_tariffs(flow_data):
    '''Function to calculate Time-of-Use (TOU) and Baseline Mean (BLM)
    '''
    #off-peak
    rate1 = [1,2,3,4,5,6]
    #mid-peak
    rate2 = [7,8,9,10,11,12,13,14,15,16,17]
    #peak
    rate3 = [18,19,20,21]
    #off-peak 2
    rate4 = [22,23,24]
    for y in tqdm(flow_data.Year.unique()):
        # compute annual tou
        idx_frame2 = flow_data.loc[(flow_data.Year==y)]
        pp1 = idx_frame2.loc[idx_frame2.Hour.isin(rate1)].RTP_Price.mean()
        pp2 = idx_frame2.loc[idx_frame2.Hour.isin(rate2)].RTP_Price.mean()
        pp3 = idx_frame2.loc[idx_frame2.Hour.isin(rate3)].RTP_Price.mean()
        pp4 = idx_frame2.loc[idx_frame2.Hour.isin(rate4)].RTP_Price.mean()
        # off-peak
        flow_data.loc[(flow_data.Year==y) & \
                      (flow_data.Hour.isin(rate1)), 'TOU_Annual_Price'] = pp1
        # mid-peak
        flow_data.loc[(flow_data.Year==y) & \
                      (flow_data.Hour.isin(rate2)), 'TOU_Annual_Price'] = pp2
        # peak
        flow_data.loc[(flow_data.Year==y) & \
                      (flow_data.Hour.isin(rate3)), 'TOU_Annual_Price'] = pp3
        # off-peak 2
        flow_data.loc[(flow_data.Year==y) & \
                      (flow_data.Hour.isin(rate4)), 'TOU_Annual_Price'] = pp3

        for m in flow_data.Month.unique():
            for d in flow_data.Day.unique():
                idx_frame = flow_data.loc[(flow_data.Day==d) & \
                                           (flow_data.Year==y) & \
                                           (flow_data.Month==m)]
                #compute mean
                p1 = idx_frame.loc[idx_frame.Hour.isin(rate1)].RTP_Price.mean()
                p2 = idx_frame.loc[idx_frame.Hour.isin(rate2)].RTP_Price.mean()
                p3 = idx_frame.loc[idx_frame.Hour.isin(rate3)].RTP_Price.mean()
                p4 = idx_frame.RTP_Price.mean()
                #off-peak
                flow_data.loc[(flow_data.Day==d) & \
                               (flow_data.Year==y) & \
                               (flow_data.Month==m) & \
                               (flow_data.Hour.isin(rate1)),'TOU_Daily_Price'] = p1
                #mid-peak
                flow_data.loc[(flow_data.Day==d) & \
                               (flow_data.Year==y) & \
                               (flow_data.Month==m) & \
                               (flow_data.Hour.isin(rate2)),'TOU_Daily_Price'] = p2
                #peak
                flow_data.loc[(flow_data.Day==d) & \
                               (flow_data.Year==y) & \
                               (flow_data.Month==m) & \
                               (flow_data.Hour.isin(rate3)),'TOU_Daily_Price'] = p3
                #daily mean
                flow_data.loc[(flow_data.Day==d) & \
                               (flow_data.Year==y) & \
                               (flow_data.Month==m) ,'TOU_Daily_Price'] = p4

    # add a mean of the entire time series
    flow_data['Baseline_Price'] = flow_data.RTP_Price.mean() #* 1.1

    return flow_data

def wwtp_treatment_energy_use():
    print('to do')

def ts_forecast(flows,fields,start,end,freq='D',exceptions=None):
    '''
    Time series forecasting model

    Arguments:
    ----------

        *flows* : dataframe
            Data containing baseline flows. Must in INFRASIM format with
            date, hour, day, month, and year.

        *fields* : list
            Columns to forecast from flows data

        *start* : str
            Start date in MM/DD/YYYY format

        *end* : str
            End date in MM/DD/YYYY format

        *freq* : str, default daily ('D')
            Datetime frequency

        *exceptions* : list
            Columns to neglect with respect to growth rate

    Returns:
    --------

        *new_flows* : New flow data with forecast

    '''

    new_flows = pd.DataFrame({'Date' : pd.date_range(start=start, end=end, freq=freq)})
    new_flows['Day'] = new_flows.Date.dt.day
    new_flows['Month'] = new_flows.Date.dt.month
    new_flows['Year'] = new_flows.Date.dt.year

    if 'Hour' in flows.columns.tolist():
        new_flows['Hour'] = new_flows.Date.dt.hour 

    # loop over each column
    for f in fields:
        new_flows[f] = float(0)
        # loop over each row
        for i in tqdm(range(0,len(new_flows))):

            try:
                baseline_value = flows.loc[(flows.Hour==new_flows.at[i,'Hour']) & \
                                           (flows.Day==new_flows.at[i,'Day'])   & \
                                           (flows.Month==new_flows.at[i,'Month']),f].values[0]
            except:
                # Add mask value for leap year
                baseline_value = -99999

            # Exceptions
            if exceptions is not None:
                if f in exceptions:
                    new_flows.at[i,f] = baseline_value
            else:
                new_flows.at[i,f] = baseline_value * ( (1+variables['demand_growth_rate'])**(new_flows.at[i,'Year']-flows.iloc[-1]['Year']) )

    return new_flows

def household_water_demand():
    ei_factor = variables['household_water_demand_showering'] * constants ['showering_ei'] + \
                variables['household_water_demand_showering'] * constants ['showering_ei']

    return ei_factor
