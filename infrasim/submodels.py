'''
    models.py
        submodels used within infrasim

    @amanmajid
'''

from . import utils

def solar_pv(area=10,month=7,panel_yield=0.76,irradiance=20):
    '''
    Solar PV model

    Arguments:
    ----------

        *area* : float, default 10
            PV array area (square meters)

        *month* : int, default 7
            Current month (to compute seasonal performance ratio)

        *panel_yield* : float, default 0.55
            Solar panel yield or efficiency (%)

        *irradiance* : float, default 20
            Incoming irradiance

    Returns:
    --------

        *output* : electricity output (kWh)

    '''
    # seasonal factors
    performance_ratio = {1  :   0.62,
                         2  :   0.62,
                         3  :   0.62,
                         4  :   0.50,
                         5  :   0.38,
                         6  :   0.38,
                         7  :   0.38,
                         8  :   0.38,
                         9  :   0.42,
                         10 :   0.42,
                         11 :   0.42,
                         12 :   0.62}

    # compute outputs
    output = area * (irradiance*0.001) * panel_yield * performance_ratio[month]
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
        ''' cut out speed '''
        output = 11.5
    elif wss<3:
        ''' cut in speed '''
        output = 0
    else:
        ''' power curve approximation '''
        output = 1.3328 * wss - 0.0939
    # convert to kWh per day
    output = utils.gwh_to_kwh(output/365)
    return output
