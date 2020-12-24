'''
    models.py
        submodels used within infrasim

    @amanmajid
'''

from . import utils

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
