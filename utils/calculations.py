
def calculate_aqi(parameter, concentration):
    # Define specific AQI breakpoints and corresponding AQI values for each parameter based on Indian standards
    parameter_breakpoints = {
        'SO2': [0, 40, 80, 380, 800, 1600],
        'OZONE': [0, 50, 100, 168, 209, 748],
        'CO': [0, 30, 60, 90, 180, 280],
        'PM2.5': [0, 30, 60, 90, 120, 250],
        'NO2': [0, 40, 80, 180, 280, 400],
        'NH3': [0, 200, 400, 800, 1200, 1800],
        'PM10': [0, 50, 100, 250, 350, 430],
        'AQI': [0, 50, 100, 250, 350, 430],
    }

    aqi_values = [0, 50, 100, 200, 300, 400, 500]

    for i in range(len(parameter_breakpoints[parameter]) - 1):
        if parameter_breakpoints[parameter][i] <= concentration <= parameter_breakpoints[parameter][i + 1]:
            break

    return int(((aqi_values[i + 1] - aqi_values[i]) / (
                parameter_breakpoints[parameter][i + 1] - parameter_breakpoints[parameter][i])) * (
                concentration - parameter_breakpoints[parameter][i]) + aqi_values[i])

