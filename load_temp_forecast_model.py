import pandas as pd
import matplotlib.pyplot as plt
import os
from prophet import Prophet

# Specifying File Paths, this is my path, you will need to replace with yours
th_file = 'C:\\Users\\kiptoo\\Desktop\\Upwork_Projects\\Clients Data\\Travis\\load_files_wMay\\th.csv'
th_temp_file = 'C:\\Users\\kiptoo\\Desktop\\Upwork_Projects\\Clients Data\\Travis\\temperature_files_wMay\\th_temp_w_may.csv'
rp_file = 'C:\\Users\\kiptoo\\Desktop\\Upwork_Projects\\Clients Data\\Travis\\load_files_wMay\\rp.csv'
rp_temp_file = 'C:\\Users\\kiptoo\\Desktop\\Upwork_Projects\\Clients Data\\Travis\\temperature_files_wMay\\rp_temp_w_may.csv'
lsc_file = 'C:\\Users\\kiptoo\\Desktop\\Upwork_Projects\\Clients Data\\Travis\\load_files_wMay\\lsc.csv'
lsc_temp_file = 'C:\\Users\\kiptoo\\Desktop\\Upwork_Projects\\Clients Data\\Travis\\temperature_files_wMay\\lsc_temp_w_may.csv'

# Loading data into DataFrames
th_data = pd.read_csv(th_file)
th_temp_data = pd.read_csv(th_temp_file)
rp_data = pd.read_csv(rp_file)
rp_temp_data = pd.read_csv(rp_temp_file)
lsc_data = pd.read_csv(lsc_file)
lsc_temp_data = pd.read_csv(lsc_temp_file)

# Convert timestamp to datetime
th_data['Datetime MST'] = pd.to_datetime(th_data['Datetime MST'], format='%m/%d/%Y %H:%M')
rp_data['Datetime EST'] = pd.to_datetime(rp_data['Datetime EST'], format='%m/%d/%Y %H:%M')
lsc_data['Datetime MST'] = pd.to_datetime(lsc_data['Datetime MST'], format='%m/%d/%Y %H:%M')

# Set the timestamp as the index
th_data.set_index('Datetime MST', inplace=True)
rp_data.set_index('Datetime EST', inplace=True)
lsc_data.set_index('Datetime MST', inplace=True)

# Resample the load data to 15-minute intervals
th_data = th_data.resample('15T').mean()
rp_data = rp_data.resample('15T').mean()
lsc_data = lsc_data.resample('15T').mean()

# Convert timestamp to datetime
th_temp_data['time'] = pd.to_datetime(th_temp_data['time'], format='%Y-%m-%d %H:%M:%S')
rp_temp_data['time'] = pd.to_datetime(rp_temp_data['time'], format='%Y-%m-%d %H:%M:%S')
lsc_temp_data['time'] = pd.to_datetime(lsc_temp_data['time'], format='%Y-%m-%d %H:%M:%S')

# Set the timestamp as the index
th_temp_data.set_index('time', inplace=True)
rp_temp_data.set_index('time', inplace=True)
lsc_temp_data.set_index('time', inplace=True)

# Define the Prophet model parameters
seasonality_mode = 'multiplicative'

# Perform Prophet forecasting for each dataset
datasets = {
    'th_data': {'data': th_data, 'time_column': 'Datetime MST', 'load_column': 'Load'},
    'th_temp_data': {'data': th_temp_data, 'time_column': 'time', 'load_column': 'temp'},
    'rp_data': {'data': rp_data, 'time_column': 'Datetime EST', 'load_column': 'Load'},
    'rp_temp_data': {'data': rp_temp_data, 'time_column': 'time', 'load_column': 'temp'},
    'lsc_data': {'data': lsc_data, 'time_column': 'Datetime MST', 'load_column': 'Load'},
    'lsc_temp_data': {'data': lsc_temp_data, 'time_column': 'time', 'load_column': 'temp'},
}

forecasts = {}

for dataset_name, dataset_info in datasets.items():
    # Extract the relevant columns for time and load/temperature
    dataset = dataset_info['data']
    time_column = dataset_info['time_column']
    load_column = dataset_info['load_column']

    # Create a DataFrame for Prophet with time and load/temperature columns
    prophet_df = pd.DataFrame({
        'ds': dataset.index,  # Time column
        'y': dataset[load_column]  # Load/Temperature column
    })

    # Create a Prophet model
    model = Prophet(seasonality_mode=seasonality_mode)

    # Fit the model to the data
    model.fit(prophet_df)

    # Create future dates for forecasting
    future_dates = model.make_future_dataframe(periods=4 * 24 * 30, freq='15T')  # 4 * 24 * 30 = 30 days with 15-minute intervals

    # Generate forecasts
    forecast = model.predict(future_dates)

    # Store the forecast for the dataset
    forecasts[dataset_name] = forecast

    # Plot the forecasted load/temperature
    fig = model.plot(forecast, xlabel=time_column, ylabel='Load' if 'Load' in load_column else 'Temperature')
    plt.title(f'Load/Temperature Forecast - {dataset_name}')
    plt.show()

    # Plot the components (trend, seasonality, holidays)
    fig = model.plot_components(forecast)
    plt.title(f'Components - {dataset_name}')
    plt.show()

    # Calculate error metrics
    actual_values = dataset[load_column].values
    predicted_values = forecast['yhat'].values[:len(actual_values)]
    errors = actual_values - predicted_values
    mae = abs(errors).mean()
    rmse = ((errors ** 2).mean()) ** 0.5

    print(f'Error metrics for {dataset_name}:')
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print('')

# Access the forecasts for each dataset
for dataset_name, forecast in forecasts.items():
    print(f'Forecast for {dataset_name}:')
    print(forecast[['ds', 'yhat']].tail())  # Display the last few rows of the forecast

# Save the forecasts to CSV files
output_folder = 'C:\\Users\\kiptoo\\Desktop\\Upwork_Projects\\Clients Data\\Travis\\load_files_wMay'

for dataset_name, forecast in forecasts.items():
    # Extract the dataset information
    dataset_info = datasets[dataset_name]
    dataset = dataset_info['data']
    time_column = dataset_info['time_column']
    load_column = dataset_info['load_column']

    # Extract the forecasted values and dates
    forecast_values = forecast[['ds', 'yhat']]

    # Rename the columns to match the original dataset
    forecast_values.columns = [time_column, 'predicted_mean']

    # Define the filename for the CSV file
    filename = f'{dataset_name}_forecast11.csv'

    # Save the forecast DataFrame to a CSV file
    forecast_values.to_csv(os.path.join(output_folder, filename), index=False)

    print(f'Forecast for {dataset_name} saved to {filename}')

