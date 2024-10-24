import pandas as pd

#https://www.wunderground.com/history/daily/ca/quebec-city/CYQB/date/2023-12-28
#https://www.wunderground.com/weather/ca/chicoutimi
# TODO: scrap https://www.wunderground.com/history/daily/ca/la-baie/CYBG/date/2023-%M-%D 
#       M = 


# Define the data for the temperature records
data = {
    "Time": [
        "1:00 AM", "2:00 AM", "3:00 AM", "4:00 AM", "5:00 AM", "6:00 AM", "7:00 AM", "8:00 AM", "9:00 AM", "10:00 AM",
        "11:00 AM", "12:00 PM", "1:00 PM", "2:00 PM", "3:00 PM", "4:00 PM", "5:00 PM", "6:00 PM", "7:00 PM", "8:00 PM",
        "9:00 PM", "10:00 PM", "11:00 PM", "12:00 AM"
    ],
    "Temperature": [
        "34 °F", "34 °F", "34 °F", "34 °F", "32 °F", "32 °F", "32 °F", "30 °F", "32 °F", "32 °F",
        "34 °F", "34 °F", "36 °F", "34 °F", "34 °F", "34 °F", "32 °F", "32 °F", "32 °F", "32 °F",
        "32 °F", "32 °F", "34 °F", "34 °F"
    ],
    "Dew Point": [
        "27 °F", "27 °F", "21 °F", "23 °F", "27 °F", "28 °F", "25 °F", "27 °F", "27 °F", "25 °F",
        "23 °F", "25 °F", "27 °F", "27 °F", "27 °F", "27 °F", "27 °F", "27 °F", "28 °F", "28 °F",
        "28 °F", "28 °F", "28 °F", "28 °F"
    ],
    "Humidity": [
        "75 %", "75 %", "60 %", "65 %", "80 %", "87 %", "75 %", "86 %", "80 %", "75 %",
        "65 %", "70 %", "70 %", "75 %", "75 %", "75 %", "80 %", "80 %", "87 %", "87 %",
        "87 %", "87 %", "81 %", "81 %"
    ],
    "Wind": [
        "E", "NE", "E", "ENE", "ENE", "NE", "E", "E", "E", "ENE",
        "ENE", "ENE", "E", "ENE", "ENE", "E", "E", "ENE", "ENE", "ENE",
        "ENE", "ENE", "ENE", "ENE"
    ],
    "Wind Speed": [
        "7 mph", "9 mph", "7 mph", "13 mph", "9 mph", "8 mph", "9 mph", "10 mph", "8 mph", "14 mph",
        "14 mph", "10 mph", "10 mph", "15 mph", "14 mph", "9 mph", "12 mph", "10 mph", "15 mph", "9 mph",
        "10 mph", "9 mph", "8 mph", "9 mph"
    ],
    "Wind Gust": [
        "0 mph", "0 mph", "0 mph", "0 mph", "0 mph", "0 mph", "0 mph", "0 mph", "0 mph", "0 mph",
        "0 mph", "0 mph", "0 mph", "0 mph", "0 mph", "17 mph", "0 mph", "0 mph", "0 mph", "0 mph",
        "0 mph", "0 mph", "0 mph", "0 mph"
    ],
    "Pressure": [
        "29.78 in", "29.77 in", "29.76 in", "29.74 in", "29.75 in", "29.73 in", "29.72 in", "29.73 in", "29.74 in", "29.73 in",
        "29.74 in", "29.72 in", "29.70 in", "29.66 in", "29.64 in", "29.64 in", "29.64 in", "29.63 in", "29.63 in", "29.62 in",
        "29.61 in", "29.59 in", "29.56 in", "29.55 in"
    ],
    "Precip.": [
        "0.0 in", "0.0 in", "0.0 in", "0.0 in", "0.0 in", "0.0 in", "0.0 in", "0.0 in", "0.0 in", "0.0 in",
        "0.0 in", "0.0 in", "0.0 in", "0.0 in", "0.0 in", "0.0 in", "0.0 in", "0.0 in", "0.0 in", "0.0 in",
        "0.0 in", "0.0 in", "0.0 in", "0.0 in"
    ],
    "Condition": [
        "Cloudy", "Cloudy", "Cloudy", "Cloudy", "Cloudy", "Cloudy", "Partly Cloudy", "Partly Cloudy", "Mostly Cloudy", "Mostly Cloudy",
        "Cloudy", "Mostly Cloudy", "Cloudy", "Cloudy", "Cloudy", "Cloudy", "Cloudy", "Cloudy", "Cloudy", "Cloudy",
        "Cloudy", "Cloudy", "Cloudy", "Cloudy"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV file
csv_file_path = "temperature_data_28_12_2023.csv"
df.to_csv(csv_file_path, index=False)

# Return the file path for download
csv_file_path
