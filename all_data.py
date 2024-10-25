from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service as FirefoxService
import pandas as pd
from time import sleep

#https://www.wunderground.com/history/daily/ca/quebec-city/CYQB/date/2023-12-28
#https://www.wunderground.com/weather/ca/chicoutimi
# TODO: scrap https://www.wunderground.com/history/daily/ca/la-baie/CYBG/date/2023-%M-%D 

YEAR = 2020
agenda = {"1":31,
          "2":28,
          "3":31,
          "4":30,
          "5":31,
          "6":30,
          "7":31,
          "8":31,
          "9":30,
          "10":31,
          "11":30,
          "12":31
} 
HEADER = ["Time", "Temperature" ,"Dew Point", "Humidity", "Wind", "Wind Speed", "Wind Gust", "Pressure", "Precip.", "Condition"]
DRIVER = webdriver.Firefox(service=FirefoxService(executable_path='/usr/local/bin/geckodriver'))
DATA = []

def scrape_weather_data(year = YEAR, month, day):
    # Construire l'URL
    url = f"https://www.wunderground.com/history/daily/ca/la-baie/CYBG/date/{year}-{month:02d}-{day:02d}"
    DRIVER.implicitly_wait(5)
    DRIVER.get(url)
    if month == 1 and day == 1  :
        sleep(2)  # Attendre quelques secondes pour que le contenu dynamique se charge
    print("WAITED")
    sleep(0.5)
    try:
        table = DRIVER.find_element(By.XPATH, '//table[@class="mat-table cdk-table mat-sort ng-star-inserted"]')
    except:
        print(f"[ERROR] No data found for the date : {day}/{YEAR}/{year}")
    rows = table.find_elements(By.TAG_NAME, 'tr')

    
    i = 0
    print(f"[INFO] {len(rows)} rows détécté")
    for row in rows:
        cols = row.find_elements(By.TAG_NAME, 'td')
        if cols:
            tmp_data = [col.text for col in cols]
            tmp_data[0] = f"{year}/{month}/{day} {tmp_data[0]}"
            DATA.append(tmp_data)
        
        elif i > len(rows) - 11:
            break
        i+=1


for month in agenda:
    for day in range(1,agenda[month] + 1):
        print(f"[INFO] Scraping date: {day}/{month}/{YEAR}")
        scrape_weather_data(YEAR, int(month), day)
        DRIVER.find_element(By.TAG_NAME, 'body').send_keys(Keys.CONTROL + 't')


df = pd.DataFrame(data, columns=HEADER)

if df is not None:
    # 1. Conversion des températures de Fahrenheit à Celsius
    # Formule : (F - 32) * 5/9
    df['Temperature (°C)'] = df['Temperature'].str.replace('°F', '').astype(float).apply(lambda f: (f - 32) * 5/9)
    # 2. Conversion de la vitesse du vent de mph à m/s
    # Formule : 1 mph = 0.44704 m/s
    df['Wind Speed (m/s)'] = df['Wind Speed'].str.replace(' mph', '').astype(float) * 0.44704
    # 3. Conversion de la pression de inHg à Pa
    # Formule : 1 inHg = 3386.39 Pa
    df['Pressure (Pa)'] = df['Pressure'].str.replace(' in', '').astype(float) * 3386.39
    print(f"[INFO] Succesfull created DataFrame : \n{df.head}")
    df.to_csv(f"WD__{YEAR}.csv", index=False)

#driver.quit()