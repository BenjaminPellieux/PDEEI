from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service as FirefoxService
import pandas as pd
from time import sleep

YEAR = 2023
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
# Configurer le chemin vers geckodriver (s'il est dans votre PATH, ce n'est pas nécessaire)
service = FirefoxService(executable_path='/usr/local/bin/geckodriver')
HEADER = ["Time", "Temperature" ,"Dew Point", "Humidity", "Wind", "Wind Speed", "Wind Gust", "Pressure Precip.", "Condition"]
# Démarrer le navigateur Firefox
driver = webdriver.Firefox(service=service)
data = []

def scrape_weather_data(year, month, day):
    # Construire l'URL
    url = f"https://www.wunderground.com/history/daily/ca/la-baie/CYBG/date/{year}-{month:02d}-{day:02d}"
    driver.implicitly_wait(10)
    driver.get(url)
    sleep(2)  # Attendre quelques secondes pour que le contenu dynamique se charge
    print("WAITED")

    sleep(2)
        # Rechercher la table (ajuster le sélecteur si nécessaire)
        
    table = driver.find_element(By.XPATH, '//table[@class="mat-table cdk-table mat-sort ng-star-inserted"]')
    rows = table.find_elements(By.TAG_NAME, 'tr')

    
    i = 0
    print(f"[DEBUG] {len(rows)=}")
    for row in rows:
        cols = row.find_elements(By.TAG_NAME, 'td')
        if cols:
            
            tmp_data = [col.text for col in cols]
            tmp_data[0] = f"{year}/{month}/{day} {tmp_data[0]}"
            print(f"[DEBUG] {i}/{len(rows)} {tmp_data}")
            data.append(tmp_data)
        elif i>len(rows)-11:
            break
        i+=1


for month in agenda:
    for day in range(1,agenda[month] + 1):
        print(f"[DEBUG] Scraping date: {day}/{month}/{YEAR}")
        scrape_weather_data(YEAR, int(month), day)
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.CONTROL + 't')

print(f"DATA {data=}")


df = pd.DataFrame(data, columns=headers)

if df is not None:
    print(df)
    df.to_csv("weather_data_seleniumv2.csv", index=False)

# Fermer le navigateur
#driver.quit()
