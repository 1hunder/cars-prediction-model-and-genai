import pandas as pd
import time
import concurrent.futures
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException

# ============ CONFIG =============
INPUT_CSV = "otomoto_fast_scraped.csv"
OUTPUT_CSV = "otomoto_fixed_selenium.csv"
MAX_THREADS = 2
HEADLESS = True
CHROMEDRIVER_PATH = "./chromedriver.exe"

# ============ PARSER ============
def parse_offer_with_selenium(url):
    data = {
        "Brand": None, "Model": None, "Year": None, "Mileage": None,
        "Fuel Type": None, "Transmission": None, "Engine Size": None,
        "Horsepower": None, "Color": None, "Price": None,
        "Description": None, "URL": url
    }

    try:
        options = webdriver.ChromeOptions()
        if HEADLESS:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")

        service = Service(CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=options)

        driver.set_page_load_timeout(15)
        driver.get(url)
        time.sleep(2) 

        def safe_find(testid):
            try:
                el = driver.find_element(By.CSS_SELECTOR, f'div[data-testid="{testid}"] p.ekwurce9')
                return el.text.strip()
            except NoSuchElementException:
                return None

        data["Brand"] = safe_find("make")
        data["Model"] = safe_find("model")
        data["Year"] = safe_find("year")
        data["Mileage"] = safe_find("mileage")
        data["Fuel Type"] = safe_find("fuel_type")
        data["Transmission"] = safe_find("gearbox")
        data["Engine Size"] = safe_find("engine_capacity")
        data["Horsepower"] = safe_find("engine_power")
        data["Color"] = safe_find("color")

        try:
            price = driver.find_element(By.CSS_SELECTOR, "span.offer-price__number")
            data["Price"] = price.text.replace("PLN", "").replace(" ", "").replace("\xa0", "")
        except NoSuchElementException:
            pass

        try:
            desc = driver.find_element(By.CSS_SELECTOR, "div.ooa-unlmzs")
            data["Description"] = desc.text.strip()
        except NoSuchElementException:
            pass

        driver.quit()
    except Exception as e:
        print(f"❌ Failed: {url} | {e}")
        try:
            driver.quit()
        except:
            pass

    return data

# ============ MAIN WORKFLOW ============
df = pd.read_csv(INPUT_CSV)
df_with_nans = df[df.isna().any(axis=1)].copy()
print(f"🔍 Found {len(df_with_nans)} rows to fix")

urls_to_fix = df_with_nans["URL"].dropna().unique().tolist()
results = []

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    future_to_url = {executor.submit(parse_offer_with_selenium, url): url for url in urls_to_fix}
    for i, future in enumerate(concurrent.futures.as_completed(future_to_url), 1):
        result = future.result()
        if result:
            results.append(result)
        print(f"✅ Parsed {i}/{len(urls_to_fix)}")

df_fixed = pd.DataFrame(results)
df.set_index("URL", inplace=True)
df_fixed.set_index("URL", inplace=True)
df.update(df_fixed)
df.reset_index(inplace=True)

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"💾 Saved updated file to: {OUTPUT_CSV}")
