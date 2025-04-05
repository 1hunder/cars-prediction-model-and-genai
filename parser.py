import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import time

# -------------------- CONFIG --------------------
BASE_URL = "https://www.otomoto.pl/osobowe?page={}"
MAX_PAGES = 500
MAX_THREADS = 20

# Label mapping based on your HTML
LABEL_MAP = {
    "Marka pojazdu": "Brand",
    "Model pojazdu": "Model",
    "Rok produkcji": "Year",
    "Przebieg": "Mileage",
    "Rodzaj paliwa": "Fuel Type",
    "Skrzynia biegów": "Transmission",
    "Pojemność skokowa": "Engine Size",
    "Kolor": "Color"
}

# -------------------- SCRAPING FUNCTIONS --------------------

def get_offer_links(pages):
    offer_links = set()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    for page in range(1, pages + 1):
        url = BASE_URL.format(page)
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.select('a[href*="/oferta/"]')
            for tag in links:
                href = tag.get("href")
                if href:
                    offer_links.add(href.split("?")[0])
            print(f"✅ Page {page}: {len(links)} links")
        except Exception as e:
            print(f"❌ Error on page {page}: {e}")
    return list(offer_links)


def parse_offer(url):
    data = {
        "Brand": None, "Model": None, "Year": None, "Mileage": None,
        "Fuel Type": None, "Transmission": None, "Engine Size": None,
        "Horsepower": None, "Color": None, "Price": None,
        "Description": None, "URL": url
    }

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        def get_by_testid(testid):
            container = soup.select_one(f'div[data-testid="{testid}"]')
            if container:
                value = container.select_one("p.ekwurce9")
                return value.get_text(strip=True) if value else None
            return None

        data["Brand"] = get_by_testid("make")
        data["Model"] = get_by_testid("model")
        data["Year"] = get_by_testid("year")
        data["Mileage"] = get_by_testid("mileage")
        data["Fuel Type"] = get_by_testid("fuel_type")
        data["Transmission"] = get_by_testid("gearbox")
        data["Engine Size"] = get_by_testid("engine_capacity")
        data["Horsepower"] = get_by_testid("engine_power")
        data["Color"] = get_by_testid("color")

        # Price
        price_tag = soup.select_one("span.offer-price__number")
        if price_tag:
            data["Price"] = price_tag.get_text(strip=True).replace("PLN", "").replace(" ", "").replace("\xa0", "")

        # Description
        desc_tag = soup.select_one("div.ooa-unlmzs")
        if desc_tag:
            data["Description"] = desc_tag.get_text(strip=True)

    except Exception as e:
        print(f"❌ Error parsing {url}: {e}")

    return data

# -------------------- MAIN SCRAPE LOGIC --------------------
start = time.time()
all_links = get_offer_links(MAX_PAGES)
print(f"🔗 Total unique links: {len(all_links)}")

results = []
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = [executor.submit(parse_offer, link) for link in all_links]
    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()
        if result:
            results.append(result)
        if i % 100 == 0:
            print(f"✅ Parsed {i} offers")

# Save to DataFrame
df = pd.DataFrame(results)
print(f"⏱️ Done in {round(time.time() - start, 2)} seconds")
df.to_csv("otomoto_fast_scraped.csv", index=False, encoding="utf-8-sig")
df.head()