"""
Amazon Echo Dot Reviews Scraper 
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import pickle
import os

# ============================== Utilities ==============================

def get_name(review):
    t = review.find("span", class_="a-profile-name")
    return t.text.strip() if t else ""

def get_rating(review):
    try:
        return review.find("i", {"data-hook": "review-star-rating"}).text.strip()
    except:
        return ""

def get_review_text(review):
    t = review.find("span", {"data-hook": "review-body"})
    if t:
        inner = t.find("span")
        return inner.text.strip() if inner else t.text.strip()
    return ""

def get_colour(review):
    c = review.find("a", {"data-hook": "format-strip"})
    return c.text.replace("Colour:", "").strip() if c else ""

def get_date(review):
    d = review.find("span", {"data-hook": "review-date"})
    if d:
        dt = d.text.strip()
        return dt.split("on")[-1].strip() if "on" in dt else dt
    return ""

def get_review_title(review):
    t = review.find("a", {"data-hook": "review-title"})
    if t:
        sp = t.find_all("span")
        return sp[-1].text.strip() if sp else ""
    return ""

def get_review_id(review):
    return review.get('id', "") or ""

# ============================== Driver Setup ==============================

def setup_driver():
    options = Options()
    user_data = os.path.join(os.getcwd(), "chrome_profile")
    options.add_argument(f"user-data-dir={user_data}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-notifications")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def manual_login(driver):
    print("\n  Login required! Browser opening...")
    driver.get("https://www.amazon.in")
    input("  Login to Amazon manually, then press ENTER here...")
    pickle.dump(driver.get_cookies(), open("amazon_cookies.pkl", "wb"))
    print("  Cookies saved.")

def load_cookies(driver):
    if os.path.exists("amazon_cookies.pkl"):
        driver.get("https://www.amazon.in")
        for ck in pickle.load(open("amazon_cookies.pkl", "rb")):
            try:
                driver.add_cookie(ck)
            except:
                pass
        return True
    return False

# ============================== Scraper ==============================

def scrape_reviews(asin, start_page, end_page, force_login):
    driver = setup_driver()

    if force_login or not load_cookies(driver):
        manual_login(driver)

    seen = set()
    data = { "name":[], "rating":[], "review":[], "colour":[], "date":[], "title":[], "review_id":[] }

    base = f"https://www.amazon.in/product-reviews/{asin}/?reviewerType=all_reviews"

    print(f"\n  Starting scrape for ASIN: {asin}")
    driver.get(base)
    time.sleep(3)

    for page in range(start_page, end_page + 1):
        print(f" Page {page}...", end=" ")

        if page > 1:
            try:
                next_btn = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "li.a-last a"))
                )
                driver.execute_script("arguments[0].scrollIntoView();", next_btn)
                next_btn.click()
            except:
                driver.get(f"{base}&pageNumber={page}")

        time.sleep(random.uniform(3,5))

        # scroll page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2)")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        items = soup.find_all("div", {"data-hook": "review"}) or soup.find_all("li", {"data-hook":"review"})

        if not items:
            print(" No reviews, maybe blocked")
            continue

        new = 0
        for r in items:
            rid = get_review_id(r)
            text = get_review_text(r)
            uk = rid + text[:40]

            if uk in seen or not text:
                continue

            seen.add(uk); new += 1
            data["name"].append(get_name(r))
            data["rating"].append(get_rating(r))
            data["review"].append(text)
            data["colour"].append(get_colour(r))
            data["date"].append(get_date(r))
            data["title"].append(get_review_title(r))
            data["review_id"].append(rid)

        print(f" {new} new")

        if new == 0:
            print(" No new reviews -> stopping")
            break

    driver.quit()
    return data

# ============================== Main ==============================

if __name__ == "__main__":
    ASIN = "B09B8XJDW5"  #  Echo Dot 5th Gen
    START = 1
    END = 80             
    FORCE_LOGIN = False  # Set True on first run

    print("="*80, "\nAMAZON ECHO DOT REVIEW SCRAPER\n", "="*80)
    result = scrape_reviews(ASIN, START, END, FORCE_LOGIN)

    df = pd.DataFrame(result)
    df = df[df['review'].str.strip() != ""]
    df.drop("review_id", axis=1, inplace=True)

    file = "echo_dot_reviews.csv"
    df.to_csv(file, index=False, encoding="utf-8-sig")

    print(f"\n DONE — Total valid reviews: {len(df)}")
    print(f" Saved to {file}")
    print(df.head())
