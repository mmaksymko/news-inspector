import random
import time
from playwright.sync_api import sync_playwright


def random_delay(min_seconds: float = 1.5, max_seconds: float = 3):
    time.sleep(random.uniform(min_seconds, max_seconds))

def mimic_user_interaction(page):
    # Simulate mouse movements
    for _ in range(random.randint(2, 5)):
        x = random.randint(50, 400)
        y = random.randint(50, 400)
        page.mouse.move(x, y, steps=random.randint(10, 25))
        random_delay(0.2, 0.5)

    # Simulate random scrolls
    for _ in range(random.randint(2, 6)):
        scroll_amount = random.randint(300, 800)
        page.mouse.wheel(0, scroll_amount)
        random_delay(1, 2)
        
def load_and_bypass_cloudflare(url:str, timeout: int = 5000) -> str:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": random.randint(1000, 1200), "height": random.randint(800, 1000)},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        )
        
        page = context.new_page()

        page.goto(url, timeout=timeout, wait_until="domcontentloaded")        
         
        mimic_user_interaction(page)
        
        random_delay(1, 3)
        
        content = page.content()
        browser.close()

        return content
    return content