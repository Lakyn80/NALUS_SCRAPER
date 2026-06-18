import requests, re

SEARCH_URL = "https://nalus.usoud.cz/Search/Search.aspx"
RESULTS_URL = "https://nalus.usoud.cz/Search/Results.aspx"
headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "cs,en-US;q=0.9"}

session = requests.Session()
r = session.get(SEARCH_URL, headers=headers, timeout=15)

hidden = {}
for m in re.finditer(r'<input[^>]+type=["\']hidden["\'][^>]+name=["\'](?P<n>[^"\']+)["\'][^>]+value=["\'](?P<v>.*?)["\']', r.text, re.IGNORECASE|re.DOTALL):
    hidden[m.group("n")] = m.group("v")

payload = {
    **hidden,
    "__EVENTTARGET": "",
    "__EVENTARGUMENT": "",
    "ctl00$MainContent$nalezy": "on",
    "ctl00$MainContent$usneseni": "on",
    "ctl00$MainContent$stanoviska_plena": "on",
    "ctl00$MainContent$naveti": "on",
    "ctl00$MainContent$vyrok": "on",
    "ctl00$MainContent$oduvodneni": "on",
    "ctl00$MainContent$odlisne_stanovisko": "on",
    "ctl00$MainContent$text": "",
    "ctl00$MainContent$decidedFrom": "1.1.2024",
    "ctl00$MainContent$decidedTo": "31.12.2024",
    "ctl00$MainContent$but_search": "Vyhledat",
}

resp = session.post(SEARCH_URL, data=payload, headers={**headers, "Content-Type": "application/x-www-form-urlencoded", "Referer": SEARCH_URL, "Origin": "https://nalus.usoud.cz"}, allow_redirects=True, timeout=15)
print("URL:", resp.url)
print("Status:", resp.status_code)

# Extract total results
m = re.search(r'Výsledky\s+\d+\s*[-–]\s*\d+\s+z\s+celkem\s+(\d+)', resp.text)
if m:
    print("Total results:", m.group(1))
else:
    # Try alternate pattern
    m2 = re.search(r'z celkem (\d+)', resp.text)
    print("Alt match:", m2.group(0) if m2 else "NOT FOUND")
    print("Výsledky in text:", "Výsledky" in resp.text)
