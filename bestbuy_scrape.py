import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#review URLs (most recent)
BASE_URL= (
    "https://www.bestbuy.com/site/reviews/"
    "logitech-astro-a50-x-lightspeed-wireless-w-playsync-over-the-ear-gaming-headset-"
    "base-station-for-xbox-series-xs-ps5-pc-mac-black/6572603"
)

REVIEW_PAGE_URLS = [
    f"{BASE_URL}?sort=MOST_RECENT&variant=A&page=1",
    f"{BASE_URL}?sort=MOST_RECENT&variant=A&page=2",
    f"{BASE_URL}?sort=MOST_RECENT&variant=A&page=3",
]

HEADERS= {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


#fetch & parse reviews
def fetch_html(url):
    #download page HTML
    print(f"Fetching: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def parse_reviews_from_html(html, source_url):

    soup = BeautifulSoup(html, "html.parser")
    full_text = soup.get_text("\n")
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]

    reviews = []
    i = 0

    while i < len(lines):
        line = lines[i]

        m = re.match(r"Rated\s+([1-5])\s+out of 5 stars", line)
        if not m:
            i += 1
            continue

        rating = int(m.group(1))

        reviewer_name = None
        for k in range(i - 1, max(-1, i - 8), -1):
            cand = lines[k].strip()
            if not cand:
                continue
            if cand.startswith("Pros mentioned") or cand.startswith("Cons mentioned"):
                continue
            if cand.startswith("Rated ") or cand.startswith("Page "):
                continue
            if cand.startswith("Customers are saying"):
                continue
            if len(cand.split()) <= 3:
                reviewer_name = cand
                break

        title = ""
        review_date = ""
        body_lines =[]

        j = i + 1

        while j < len(lines) and not lines[j].startswith("Posted "):
            if lines[j].startswith("#"):
                title = lines[j].lstrip("#").strip()
            j += 1

        if j < len(lines) and lines[j].startswith("Posted "):
            review_date = lines[j].replace("Posted", "").strip()
            j += 1

        if j < len(lines) and lines[j].startswith("This reviewer received"):
            j += 1

        while j < len(lines):
            stop = lines[j]
            if (
                stop.startswith("This review is from")
                or stop.startswith("I would recommend this")
                or stop.startswith("Helpful (")
                or stop.startswith("Report")
                or stop.startswith("Brand response")
                or stop.startswith("Comment")
                or stop.startswith("Pros mentioned:")
                or stop.startswith("Cons mentioned:")
                or re.match(r"^Rated\s+[1-5]\s+out of 5 stars", stop)
                or stop.startswith("Page ")
                or stop.startswith("Customers are saying")
            ):
                break
            body_lines.append(stop)
            j += 1

        review_text = " ".join(body_lines).strip()

        if review_text:
            reviews.append(
                {
                    "source_url": source_url,
                    "reviewer_name": reviewer_name,
                    "review_title": title,
                    "review_text": review_text,
                    "star_rating": rating,
                    "review_date": review_date,
                }
            )

        i = j  

    print(f"Found {len(reviews)} reviews on this page.")
    return reviews


#scrape all reviews
all_reviews =[]

for url in REVIEW_PAGE_URLS:
    try:
        html = fetch_html(url)
        page_reviews = parse_reviews_from_html(html, url)
        all_reviews.extend(page_reviews)
        time.sleep(1.5) 
    except Exception as e:
        print(f"Error fetching/parsing {url}: {e}")

print(f"Total reviews collected: {len(all_reviews)}")

if not all_reviews:
    print("No reviews collected. Check your URLs or parsing logic.")
else:
    df = pd.DataFrame(all_reviews)
    df.to_csv("astro_a50x_reviews_raw.csv", index=False, encoding="utf-8")
    print("Saved raw reviews to astro_a50x_reviews_raw.csv")
    print(df.head())


#cleaning
if all_reviews:
    df = pd.read_csv("astro_a50x_reviews_raw.csv")

    df["review_title"] = df["review_title"].fillna("")
    df["review_text"] = df["review_text"].fillna("")
    df["full_text"] = (df["review_title"] + " " + df["review_text"]).str.strip()

    def basic_clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    df["clean_text"] = df["full_text"].apply(basic_clean)

    df = df[(df["clean_text"].str.len() > 0) & df["star_rating"].notna()]

    print("\nSample cleaned review:\n", df["clean_text"].iloc[0][:400], "...\n")

#prediction
    X = df["clean_text"]
    y = df["star_rating"]

    print("Star rating distribution:")
    print(y.value_counts())

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    #naive bayes pipeline
    nb_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("nb", MultinomialNB()),
        ]
    )

    print("\nTraining Naive Bayes model...")
    nb_pipeline.fit(X_train, y_train)
    y_pred_nb = nb_pipeline.predict(X_test)

    print("\nNaive Bayes Accuracy:", (y_pred_nb == y_test).mean())
    print("\nNaive Bayes Classification Report:")
    print(classification_report(y_test, y_pred_nb))

    print("Naive Bayes Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_nb))

    #logistic regression pipeline
    lr_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("lr", LogisticRegression(max_iter=1000)),
        ]
    )

    print("\nTraining Logistic Regression model...")
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)

    print("\nLogistic Regression Accuracy:", (y_pred_lr == y_test).mean())
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_lr))

    print("Logistic Regression Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr))
