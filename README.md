# Best Buy Review Text Mining

This project was completed by Michael Fillinger for CSI 5810 at Oakland University.

The goal of this project is to analyze customer reviews for the Logitech Astro A50X gaming headset and build a text-based model that predicts whether a review is positive or not positive based only on the written text.
All reviews were collected directly from Best Buy using a custom Python scraping script.

## How to Run

1. Make sure the following files are in the same folder:
   `bestbuy_scrape.py`
   `project2.py`
   `astro_a50x_reviews_raw.csv`
   `requirements.txt`

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. (Optional) Re-scrape reviews from Best Buy:
   ```bash
   python bestbuy_scrape.py

4. Run the analysis and model training:
   ```bash
   python project2.py

5. View your output directly in the terminal, including accuracy, classification reports, and confusion matrices.
   
