from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM
import torch
import requests
from bs4 import BeautifulSoup
import re

from flask import Flask
import nest_asyncio
import asyncio
from pyppeteer import launch
import pickle

# Yelp URL scraping function


def yelpData(yelp_URL):
    # yelp_URL=
    r = requests.get(yelp_URL)
    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile('.*comment.*')
    results = soup.find_all('p', {'class': regex})
    reviews = [result.text for result in results]

    return reviews

# Amazon URL scraping function


def amzoneData(amazon_url):
    nest_asyncio.apply()

    async def scrape_amazon_reviews(url):
        browser = await launch(headless=True)
        page = await browser.newPage()

        # Increase the navigation timeout to 60 seconds (60000 ms)
        navigation_timeout = 60000
        await page.goto(url, waitUntil='networkidle0', timeout=navigation_timeout)

        # Scroll down to load all reviews
        while True:
            try:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                # Add additional wait time (5 seconds) to allow the content to load
                await page.waitForTimeout(6000)
                load_more_button = await page.querySelector("span[data-action='reviews:page-next']")
                if load_more_button:
                    await load_more_button.click()
                    # Add additional wait time (5 seconds) after clicking the "Load more" button
                    await page.waitForTimeout(6000)
                else:
                    break
            except:
                break

        # Find review containers
        review_elements = await page.querySelectorAll("div[data-hook='review']")
        reviews = [await page.evaluate("(element) => element.innerText", element) for element in review_elements]

        await browser.close()
        return reviews

    # amazon_url = ""
    reviews = asyncio.get_event_loop().run_until_complete(
        scrape_amazon_reviews(amazon_url))

    return reviews

# Function to check the URL platform (Yelp or Amazon)


def check_url_platform(url):
    # Regular expressions to match Yelp and Amazon URLs
    yelp_pattern = r"(https?://)?(www\.)?yelp\.com/.*"
    amazon_pattern = r"(https?://)?(www\.)?amazon\.in/.*"

    # Check if the URL matches Yelp or Amazon patterns
    if re.match(yelp_pattern, url):
        return "Yelp"
    elif re.match(amazon_pattern, url):
        return "Amazon"
    else:
        return "Unknown"


# Function to process the input URL and perform scraping, summarization, and sentiment analysis
def process_input_url(input_url):
    platform = check_url_platform(input_url)
    if platform == "Yelp":
        reviews = yelpData(input_url)
    elif platform == "Amazon":
        reviews = amzoneData(input_url)
    else:
        print("The platform of the URL is unknown.")
        return None, None, None

# combining list data
    input_text = ""
    for i in reviews:
        input_text += i


# . to generate pickle file

    # # summary generation
    # with open('tokenizer_model_forsum.pkl', 'rb') as f:
    #     tokenizer_forsum, model_forsum = pickle.load(f)

    # # Example usage:
    # def generate_summary(text, max_length=800):
    #     inputs = tokenizer_forsum.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        
    #     with torch.no_grad():
    #         summary_ids = model_forsum.generate(inputs, max_length=max_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    #     summary = tokenizer_forsum.decode(summary_ids[0], skip_special_tokens=True)
    #     return summary

    # summary = generate_summary(input_text)






    # Summarization and sentiment analysis
    pipe = pipeline("summarization", model="facebook/bart-large-cnn")
    tokenizer_forsum = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model_forsum = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    # Tokenization
    inputs = tokenizer_forsum.encode(
        "summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)

    # Model Inference
    with torch.no_grad():
        summary_ids = model_forsum.generate(inputs, num_beams=5, max_length=1000, early_stopping=True)

    # Decode the output summary
    summary = tokenizer_forsum.decode(summary_ids[0], skip_special_tokens=True)



# .to generate pickle file


    # # Sentiment analysis
    # with open('tokenizer_model.pkl', 'rb') as f:
    #     tokenizer, model = pickle.load(f)

    # # Example usage:
    
    # encoded_input = tokenizer.encode(summary, return_tensors='pt', padding=True, truncation=True)
    # output = model(encoded_input)
    # cnt = int(torch.argmax(output.logits))+1



       # Sentiment analysis
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    tokens = tokenizer.encode('summary  ', return_tensors='pt')
    result = model(tokens)
    cnt = int(torch.argmax(result.logits)) + 1

    # Classify sentiment rating
    def classify_rating(rating):
        if cnt == 1:
            return "Extreme bad"
        elif cnt <= 2:
            return "Bad"
        elif cnt == 3:
            return "Neutral"
        elif cnt == 4:
            return "Good"
        else:
            return "Very Good"

    rate = classify_rating(cnt)

    return summary, cnt, rate


def mainLoad(mainURL):
    input_url = mainURL

    # Process the input URL and get the summary, cnt, and rate
    summary, cnt, rate = process_input_url(input_url)

    return summary, cnt, rate
