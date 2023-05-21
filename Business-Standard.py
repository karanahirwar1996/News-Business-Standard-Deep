from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import json
import requests
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from multiprocessing import Pool
from nltk.sentiment import SentimentIntensityAnalyzer
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound'] * 100

def calculate_positivity_score(positive_count, negative_count):
    total_count = positive_count + negative_count
    positivity_score = (positive_count - negative_count) / total_count * 100 if total_count != 0 else 0
    return round(positivity_score, 2)

def predict_stock_sentiment(sentence):
    words = word_tokenize(sentence.lower())
    
    positive_keywords = list(pd.read_csv("./positive-words.csv")['words'])
    negative_keywords =list(pd.read_csv("./negative-words.csv")['words'])
    
    positive_count = 0
    negative_count = 0
    lemmatizer = WordNetLemmatizer()
    pos_list=[]
    neg_list=[]
    for p in positive_keywords:
            pos_list.append(lemmatizer.lemmatize(p, pos='v'))
    for n in negative_keywords:
            neg_list.append(lemmatizer.lemmatize(n, pos='v'))
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        if lemma in pos_list:
            positive_count += 1
        elif lemma in neg_list:
            negative_count += 1
    
    positivity_score = calculate_positivity_score(positive_count, negative_count)
    
    return positivity_score

def extract_date(timestamp):
    date_str = timestamp.split("T")[0]
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    formatted_date = date_obj.strftime("%d/%m/%Y")
    return formatted_date

def extract_time(timestamp):
    time_str = timestamp.split("T")[1].split("+")[0]
    time_obj = datetime.strptime(time_str, "%H:%M:%S")
    formatted_time = time_obj.strftime("%I:%M %p")
    return formatted_time

def scrape_datetime(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.content, features="html.parser")
        script_elements = soup.find_all("script", {"type": "application/ld+json"})
        if len(script_elements)!=6:
            return None
        d = json.loads(script_elements[4].string)
        datetime = d.get("datePublished")
        return datetime
    except:
        return None

        
def scrape_article_info(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.content, features="html.parser")
        script_elements = soup.find_all("script", {"type": "application/ld+json"})
        if len(script_elements) != 6:
            return None
        d = json.loads(script_elements[4].string)
        article_body = d.get("articleBody")
        return article_body
    except:
        return None

def scrape_page(page_num):
    url = f'https://www.business-standard.com/amp/markets-news/page-{page_num}'
    res = requests.get(url)
    soup = BeautifulSoup(res.text, features="html.parser")
    script_elements = soup.find_all("script", {"type": "application/ld+json"})
    if len(script_elements) != 6:
        return None
    json_data = json.loads(script_elements[5].string)
    data = [(item["name"], item["url"]) for item in json_data["itemListElement"]]
    return pd.DataFrame(data, columns=["Headline", "URL"])
def positive_news():
    current_date = datetime.now().date()
    previous_date = current_date - timedelta(days=1)

    dfs = []
    pool = Pool(processes=4)  # Adjust the number of processes based on your system
    results = pool.map(scrape_page, range(1,5))
    dfs = [df for df in results if df is not None]

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(ignore_index=True)
    print(df.shape)

    results = pool.map(scrape_datetime, df['URL'])
    for i, result in enumerate(results):
        if result is not None:
            try:
                df.loc[i, 'Date'] = extract_date(result)
            except (KeyError, TypeError):
                df.loc[i, 'Date'] = None
            try:
                df.loc[i, 'Time'] = extract_time(result)
            except (KeyError, TypeError):
                df.loc[i, 'Time'] = None
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Normal Score'] = df['Headline'].apply(analyze_sentiment)

    filtered_df = df.loc[(df['Date'] <= pd.to_datetime(current_date)) & (df['Date'] >= pd.to_datetime(previous_date))].reset_index(drop=True)

    dip = []
    results = pool.map(scrape_article_info, filtered_df['URL'])
    for result in results:
        if result is not None:
            dip.append(predict_stock_sentiment(result))

    filtered_df["Deep Score"] = dip
    final = filtered_df.loc[(filtered_df['Deep Score'] >45) & (filtered_df['Normal Score'] >= 0)].reset_index(drop=True)

    sender_email = "karan.ahirwar1996@gmail.com"
    receiver_email = list(pd.read_csv("./emaillist.csv")['mail'])
    password = "uccrgtqdnusrpmnk"
    table_html = final.to_html(index=False)

    # Create the email message
    msg = MIMEMultipart()

    positive_news_count = len(final)
    msg["Subject"] = f"✨ Daily Positive News Digest-Business Standard- {current_date} ({positive_news_count} uplifting articles out of {len(filtered_df)})✨"

    msg["From"] = sender_email
    msg["To"] = ", ".join(receiver_email)

    # HTML Template for the email content
    html_template = """
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
            }}
            h1 {{
                color: #336699;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <h1>Positive News - {current_date}</h1>
        {table}
    </body>
    </html>
    """

    # Set the message content with HTML template and table
    email_content = html_template.format(current_date=current_date, table=table_html)
    message = MIMEText(email_content, 'html')
    msg.attach(message)

    # Send the email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())

    return final

positive_news()
