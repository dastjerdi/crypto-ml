import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# scraping
data = []
for number in range(1,91):
    req = requests.get("https://cryptocurrencynews.com/category/daily-news/page/" + str(number) + '/')
    page = req.text
    soup = BeautifulSoup(page, 'html.parser')
    for article in soup.find_all("a", {"rel": 'bookmark'}):
        headline = article.text[1:]
        date = article.parent.parent.find("i", {"class": ['fa-clock-o']}).parent.findNext('a').text
        data += [{'headline': headline, 'date': date}]

# make date time 
def get_date(string): 
    if string[:3] in ['Jan', 'Feb', 'Mar', 'Apr']:
        string += ' 2018'
    else: 
        string += ' 2017'
    return datetime.strptime( string , '%b %d %Y')

# put into dataframe, format properly, and export csv
data = pd.DataFrame(data)
data['date'] = data['date'].apply(get_date)
data.to_csv('data.csv')
