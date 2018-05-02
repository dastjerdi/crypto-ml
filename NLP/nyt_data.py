
import nytarchive.py

# run multiple times as API fails ocasionally
# get all articles in date range of project
api = ArchiveAPI('93af2608c5b24286a5b53fc67232b554')
articles = []
exclusions = [(2015, 1), (2015, 2), (2015, 3), (2015, 4), (2015, 5), (2015, 6), (2015, 7), (2018, 4), (2018, 5), (2018, 6), (2018, 7), (2018, 8), (2018, 9), (2018, 10), (2018, 11), (2018, 12)]
for year in [2015,2016,2017,2018]:
    for month in range(1,13):
        if (year,month) not in exclusions:
            print((year, month))
            articles += api.query(year, month)['response']['docs']
        

# filter by desk
desks = []
desired_desks = ['Business','Foreign','National']
filtered = []
for article in articles:
    if 'news_desk' in article.keys():
        if article['news_desk'] in desired_desks: 
            filtered += [article]
# filtered by section
sections_filtered = []
desired_sections = ['U.S.','World', 'Business Day', 'Technology','Real Estate', 'Politics', 'Economy', 'Europe', 'Asia Pacific', 'Middle East', 'Americas', 'Africa', 'Job Market', 'Australia', 'Canada']
for article in filtered:
    if 'section_name' in article.keys():
        if article['section_name'] in desired_sections: 
            sections_filtered += [article]

# filter by type 
news_filtered = []
desired_type = ['News']
for article in sections_filtered:
    if 'type_of_material' in article.keys():
        if article['type_of_material'] in desired_type: 
            news_filtered += [article]

# grab relevent information from filtered articles and store in dataframe
data = []
for article in news_filtered:
    entry = {}
    entry['headline'] = article['headline']['main']
    entry['date'] = article['pub_date']
    entry['keywords'] = make_key_string(get_keys(article))
    data += [entry]

data = pd.DataFrame(data)

# format date properly
data['date'] = pd.to_datetime(data['date'].apply(lambda x: x[:10]))

# make boolean marker for being about cryptocurrency
cyrpto_phrase = ['cryptocurrency', 'Bitcoin', 'Litecoin', 'Ripple','Stellar', 'Ethereum',  'Neo', 'IOTA', 'EOS', 'Bitcoin Cash', 'Ada']
data['crypto_phrase'] = data['keywords'].apply(lambda x: 1 if any(word in x for word in cyrpto_phrase) else 0)

# export data
data.to_csv('data.csv')