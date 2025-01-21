import time

import pandas as pd
from get_google_url import get_url

teams = pd.read_csv('teams.csv', encoding='latin-1')

urls = []
for index, row in teams.iterrows():
    keyword = f'wikidata id football team {row["Team"]}'
    print(keyword)
    url = get_url(keyword)
    while url is None:
        time.sleep(5)
        url = get_url(keyword)
        pass
    urls.append(url)
    pass
teams['URL'] = urls

teams.to_csv('team_url.csv', index=False)



