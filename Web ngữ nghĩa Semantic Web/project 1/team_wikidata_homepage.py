import time

import pandas as pd
from get_google_url import get_url

teams = pd.read_csv('team_url.csv')

homepages = []
for index, row in teams.iterrows():
    keyword = f'homepage football team {row["Team"]}'
    print(keyword)
    homepage = get_url(keyword)
    while homepage is None:
        time.sleep(5)
        homepage = get_url(keyword)
        pass
    homepages.append(homepage)
    pass
teams['Homepage'] = homepages

teams.to_csv('team_url_homepage.csv', index=False)



