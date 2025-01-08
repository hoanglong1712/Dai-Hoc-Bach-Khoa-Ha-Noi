import pandas as pd
from get_google_url import get_url
teams = pd.read_csv('teams.csv', encoding='latin-1')
players = pd.read_csv('players.csv', encoding='latin-1')
result = players.iloc[:, [1, 2, 3, 4, 6 ]].groupby('Squad').head(3).sort_values(by='Squad')
result = result[result['Squad'].isin(teams['Team'])]



urls = []
for index, row in result.iterrows():
    keyword = f"wikidata id football player {row['Player']}"
    print(keyword)
    url = get_url(keyword)
    while url is None:
        url = get_url(keyword)
        pass
    urls.append(url)
    pass

result['URL'] = urls

result.to_csv('team_3_players_url.csv', index=False)
