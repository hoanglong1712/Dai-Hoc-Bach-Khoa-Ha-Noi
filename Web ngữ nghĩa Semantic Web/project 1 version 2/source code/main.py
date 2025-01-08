import pandas as pd

teams = pd.read_csv('teams.csv', encoding='latin-1')
players = pd.read_csv('players.csv', encoding='latin-1')

tournaments = teams['Tournament'].unique()

'''
df = pd.DataFrame({'League': tournaments})
df.to_csv('league.csv', index=False)
'''

root = {}
for tournament in tournaments:
    team_list = teams[teams['Tournament'] == tournament]
    node_1 = {}


    root[tournament] = node_1
    pass