import pandas as pd



def work():

    teams = pd.read_csv('team_url_homepage.csv')
    #teams = teams[['Team', 'Tournament', 'URL', 'Homepage']]
    teams = teams.loc[teams['URL'].str.contains('.wikidata') ]

    players = pd.read_csv('team_3_players_url.csv')
    players = players.loc[players['URL'].str.contains('.wikidata')]

    tournaments = teams['Tournament'].unique()

    tournament_str = ''
    for tournament in tournaments:
        tournament_str += (f':{tournament.replace(" ", "_")} a schema:SportsLeague ;\n'
                           f'\tdct:title "{tournament}" ;\n'
                           f'\tschema:description "The top professional football division of the nation football league system.".\n\n')
        pass

    #print(players.columns)
    team_str = ''
    for index, row in teams.iterrows():
        team = row['Team']
        homepage = row['Homepage']
        tournament = row['Tournament']
        ps = players[players['Squad'] == team]
        url = row['URL']
        team_str += (f':{team.replace(" ", "_")} a schema:SportsTeam ;\n'
                     f'\tdct:title \"{team}\" ;\n'
                     f'\tfoaf:homepage <{homepage}> ;\n'
                     f'\tschema:memberOf :{tournament.replace(" ", "_")} ;\n'
                     f'\tschema:sameAs wdw:{url.split("/")[-1]} ; # (Wikidata entity)\n')
        if ps.empty is False:
            names = [f":{x.replace(' ', '_')}" for x in ps["Player"].tolist()]
            team_str += f'\tschema:member {", ".join(names)} ;\n'
            pass
        team_str += (f'\t:Goals "{row["Goals"]}" ; '
                     f':ShotsPerGame "{row["Shots pg"]}" ; '
                     f':yellow_cards "{row["yellow_cards"]}" ; '
                     f':red_cards "{row["red_cards"]}" ; '
                     f':PossessionPercent "{row["Possession%"]}" ; '
                     f':PassPercent "{row["Pass%"]}" ; '
                     f':AerialsWon "{row["AerialsWon"]}" ; '
                     f':Rating "{row["Rating"]}" ;\n')
        team_str += '\tdct:license <http://creativecommons.org/licenses/by-sa/4.0/> .\n\n'
        #break
        pass

    player_str = ''

    for index, row in players.iterrows():

        name = row['Player']
        team = row['Squad']
        nationality = row['Nation']
        age = row['Age']
        position = row['Pos']
        url = row['URL']

        player_str += (f':{name.replace(" ", "_")} a schema:Person, schema:SportsPerson ;\n'
                       f'\tfoaf:name "{name}" ;\n'
                       f'\tfoaf:age {age} ;\n'
                       f'\tschema:position "{position}" ;\n'
                       f'\tschema:nationality "{nationality}" ;\n'
                       f'\tschema:sameAs wdw:{url.split("/")[-1]} ; # (Wikidata entity)\n'
                       f'\tschema:memberOf :{team.replace(" ", "_")} .\n\n')

        pass

    prefix = """
@prefix : <http://example.org/football/> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix schema: <http://schema.org/> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix wdw: <https://www.wikidata.org/wiki/> .
@prefix wd: <http://www.wikidata.org/entity/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""
    postfix = """
schema:SportsTeam a rdfs:Class ;
    rdfs:label "Sports Team" .

schema:SportsPerson a rdfs:Class ;
    rdfs:label "Sports Person" .


schema:SportsLeague a rdfs:Class ;
    rdfs:label "Sports League" .    
"""
    #print(team_str)
    #print(prefic)
    #print(player_str)
    #print(tournament_str)
    return f'{prefix}\n{postfix}\n{player_str}{tournament_str}{team_str}'

if __name__ == '__main__':
    result = work()
    with open('rdf.ttl', 'w', encoding='utf-8') as f:
        f.write(result)
        pass
    pass