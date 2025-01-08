import pandas as pd


def prefix_string():
    prefix = """
@prefix : <http://www.semanticweb.org/ontologies/2025/football/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xml: <http://www.w3.org/XML/1998/namespace> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@base <http://www.semanticweb.org/ontologies/2025/football/> .

<http://www.semanticweb.org/ontologies/2025/football> rdf:type owl:Ontology .

"""
    return prefix


def object_gen(name, domain, range):
    result = f"""
###  http://www.semanticweb.org/ontologies/2025/football#{name}
:{name} rdf:type owl:ObjectProperty ;
       rdfs:subPropertyOf owl:topObjectProperty ;
       rdfs:domain :{domain} ;
       rdfs:range :{range} .
              
"""
    return result


def objects_gen(object_data: []):
    result = """
#################################################################
#    Object Properties
#################################################################

"""
    for name, domain, range in object_data:
        result += object_gen(name, domain, range)
        pass
    return result


def data_gen(name, domain, range):
    if isinstance(domain, list):
        tmp = ', '.join(f':{x}' for x in domain)
        domain = tmp
        pass
    else:
        domain = f':{domain}'
        pass
    result = f"""
###  http://www.semanticweb.org/ontologies/2025/football#{name}
:{name} rdf:type owl:DatatypeProperty ;
            rdfs:subPropertyOf owl:topDataProperty ;
            rdfs:domain {domain} ;
            rdfs:range {range} .

"""
    return result


def datas_gen(data_prop: []):
    result = """
#################################################################
#    Data properties
#################################################################

"""
    for name, domain, range in data_prop:
        result += data_gen(name, domain, range)
        pass
    return result


def class_gen(class_list: []):
    result = """

#################################################################
#    Classes
#################################################################

"""
    for item in class_list:
        result += f"""
###  http://www.semanticweb.org/ontologies/2025/football/{item}
:{item} rdf:type owl:Class .

"""
        pass
    return result


def manual_gen():
    prefix = prefix_string()
    object_properties = objects_gen([
        ('hires', 'Team', 'Player'),
        ('playsFor', 'Player', 'Team'),
        ('competesIn', 'Team', 'League')
    ])

    data_properties = datas_gen([
        ('name', 'Player', 'xsd:string'),
        ('age', 'Player', 'xsd:integer'),
        ('position', 'Player', 'xsd:string'),
        ('nationality', 'Player', 'xsd:string'),
        ('team_name', 'Team', 'xsd:string'),
        ('homepage', 'Team', 'xsd:anyURI'),
        ('goal', 'Team', 'xsd:integer'),
        ('shot_percentage', 'Team', 'xsd:decimal'),
        ('yellow_card', 'Team', 'xsd:integer'),
        ('red_card', 'Team', 'xsd:integer'),
        ('procession_percentage', 'Team', 'xsd:decimal'),
        ('pass_percentage', 'Team', 'xsd:decimal'),
        ('aerials_won', 'Team', 'xsd:decimal'),
        ('rating', 'Team', 'xsd:decimal'),
        ('league_name', 'League', 'xsd:string'),
        ('description', 'League', 'xsd:string'),
        ('link', 'WikidataLink', 'xsd:anyURI')
    ])

    classes = class_gen([
        'League', 'Player', 'Team', 'WikidataLink'
    ])

    return f'{prefix}{object_properties}{data_properties}{classes}'


def individual_league_gen(teams: pd.DataFrame):
    tournaments = teams['Tournament'].unique()
    result = ""
    for league in tournaments:
        name = league.replace(' ', '_')
        result += f"""
###  http://www.semanticweb.org/ontologies/2025/football#{name}
:{name} rdf:type owl:NamedIndividual ,
                     :League ;
            :league_name "{league}" .
"""
        pass
    return result


def individual_player_gen(players: pd.DataFrame):
    result = ""
    for index, row in players.iterrows():
        name = row['Player']
        team = row['Squad']
        nationality = row['Nation']
        age = row['Age']
        position = row['Pos']
        url = row['URL']

        id = name.replace(' ', '_')
        wiki = f'Wikidata_{id}'
        team = team.replace(' ', '_')

        result += f"""        
###  http://www.semanticweb.org/ontologies/2025/football#{id}
:{id} rdf:type owl:NamedIndividual ,
                       :Player ;
              owl:sameAs :{wiki} ;
              :playsFor :{team} ;
              :age {age} ;
              :name "{name}" ;
              :nationality "{nationality}" ;
              :position "{position}" .

"""
        result += f"""
###  http://www.semanticweb.org/ontologies/2025/football#{wiki}
:{wiki} rdf:type owl:NamedIndividual ,
                                :WikidataLink ;
                       :link "{url}"^^xsd:anyURI .

"""
        pass

    return result


def individual_team_gen(teams: pd.DataFrame, players: pd.DataFrame):
    result = ''

    for index, row in teams.iterrows():
        team = row['Team']
        homepage = row['Homepage']
        tournament = row['Tournament']
        ps = players[players['Squad'] == team]
        url = row['URL']

        id = team.replace(' ', '_')
        tournament = f":{tournament.replace(' ', '_')}"
        wiki = f'Wikidata_{id}'

        result += f"""
###  http://www.semanticweb.org/ontologies/2025/football#{id}
:{id} rdf:type owl:NamedIndividual ,
                    :Team ;"""
        if ps.empty is False:
            player_names = [f":{x.replace(' ', '_')}" for x in ps["Player"].tolist()]
            player_str = ', '.join(player_names)
            result += f"""
         :hires {player_str} ;"""
            pass

        result += f"""
           :homepage "{homepage}"^^xsd:anyURI ;
           :goal {row['Goals']} ;
           :shot_percentage {row['Shots pg']} ;
           :yellow_card {row['yellow_cards']} ;
           :red_card {row['red_cards']} ;
           :procession_percentage {row['Possession%']} ;
           :pass_percentage {row['Pass%']} ;
           :aerials_won {row['AerialsWon']} ;
           :rating {row['Rating']} ;
           :competesIn {tournament} ;
           owl:sameAs :{wiki} ;
           :team_name "{team}" .

"""
        result += f"""
###  http://www.semanticweb.org/ontologies/2025/football#{wiki}
:{wiki} rdf:type owl:NamedIndividual ,
                    :WikidataLink ;
           :link "{url}"^^xsd:anyURI .

        """
        
        pass

    return result

def manual_axioms():
    return """
#################################################################
#    General axioms
#################################################################

[ rdf:type owl:AllDisjointClasses ;
  owl:members ( :League
                :Player
                :Team
              )
] .
"""

def owl_gen(teams: pd.DataFrame, players: pd.DataFrame):
    manual_str = manual_gen()
    individual_str = individual_league_gen(teams)
    player_str = individual_player_gen(players)
    team_str = individual_team_gen(teams, players)
    axioms_str = manual_axioms()
    return f'{manual_str}{individual_str}{player_str}{team_str}{axioms_str}'
    #return f'{manual_str}{individual_str}{player_str}{team_str}'


if __name__ == '__main__':
    teams = pd.read_csv('team_url_homepage.csv')
    # teams = teams[['Team', 'Tournament', 'URL', 'Homepage']]
    teams = teams.loc[teams['URL'].str.contains('.wikidata')]

    players = pd.read_csv('team_3_players_url.csv')
    players = players.loc[players['URL'].str.contains('.wikidata')]

    result = owl_gen(teams, players)


    with open('footbal_test.ttl', 'w', encoding='utf-8') as f:
        f.write(result)
        pass

    pass
