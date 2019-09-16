from tqdm import tqdm
from pytorch_transformers import (CONFIG_NAME, WEIGHTS_NAME)
import numpy as np
import random
import torch
import os

relations = [
    'AtLocation', 'CapableOf', 'Causes', 'CausesDesire',
    'CreatedBy', 'DefinedAs', 'DesireOf', 'Desires', 'HasA',
    'HasFirstSubevent', 'HasLastSubevent', 'HasPainCharacter',
    'HasPainIntensity', 'HasPrerequisite', 'HasProperty',
    'HasSubevent', 'InheritsFrom', 'InstanceOf', 'IsA',
    'LocatedNear', 'LocationOfAction', 'MadeOf', 'MotivatedByGoal',
    'NotCapableOf', 'NotDesires', 'NotHasA', 'NotHasProperty',
    'NotIsA', 'NotMadeOf', 'PartOf', 'ReceivesAction', 'RelatedTo',
    'SymbolOf', 'UsedFor'
]

split_into_words = {
    'AtLocation': "at location",
    'CapableOf': "capable of",
    'Causes': "causes",
    'CausesDesire': "causes desire",
    'CreatedBy': "created by",
    'DefinedAs': "defined as",
    'DesireOf': "desire of",
    'Desires': "desires",
    'HasA': "has a",
    'HasFirstSubevent': "has first subevent",
    'HasLastSubevent': "has last subevent",
    'HasPainCharacter': "has pain character",
    'HasPainIntensity': "has pain intensity",
    'HasPrerequisite': "has prequisite",
    # actually it is "has prerequisite, but models were trained on it ..."
    'HasProperty': "has property",
    'HasSubevent': "has subevent",
    'InheritsFrom': "inherits from",
    'InstanceOf': 'instance of',
    'IsA': "is a",
    'LocatedNear': "located near",
    'LocationOfAction': "location of action",
    'MadeOf': "made of",
    'MotivatedByGoal': "motivated by goal",
    'NotCapableOf': "not capable of",
    'NotDesires': "not desires",
    'NotHasA': "not has a",
    'NotHasProperty': "not has property",
    'NotIsA': "not is a",
    'NotMadeOf': "not made of",
    'PartOf': "part of",
    'ReceivesAction': "receives action",
    'RelatedTo': "related to",
    'SymbolOf': "symbol of",
    'UsedFor': "used for",
    "/olympics/olympic_participating_country/athletes./olympics/olympic_athlete_affiliation/olympics": "participate in olympics",
    "/film/film/music": "has music composed by",
    "/award/award_category/category_of": "is an award category of",
    "/music/performance_role/regular_performances./music/group_membership/role": None,
    "/location/country/capital": "capital",
    "/government/legislative_session/members./government/government_position_held/legislative_sessions": "members also attend",
    "/location/administrative_division/first_level_division_of": "is one of the first level administrtive division of",
    "/tv/tv_program/program_creator": "created by",
    "/base/americancomedy/celebrity_impressionist/celebrities_impersonated": "has impersonated",
    "/government/legislative_session/members./government/government_position_held/district_represented": "has members representing",
    "/education/educational_institution/students_graduates./education/education/student": "has graduated student",
    "/tv/tv_program/languages": "is spoken in",
    "/people/person/spouse_s./people/marriage/location_of_ceremony": "location of marriage ceremony",
    "/award/award_ceremony/awards_presented./award/award_honor/award_winner": "awarded to",
    "/government/politician/government_positions_held./government/government_position_held/basic_title": "has title in government",
    "/location/location/time_zones": "in time zone",
    "/film/person_or_entity_appearing_in_film/films./film/personal_film_appearance/type_of_appearance": "type of appearance in film",
    "/military/military_conflict/combatants./military/military_combatant_group/combatants": "has combatants",
    "/film/film_set_designer/film_sets_designed": "designed film set",
    "/medicine/symptom/symptom_of": "is a symptom of",
    "/language/human_language/countries_spoken_in": "is spoken in",
    "/base/popstra/celebrity/dated./base/popstra/dated/participant": "dated with",
    "/people/person/place_of_birth": "is born in",
    "/organization/organization/headquarters./location/mailing_address/country": "is located in",
    "/education/educational_institution/students_graduates./education/education/major_field_of_study": "major field of study",
    "/music/group_member/membership./music/group_membership/group": "is a member of",
    "/music/performance_role/regular_performances./music/group_membership/group": "played by",
    "/base/x2010fifaworldcupsouthafrica/world_cup_squad/current_world_cup_squad./base/x2010fifaworldcupsouthafrica/current_world_cup_squad/current_club": "from club",
    "/location/statistical_region/religions./location/religion_percentage/religion": "has main religion",
    "/education/educational_institution_campus/educational_institution": "educational institution",
    "/award/award_winning_work/awards_won./award/award_honor/honored_for": "won same awards with",
    "/film/film/produced_by": "is produced by",
    "/award/award_winning_work/awards_won./award/award_honor/award_winner": "earn awards to",
    "/people/profession/specialization_of": "is a specialization of",
    "/film/film/edited_by": "is edited by",
    "/film/film/release_date_s./film/film_regional_release_date/film_release_distribution_medium": "distributed by",
    "/location/statistical_region/gdp_real./measurement_unit/adjusted_money_value/adjustment_currency": "gdp measured by adjustment currency",
    "/sports/sports_position/players./american_football/football_historical_roster_position/position_s": "both football positions",
    "/location/location/partially_contains": "partially contains",
    "/film/film/story_by": "story by",
    "/base/popstra/location/vacationers./base/popstra/vacation_choice/vacationer": "has vacationer",
    "/base/localfood/seasonal_month/produce_available./base/localfood/produce_availability/seasonal_months": None,
    "/baseball/baseball_team/team_stats./baseball/baseball_team_stats/season": "participate in baseball season",
    "/music/instrument/instrumentalists": "played by",
    "/government/governmental_body/members./government/government_position_held/legislative_sessions": "take part in",
    "/organization/organization/headquarters./location/mailing_address/citytown": "headquarter located in",
    "/location/statistical_region/rent50_2./measurement_unit/dated_money_value/currency": None,
    "/sports/sports_position/players./sports/sports_team_roster/team": None,
    "/influence/influence_node/influenced_by": "is influenced by",
    "/food/food/nutrients./food/nutrition_fact/nutrient": "has nutrient",
    "/film/film/film_festivals": "is shown in festival",
    "/music/artist/track_contributions./music/track_contribution/role": "contribute to track as a role of",
    "/award/award_category/disciplines_or_subjects": "is an award of the discipline of",
    "/film/film_distributor/films_distributed./film/film_film_distributor_relationship/film": "distributed film",
    "/medicine/disease/notable_people_with_this_condition": "notable people with this condition",
    "/tv/tv_program/country_of_origin": "originated in",
    "/organization/organization/headquarters./location/mailing_address/state_province_region": "located in",
    "/sports/sports_position/players./sports/sports_team_roster/position": None,
    "/location/hud_foreclosure_area/estimated_number_of_mortgages./measurement_unit/dated_integer/source": "number of mortgages measured by",
    "/film/film/other_crew./film/film_crew_gig/film_crew_role": "take the role of",
    "/people/deceased_person/place_of_death": "place of death",
    "/base/marchmadness/ncaa_basketball_tournament/seeds./base/marchmadness/ncaa_tournament_seed/team": "has seed team",
    "/user/jg/default_domain/olympic_games/sports": "includes sports",
    "/location/hud_county_place/place": None,
    "/film/film/distributors./film/film_film_distributor_relationship/region": "distributed in region",
    "/music/genre/artists": "played by artists",
    "/film/film/costume_design_by": "costume designed by",
    "/people/ethnicity/geographic_distribution": "geographically distributed in",
    "/organization/endowed_organization/endowment./measurement_unit/dated_money_value/currency": "endowment measured by currency",
    "/people/person/gender": "of gender",
    "/base/eating/practicer_of_diet/diet": "practicer of diet",
    "/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_language": "serve with language",
    "/film/film/release_date_s./film/film_regional_release_date/film_regional_debut_venue": "premiere at",
    "/music/artist/origin": "comes from",
    "/influence/influence_node/peers./influence/peer_relationship/peers": "is in peer relationship with",
    "/sports/sports_league_draft/picks./sports/sports_league_draft_pick/school": "has attendee",
    "/award/award_category/winners./award/award_honor/award_winner": "awarded to",
    "/olympics/olympic_games/participating_countries": "participating countries",
    "/film/film/production_companies": "produced by companies",
    "/government/political_party/politicians_in_this_party./government/political_party_tenure/politician": "has member",
    "/education/university/domestic_tuition./measurement_unit/dated_money_value/currency": "domestic tuition measured by currency",
    "/film/film/other_crew./film/film_crew_gig/crewmember": "has pther crew member",
    "/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/contact_category": "category of contact",
    "/award/award_nominee/award_nominations./award/award_nomination/award_nominee": "has same award nomination as ",
    "/film/film/dubbing_performances./film/dubbing_performance/actor": "dubbed by",
    "/location/country/form_of_government": "form of government",
    "/base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency": "net worth measured by currency",
    "/film/film/distributors./film/film_film_distributor_relationship/film_distribution_medium": "film distributed in the form of",
    "/base/biblioness/bibs_location/state": None,
    "/sports/sports_team/roster./american_football/football_historical_roster_position/position_s": "has position",
    "/location/capital_of_administrative_division/capital_of./location/administrative_division_capital_relationship/administrative_division": "capital of",
    "/music/record_label/artist": "owns artist",
    "/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics": "won olympic medals in",
    "/business/business_operation/assets./measurement_unit/dated_money_value/currency": "assets measured by currency",
    "/film/film/written_by": "written by",
    "/film/film/release_date_s./film/film_regional_release_date/film_release_region": "released in",
    "/tv/tv_writer/tv_programs./tv/tv_program_writer_relationship/tv_program": "writer tv program",
    "/film/film/film_production_design_by": "film production designed by",
    "/travel/travel_destination/how_to_get_here./travel/transportation/mode_of_transportation": "transportation to get there",
    "/sports/sports_team/roster./baseball/baseball_roster_position/position": "has position",
    "/award/award_nominee/award_nominations./award/award_nomination/award": "is nominated for",
    "/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor": "has an actor",
    "/base/popstra/celebrity/canoodled./base/popstra/canoodled/participant": "canoodled with",
    "/common/topic/webpage./common/webpage/category": "category of webpage",
    "/location/country/second_level_divisions": "is one of the second level administrtive division of",
    "/military/military_combatant/military_conflicts./military/military_combatant_group/combatants": "has military conflicts with",
    "/award/award_category/winners./award/award_honor/ceremony": "is awarded in ceremony",
    "/celebrities/celebrity/celebrity_friends./celebrities/friendship/friend": "is a friend of",
    "/tv/non_character_role/tv_regular_personal_appearances./tv/tv_regular_personal_appearance/person": "role taken by",
    "/sports/sports_league/teams./sports/sports_league_participation/team": "has participants",
    "/olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/olympics": "is a sport in",
    "/user/tsegaran/random/taxonomy_subject/entry./user/tsegaran/random/taxonomy_entry/taxonomy": None,
    "/base/petbreeds/city_with_dogs/top_breeds./base/petbreeds/dog_city_relationship/dog_breed": "has top dog breed",
    "/education/educational_degree/people_with_this_degree./education/education/student": "people with this degree",
    "/business/business_operation/industry": "main business",
    "/education/university/fraternities_and_sororities": "has fraternities and sororities",
    "/people/person/places_lived./people/place_lived/location": "lived in",
    "/dataworld/gardening_hint/split_to": None,
    "/organization/role/leaders./organization/leadership/organization": "is a leader position in",
    "/music/performance_role/guest_performances./music/recording_contribution/performance_role": "is a performance role in",
    "/base/aareas/schema/administrative_area/capital": "has capital",
    "/music/genre/parent_genre": "is genre of",
    "/people/person/spouse_s./people/marriage/type_of_union": "has type of union with spouse",
    "/soccer/football_player/current_team./sports/sports_team_roster/team": "is a soccer player in",
    "/sports/sports_team_location/teams": "has team",
    "/people/cause_of_death/people": "causes death to",
    "/location/location/adjoin_s./location/adjoining_relationship/adjoins": "adjoins",
    "/business/business_operation/operating_income./measurement_unit/dated_money_value/currency": "has income type of",
    "/government/government_office_category/officeholders./government/government_position_held/jurisdiction_of_office": "is government position in",
    "/film/film/featured_film_locations": "has filming locations in",
    "/olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/country": "has atheletes from",
    "/base/aareas/schema/administrative_area/administrative_area_type": "has administrative type of",
    "/film/film/personal_appearances./film/personal_film_appearance/person": "has appearances of",
    "/sports/sports_team/colors": "has color in",
    "/base/biblioness/bibs_location/country": "locates in",
    "/film/film/estimated_budget./measurement_unit/dated_money_value/currency": "has value in the currency of",
    "/education/educational_institution/colors": "has color in",
    "/award/hall_of_fame/inductees./award/hall_of_fame_induction/inductee": "has inductee",
    "/government/politician/government_positions_held./government/government_position_held/legislative_sessions": "has politic position in",
    "/film/actor/film./film/performance/film": "acts in",
    "/award/award_winner/awards_won./award/award_honor/award_winner": "wins the same award with",
    "/education/university/local_tuition./measurement_unit/dated_money_value/currency": "has tuition measured by",
    "/base/popstra/celebrity/breakup./base/popstra/breakup/participant": "breaks up with",
    "/time/event/instance_of_recurring_event": "is an instance of",
    "/people/person/profession": "has profession in",
    "/education/field_of_study/students_majoring./education/education/student": "is a major of",
    "/user/ktrueman/default_domain/international_organization/member_states": "has member",
    "/music/instrument/family": "belongs to the musical instrument family of",
    "/ice_hockey/hockey_team/current_roster./sports/sports_team_roster/position": "has position type of",
    "/people/person/languages": "can speak",
    "/base/locations/continents/countries_within": "contains the country",
    "/music/artist/contribution./music/recording_contribution/performance_role": "contributes to the musical performance role of",
    "/government/politician/government_positions_held./government/government_position_held/jurisdiction_of_office": "has political power to govern",
    "/people/marriage_union_type/unions_of_this_type./people/marriage/location_of_ceremony": "is held in",
    "/education/educational_institution/campuses": "is an institution of",
    "/organization/organization/child./organization/organization_relationship/child": "has a children organization",
    "/sports/sports_team/sport": "is a team in",
    "/location/statistical_region/places_exported_to./location/imports_and_exports/exported_to": "exports to",
    "/award/award_nominated_work/award_nominations./award/award_nomination/nominated_for": "win awards at the same time with",
    "/soccer/football_team/current_roster./soccer/football_roster_position/position": "has position in",
    "/film/actor/film./film/performance/special_performance_type": "performs in the type of",
    "/award/award_ceremony/awards_presented./award/award_honor/honored_for": "gives award honor to",
    "/organization/organization_member/member_of./organization/organization_membership/organization": "is a member of",
    "/sports/sport/pro_athletes./sports/pro_sports_played/athlete": "has previous athlete",
    "/sports/pro_athlete/teams./sports/sports_team_roster/team": "previously played in",
    "/time/event/locations": "located in",
    "/tv/tv_personality/tv_regular_appearances./tv/tv_regular_personal_appearance/program": "appears in the program of",
    "/business/business_operation/revenue./measurement_unit/dated_money_value/currency": "operates business in the currency of",
    "/people/person/employment_history./business/employment_tenure/company": "worked in",
    "/location/statistical_region/gdp_nominal./measurement_unit/dated_money_value/currency": "measures gdp by the currency of",
    "/soccer/football_team/current_roster./sports/sports_team_roster/position": "plays in the position of",
    "/film/film/language": "has film language",
    "/location/statistical_region/gdp_nominal_per_capita./measurement_unit/dated_money_value/currency": "measures per capita gdp by the currency of",
    "/olympics/olympic_games/sports": "had sports",
    "/location/statistical_region/gni_per_capita_in_ppp_dollars./measurement_unit/dated_money_value/currency": "measures per capital ppp gdp by the currency of",
    "/education/field_of_study/students_majoring./education/education/major_field_of_study": "is a major as well as",
    "/medicine/disease/risk_factors": "has risk factor of",
    "/film/film/cinematography": "is cinematographed by",
    "/tv/tv_program/genre": "is genre of",
    "/film/film/genre": "is genre of",
    "/film/film/country": "is produced from",
    "/award/award_winning_work/awards_won./award/award_honor/award": "receives the award",
    "/film/film/executive_produced_by": "has executive producer",
    "/people/person/sibling_s./people/sibling_relationship/sibling": "is a sibling of",
    "/film/actor/dubbing_performances./film/dubbing_performance/language": "dubs by the language of",
    "/tv/tv_producer/programs_produced./tv/tv_producer_term/producer_type": "is a",
    "/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/medal": "won",
    "/sports/professional_sports_team/draft_picks./sports/sports_league_draft_pick/draft": "participated in",
    "/film/film/runtime./film/film_cut/film_release_region": "is released in",
    "/film/film/prequel": "has prequel",
    "/sports/sports_team/roster./american_football/football_roster_position/position": "has football position in",
    "/education/university/international_tuition./measurement_unit/dated_money_value/currency": "measures its tuition by the currency of",
    "/music/performance_role/track_performances./music/track_contribution/role": "plays the role of",
    "/location/country/official_language": "has official language",
    "/tv/tv_program/tv_producer./tv/tv_producer_term/producer_type": "has the producer type of",
    "/music/group_member/membership./music/group_membership/role": "plays the role in",
    "/tv/tv_network/programs./tv/tv_network_duration/program": "has tv program",
    "/people/deceased_person/place_of_burial": "is buried in",
    "/tv/tv_producer/programs_produced./tv/tv_producer_term/program": "produces",
    "/base/aareas/schema/administrative_area/administrative_parent": "is a part of",
    "/base/saturdaynightlive/snl_cast_member/seasons./base/saturdaynightlive/snl_season_tenure/cast_members": "is a cast member in the time with",
    "/broadcast/content/artist": "has the artist",
    "/award/award_category/nominees./award/award_nomination/nominated_for": "has nominees",
    "/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_location": "has service location in",
    "/film/film/film_format": "is a format of",
    "/education/educational_institution/school_type": "is a type of",
    "/education/educational_degree/people_with_this_degree./education/education/major_field_of_study": "has a field in",
    "/olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/medal": "has medal honor of",
    "/people/ethnicity/people": "is the ethnicity of",
    "/award/award_nominee/award_nominations./award/award_nomination/nominated_for": "win award by",
    "/location/location/contains": "contains",
    "/people/ethnicity/languages_spoken": "speaks",
    "/base/popstra/celebrity/friendship./base/popstra/friendship/participant": "is a friend of",
    "/location/hud_county_place/county": "is a county in",
    "/sports/sports_team/roster./basketball/basketball_roster_position/position": "has basketball position in",
    "/sports/professional_sports_team/draft_picks./sports/sports_league_draft_pick/school": "picks a player from",
    "/film/special_film_performance_type/film_performance_type./film/performance/film": "is the type of",
    "/celebrities/celebrity/sexual_relationships./celebrities/romantic_relationship/celebrity": "has a romantic relationship with",
    "/location/us_county/county_seat": "has a county seat of",
    "/organization/non_profit_organization/registered_with./organization/non_profit_registration/registering_agency": "is registered with",
    "/base/culturalevent/event/entity_involved": "involves",
    "/american_football/football_team/current_roster./sports/sports_team_roster/position": "has football position in",
    "/people/person/nationality": "is a person of",
    "/travel/travel_destination/climate./travel/travel_destination_monthly_climate/month": "can be traveled in",
    "/film/film_subject/films": "is the subject of",
    "/award/ranked_item/appears_in_ranked_lists./award/ranking/list": "gets rank in the list of",
    "/people/person/spouse_s./people/marriage/spouse": "has a marriage with",
    "/user/alexander/philosophy/philosopher/interests": "has interests in",
    "/location/administrative_division/country": "is a part of",
    "/business/job_title/people_with_this_title./business/employment_tenure/company": "is a position in",
    "/film/film/film_art_direction_by": "has art director",
    "/media_common/netflix_genre/titles": "is the genre of",
    "/organization/organization_founder/organizations_founded": "founded",
    "/organization/organization/place_founded": "is founded in",
    "/people/person/religion": "has the belief of",
    "/education/educational_degree/people_with_this_degree./education/education/institution": "is a degree in",
    "/film/director/film": "directs"
}

def load_comet_dataset(dataset_path=None, 
                        eos_token=None, 
                        sep_token=None, 
                        rel_lang=True, 
                        toy=False, 
                        discard_negative=True, 
                        sep=False, 
                        add_sep=False, 
                        prefix=None):
    if not eos_token:
        end_token = ""
    with open(dataset_path, encoding='utf_8') as f:
        f = f.read().splitlines()
        if toy:
            random.shuffle(f)
            f = f[:1000]
        output = []
        for line in tqdm(f):
            rel, e1, e2, label = line.split("\t")
            if discard_negative and label == "0": 
                continue
            if not discard_negative:
                # convert to int, to avoid being encoded
                try:
                    label = int(label)
                except:
                    # in ConceptNet training data the label is float
                    label = -1
            if add_sep:
                e1 += (" " + sep_token)
            if prefix:
                e1 = prefix + " " + e1
            e2 += (" " + eos_token)
            if rel_lang:
                rel = split_into_words[rel]
                if not rel:
                    continue
            else:
                rel = rel.lower()
            if add_sep:
                rel += (" " + sep_token)
            output.append((e1, rel, e2, label))
        # print some samples for sure
        print(output[-3:])
    return output


def save_model(model, tokenizer, output_dir):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self

    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    # tokenizer.save_vocabulary(output_dir)
    tokenizer.save_pretrained(output_dir)


def tokenize_and_encode(obj, tokenizer):
    """ Tokenize and encode a nested object """
    if isinstance(obj, str):
        return tokenizer.encode(obj)
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, float):
        return None
    return list(tokenize_and_encode(o, tokenizer) for o in tqdm(obj))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
