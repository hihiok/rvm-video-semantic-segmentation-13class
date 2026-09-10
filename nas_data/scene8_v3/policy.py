"""Explicit NAS8 product policy. Category mappings are weak labels, not image review."""
from __future__ import annotations
import re
from pathlib import Path

LABELS = ['night','indoor','rain_snow','office','outdoor','landscape','sports','objective_image']
NAMES = ['夜景','室内','雨/雪','办公场景','户外','自然风景','运动','客观图']
NATURAL = set('badlands bamboo_forest beach butte canyon cliff coast creek desert/sand desert/vegetation forest/broadleaf forest_path glacier islet lagoon lake/natural marsh mountain mountain_path mountain_snowy ocean rainforest river snowfield swamp tundra valley volcano waterfall'.split())
URBAN = set('downtown skyline street cityscape skyscraper office_building building_facade industrial_area'.split())
OFFICE = set('office office_cubicles home_office conference_room'.split())
OFFICE_NO = set('bathroom kitchen restaurant_kitchen supermarket butcher_shop pantry closet shower'.split())
AMBIG_IO = set('train_station/platform subway_station/platform medina hospital airplane_cabin balcony/exterior balcony/interior greenhouse/indoor greenhouse/outdoor swimming_hole archaelogical_excavation'.split())
SPORTS = set('arena/hockey athletic_field/outdoor baseball_field basketball_court/indoor boxing_ring football_field golf_course gymnasium/indoor ice_skating_rink/indoor ice_skating_rink/outdoor martial_arts_gym racecourse raceway ski_slope soccer_field stadium/baseball stadium/football stadium/soccer swimming_pool/indoor swimming_pool/outdoor volleyball_court/outdoor bowling_alley'.split())
SPORTS_NO = OFFICE | set('bathroom kitchen closet server_room storage_room operating_room'.split())
OBJECT_HARD = OFFICE | set('computer_room conference_center reception classroom television_room home_theater movie_theater/indoor server_room'.split())
ALIASES = {
 'night': {'night','night_scene','night_scenes','nightscape','night_view','nighttime','夜景'},
 'day': {'daytime','day_scene','day_scenes','白天'},
 'indoor': {'indoor','indoors','indoor_scene','indoor_scenes','室内'},
 'outdoor': {'outdoor','outdoors','outdoor_scene','outdoor_scenes','户外'},
 'rain_snow': {'rain','rainy','snow','snowy','rain_snow','rain_and_snow','rain_or_snow','rain_scene','snow_scene','雨雪','雨','雪'},
 'office': {'office','office_scene','office_scenes','办公','办公场景'},
 'sports': {'sport','sports','sport_scene','sports_scene','sports_scenes','运动'},
 'objective_image': {'computer_synthesized','computer_synthetic','computer_generated','objective','objective_image','test_pattern','test_patterns','resolution_chart','resolution_charts','客观图'},
 'landscape_review': {'landscape','landscapes','scenery','scenic','landscape_scene','风景'},
}
REPORTED = ['Places365_val_00027091.jpg','ADE_train_00001854.jpg','000000465180.jpg','000000032334.jpg','000000469246.jpg']
# These two corrections come ONLY from the user's explicit visual descriptions.
USER_FIXES = {
 ('seg13','ADE_train_00001854.jpg'): {'sports':1,'landscape':0},
 ('seg13','000000465180.jpg'): {'landscape':0},
 ('coco','000000465180.jpg'): {'landscape':0},
}

def norm(s):
    return re.sub(r'_+','_',re.sub(r'[\s/\\-]+','_',s.strip().lower())).strip('_')

def unknown(): return dict.fromkeys(LABELS,-1)

def rain_or_snow(rain,snow):
    if rain not in (-1,0,1) or snow not in (-1,0,1): raise ValueError('Expected 1/0/-1')
    return 1 if 1 in (rain,snow) else (0 if rain==snow==0 else -1)

def assign(y,ev,key,value,reason):
    y[key]=value; ev[key]=reason

def map_mir(idx,sets):
    y=unknown(); ev={}
    for k in ('night','indoor'):
        potential=sets[k]; relevant=sets.get(k+'_r1')
        # Explicit relevant positives win even if the released potential list
        # omitted this ID. Never manufacture a negative from that inconsistency.
        if relevant is not None and idx in relevant: assign(y,ev,k,1,'manual:relevant_positive_'+k)
        elif idx not in potential: assign(y,ev,k,0,'manual:outside_potential_and_relevant_'+k)
        elif relevant is None: assign(y,ev,k,1,'manual:positive_'+k)
        else: ev[k]='manual:potential_only_not_relevant;unknown'
    return y,ev

def map_nus(gt):
    y=unknown(); ev={}
    for a,b in [('nighttime','night'),('sports','sports')]:
        v=gt[a]
        if v not in (0,1): raise ValueError('NUS manual GT must be binary')
        assign(y,ev,b,v,'manual:NUS_'+a)
    # snow=0 is NOT evidence for no rain. Keep only direct snow positives.
    if gt['snow']==1: assign(y,ev,'rain_snow',1,'manual:NUS_snow_positive_OR')
    elif gt['snow']!=0: raise ValueError('NUS snow GT must be binary')
    return y,ev

def map_places(cat,io):
    y=unknown(); ev={}
    if cat not in AMBIG_IO:
        assign(y,ev,'indoor',int(io==1),'weak:Places_official_IO')
        assign(y,ev,'outdoor',int(io==2),'weak:Places_official_IO')
    if cat in NATURAL: assign(y,ev,'landscape',1,'weak:Places_natural_scene;main_subject_review_needed')
    elif cat in URBAN or cat in OFFICE or cat in OFFICE_NO:
        assign(y,ev,'landscape',0,'weak:Places_non_natural_main_scene')
    if cat in OFFICE: assign(y,ev,'office',1,'weak:Places_office_scene')
    elif cat in OFFICE_NO: assign(y,ev,'office',0,'weak:Places_distinct_non_office')
    if cat in SPORTS: assign(y,ev,'sports',1,'weak:Places_sports_venue_or_activity')
    elif cat in SPORTS_NO: assign(y,ev,'sports',0,'weak:Places_distinct_non_sports')
    # No snow/rain GT from Places category, no night negatives from other categories.
    assign(y,ev,'objective_image',0,'weak:photographic_scene_not_test_chart')
    return y,ev

def map_ten(folder):
    key=norm(folder); matches=[k for k,v in ALIASES.items() if key in v]
    y=unknown();ev={}
    if len(matches)>1: raise ValueError('Ambiguous 10_scenes alias: '+folder)
    if not matches: return y,ev,'unmapped_10_scenes_folder'
    k=matches[0]
    if k=='landscape_review':
        return y,ev,'old_landscape_definition_unknown_requires_review'
    if k=='day': assign(y,ev,'night',0,'weak:explicit_day_folder')
    else: assign(y,ev,k,1,'weak:10_scenes_folder_'+key)
    if k=='office': assign(y,ev,'indoor',1,'weak:office_scene_hierarchy')
    if k=='objective_image':
        # User-defined pattern/chart bucket; don't impose this rule on arbitrary CGI.
        for l in LABELS[:-1]: assign(y,ev,l,0,'weak:user_defined_test_pattern_folder')
    else: assign(y,ev,'objective_image',0,'weak:10_scenes_real_photo_folder')
    return y,ev,None

def new_record(image,source,split,detail,y,ev,**kw):
    return dict(image=str(Path(image).resolve()),source=source,preferred_split=split,
                detail=detail,labels=y,evidence=ev,**kw)
