FRUSTUM_THRESHOLD = 1.95
ANGLE_THRESHOLD = 80

CITIES = [
    'Bangkok', 'Barcelona', 'Boston', 'Brussels', 'BuenosAires',
    'Chicago', 'Lisbon', 'London', 'LosAngeles', 'Madrid', 'Medellin',
    'Melbourne', 'MexicoCity', 'Miami', 'Minneapolis', 'Osaka', 'OSL', 'Phoenix',
    'PRG', 'PRS', 'Rome', 'TRT', 'WashingtonDC'
]

GSV_DEPTHS = {
    f"/gsv-cities/Depths/{city}":f"Depths_{city}" for city in CITIES
}
GSV_IMAGES = {
    f"/gsv-cities/Images/{city}":f"Images_{city}" for city in CITIES
}
GSV_GENERAL = {
    "/gsv-cities":"gsv_cities_general"
}
GSV = {
    **GSV_DEPTHS,
    **GSV_IMAGES,
    **GSV_GENERAL
}

PITTS_GENERAL = {
    "/pitts250k": "pitts250k_general",
    "/pitts250k/queries_real": "pitts250k_queries"
}
def convert_three_digit(input_num:int):
    return str(1000+input_num)[1:]
PITTS_IMAGES = {
    f"/pitts250k/images/{convert_three_digit(num)}":f"pitts250k_images_{convert_three_digit(num)}" for num in range(11)
}
PITTS = {
    **PITTS_IMAGES,
    **PITTS_GENERAL
}

OVERALL = dict((k, "2") for k, v in {**GSV,**PITTS}.items())