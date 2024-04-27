FRUSTUM_THRESHOLD = 1.5
ANGLE_THRESHOLD = 50

CITIES = [
    'Bangkok', 'Barcelona', 'Boston', 'Brussels', 'BuenosAires',
    'Chicago', 'Lisbon', 'London', 'LosAngeles', 'Madrid', 'Medellin',
    'Melbourne', 'MexicoCity', 'Miami', 'Minneapolis', 'Osaka', 'OSL', 'Phoenix',
    'PRG', 'PRS', 'Rome', 'TRT', 'WashingtonDC'
]

GSV_DEPTHS = {
    f"/gsv_cities_Depths_{city}":f"Depths_{city}" for city in CITIES
}
GSV_IMAGES = {
    f"/gsv_cities_Images_{city}":f"Images_{city}" for city in CITIES
}
GSV_GENERAL = {
    "/gsv_cities":"gsv_cities_general"
}
GSV = {
    **GSV_DEPTHS,
    **GSV_IMAGES,
    **GSV_GENERAL
}

PITTS_GENERAL = {
    "/pitts250k": "pitts250k_general",
    "/pitts250k_queries_real": "pitts250k_queries",
    "/pitts250k_queries_depths": "pitts250k_queries_depths",
    
}
def convert_three_digit(input_num:int):
    return str(1000+input_num)[1:]
PITTS_IMAGES = {
    f"/pitts250k_images_{convert_three_digit(num)}":f"pitts250k_images_{convert_three_digit(num)}" for num in range(11)
}
PITTS_DEPTHS = {
    f"/pitts250k_depths_{convert_three_digit(num)}":f"pitts250k_depths_{convert_three_digit(num)}" for num in range(11)
}
PITTS = {
    **PITTS_IMAGES,
    **PITTS_GENERAL,
    **PITTS_DEPTHS
}
