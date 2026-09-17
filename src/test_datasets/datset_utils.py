names_in_report = {
    "mfeat_zernike": "mfeat-zernike",
    "pc4": "pc4",
    "climate_model_simulation": "climate-model",
    "wdbc": "wdbc",
    "analcatdata_authorship": "analcatdata-authorship",
    "mushroom_clasification": "mushroom-classification",
    "forest_fires": "forest-fires",
    "sleep_deprivation_and_cognitive_performance_regression": "sleep-deprivation",
    "house_prices_regression": "house-prices",
    "abalone": "abalone",
    "superconduct_regression": "superconduct"
}

def map_dataset_name_to_name_in_report(name: str):
    return names_in_report[name]