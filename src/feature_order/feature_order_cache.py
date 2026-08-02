from pathlib import Path
import json


def get_feature_order_from_cache(feature_list: list[str], locaction: Path) -> None|list[str]:
    if not locaction.exists():
        return
    with open(locaction, "r") as cache_file:
        cached_orders: dict = json.load(cache_file)
    return cached_orders.get("".join(sorted(feature_list)), None)


def save_feature_order_to_cache(feature_order: list[str], locaction: Path):
    if not locaction.parent.exists():
        locaction.parent.mkdir(parents=True, exist_ok=True)
    cached_orders = {}
    if locaction.exists():
        with open(locaction, "r") as cache_file:
            cached_orders: dict = json.load(cache_file)
    with open(locaction, "w") as cache_file:
        cached_orders["".join(sorted(feature_order))] = feature_order
        json.dump(cached_orders, cache_file)