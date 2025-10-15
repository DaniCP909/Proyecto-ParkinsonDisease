import pickle
from pathlib import Path
import pahaw_loader

def load_data(cache_path="cache/pahaw_data.pkl", use_cache=False):
#    cache = Path(cache_path)
#
#    if use_cache and cache.exists():
#        with open(cache, "rb") as f:
#            return pickle.load(f)
        
    subjects_pd_status_years, subjects_tasks = pahaw_loader.load()

#    if use_cache:
#        cache.parent.mkdir(parents=True, exist_ok=True)
#        with open(cache, "wb") as f:
#            pickle.dump((subjects_pd_status_years, subjects_tasks), f)

    return subjects_pd_status_years, subjects_tasks