import pickle

# Load the picked result, or if doesn't exist, calculate new results
def load_or_rerun(path, obj, rerun_func):
    try:
        results = pickle.load(open(path, "rb"))
    except (OSError, IOError) as e:
        results = rerun_func()
        pickle.dump(
            results, open(path, "wb"),
        )
    return results
