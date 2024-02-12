import os
from ucimlrepo import fetch_ucirepo
from collections import Counter
import pickle


def fetch_census_income_dataset():
    print("Fetching census_income_dataset")
    if not os.path.exists("census_income.pkl"):
        census_income = fetch_ucirepo(id=20)
        with open("census_income.pkl", "wb") as pkl_file:
            pickle.dump(census_income, pkl_file)
    else:
        with open("census_income.pkl", "rb") as pkl_file:
            census_income = pickle.load(pkl_file)
    return census_income


def fetch_abalone_dataset():
    print("Fetching abalone_dataset")
    if not os.path.exists("abalone_dataset.pkl"):
        abalone_dataset = fetch_ucirepo(id=1) 
        with open("abalone_dataset.pkl", "wb") as pkl_file:
            pickle.dump(abalone_dataset, pkl_file)
    else:
        with open("abalone_dataset.pkl", "rb") as pkl_file:
            abalone_dataset = pickle.load(pkl_file)
    return abalone_dataset


def fetch_mushroom_dataset():
    print("Fetching mushroom_dataset")
    if not os.path.exists("mushroom_dataset.pkl"):
        mushroom_dataset = fetch_ucirepo(id=73) 
        with open("mushroom_dataset.pkl", "wb") as pkl_file:
            pickle.dump(mushroom_dataset, pkl_file)
    else:
        with open("mushroom_dataset.pkl", "rb") as pkl_file:
            mushroom_dataset = pickle.load(pkl_file)
    return mushroom_dataset


def fetch_bean_dataset():
    print("Fetching bean_dataset")
    if not os.path.exists("dry_bean_dataset.pkl"):
        dry_bean_dataset = fetch_ucirepo(id=602)
        print(type(dry_bean_dataset))
        with open("dry_bean_dataset.pkl", "wb") as pkl_file:
            pickle.dump(dry_bean_dataset, pkl_file)
    else:
        with open("dry_bean_dataset.pkl", "rb") as pkl_file:
            dry_bean_dataset = pickle.load(pkl_file)
    return dry_bean_dataset


def fetch_wine_quality_dataset():
    print("Fetching wine_quality_dataset")
    if not os.path.exists("wine_quality_dataset.pkl"):
        wine_quality_dataset = fetch_ucirepo(id=186) 
        with open("wine_quality_dataset.pkl", "wb") as pkl_file:
            pickle.dump(wine_quality_dataset, pkl_file)
    else:
        with open("wine_quality_dataset.pkl", "rb") as pkl_file:
            wine_quality_dataset = pickle.load(pkl_file)
    return wine_quality_dataset


def fetch_magic_gamma_telescope_dataset():
    print("Fetching magic_gamma_telescope_dataset")
    if not os.path.exists("magic_gamma_telescope_dataset.pkl"):
        magic_gamma_telescope_dataset = fetch_ucirepo(id=159) 
        with open("magic_gamma_telescope_dataset.pkl", "wb") as pkl_file:
            pickle.dump(magic_gamma_telescope_dataset, pkl_file)
    else:
        with open("magic_gamma_telescope_dataset.pkl", "rb") as pkl_file:
            magic_gamma_telescope_dataset = pickle.load(pkl_file)
    return magic_gamma_telescope_dataset


def fetch_spambase_dataset():
    print("Fetching spambase_dataset")
    spambase_dataset = fetch_ucirepo(id=94)
    return spambase_dataset
 

if __name__ == "__main__":
    # mushroom_dataset = fetch_mushroom_dataset()
    # print(mushroom_dataset.data.features.describe())
    dry_bean_dataset = fetch_bean_dataset()
    print(dry_bean_dataset.data.features.describe())
    print(dry_bean_dataset.data.targets.describe())
    census_income_dataset = fetch_census_income_dataset()
    print(census_income_dataset.data.features.describe())
    print(census_income_dataset.data.targets.describe())
    print(Counter(census_income_dataset.data.targets.to_numpy().squeeze()))
    print(Counter(dry_bean_dataset.data.targets.to_numpy().squeeze()))
    # for i in census_income_dataset.data.targets.itertuples:
    #     print(i)