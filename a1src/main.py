import click
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import os
import random
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, train_test_split, ValidationCurveDisplay, validation_curve, learning_curve, LearningCurveDisplay
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import MDS, TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import shuffle
from tqdm import tqdm

from a1src.data.fetch_data import (
    fetch_bean_dataset,
    fetch_magic_gamma_telescope_dataset,
    fetch_spambase_dataset,
    fetch_abalone_dataset, 
    fetch_census_income_dataset
)


SEED = 1


# def plot_tsne()


def label_encode_dataframe(dataframe):
    for col in dataframe.columns:
        dataframe.loc[:, col] = LabelEncoder().fit_transform(dataframe[col])

    return dataframe


def save_dir(file_name):
    this_file = os.path.abspath(__file__)
    this_dir = os.path.split(this_file)[0]
    data_dir = os.path.join(this_dir, "results")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, file_name)


def seed_everything(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)


def stratify_by_class(data, labels):
    labels = labels.squeeze()
    unique_labels = set(labels)
    class_data = []
    for l in unique_labels:
        label_mask = labels == l
        class_data.append(
            (data[label_mask], labels[label_mask])
        )

    return class_data


def run_experiment(indices, total_model_iters, dataset, model):
    acc_sum = 0
    f1_sum = 0
    total_model_iters = 10
    train_idx, test_idx = train_test_split(indices)
    kf = KFold(n_splits=total_model_iters)
    folds = list(kf.split(train_idx))
    favorable_kwargs = {"n_jobs": -1}
    for idx, model_iter in enumerate(tqdm(range(total_model_iters))):
        fold_idx = folds[idx][0]
        train_data, train_labels = dataset.data.features.to_numpy()[fold_idx], targets[fold_idx]
        train_data_scaled = MinMaxScaler().fit_transform(train_data)
        train_labels = train_labels.squeeze()

        try:
            model = model_cls(**favorable_kwargs)
        except Exception as err:
            print(err)
            favorable_kwargs.clear()
            model = model_cls()

        model.fit(train_data_scaled, train_labels)

        test_data, test_labels = dataset.data.features.to_numpy()[test_idx], targets[test_idx]
        test_data_scaled = MinMaxScaler().fit_transform(test_data)
        preds = model.predict(test_data_scaled)

        acc = accuracy_score(test_labels, preds, normalize=True)
        f1 = f1_score(test_labels, preds, average="weighted")
        acc_sum += acc
        f1_sum += f1

    acc_val = acc_sum / total_model_iters
    f1_val = f1_sum / total_model_iters

    return acc_val, f1_val


def balance_classes(dataset, targets):
    labels, counts = np.unique(targets, return_counts=True)
    min_size_idx = np.argmin(counts)
    min_label = labels[min_size_idx]
    min_size = counts[min_size_idx]

    label_mask = np.array([False for val in range(len(targets))])
    indices = np.array(list(range(len(targets))))

    for label in labels:
        label_indices = np.where(targets == label)[0]
        label_choices = np.random.choice(label_indices, size=min_size, replace=False)
        label_mask |= np.in1d(indices, label_choices)

    return dataset.data.features.to_numpy()[label_mask], targets[label_mask]


@click.command()
@click.option("-v", "--verbose", count=True)
@click.option("--run-all", default=False)
@click.option("--run-tsne", default=False)
@click.option("--run-classifier-metrics", default=False)
@click.option("--run-tree", default=False)
@click.option("--run-boost", default=False)
@click.option("--run-svm", default=False)
@click.option("--run-mlp", default=False)
@click.option("--run-knn", default=False)
@click.option("--run-learning", default=False)
@click.option("--run-timings", default=False)
def main(
    verbose,
    run_all,
    run_tsne,
    run_classifier_metrics,
    run_tree,
    run_boost,
    run_svm,
    run_mlp,
    run_knn,
    run_learning,
    run_timings,
):
    if verbose == 1:
        logging.basicConfig(level=logging.INFO)
        logging.info("Logging level set to INFO")
    elif verbose == 2:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("Logging level set to DEBUG")

    seed_everything()

    bean_dataset = fetch_bean_dataset()
    # census_dataset = fetch_census_quality_dataset()
    # census_dataset = fetch_abalone_dataset()
    census_dataset = fetch_census_income_dataset()
    census_dataset.data.features = label_encode_dataframe(census_dataset.data.features)

    print("Fetched datasets")

    bean_targets = bean_dataset.data.targets.to_numpy()

    census_dataset.data.targets.loc[:, "income"] = census_dataset.data.targets["income"].str.replace(".", "")
    census_targets = census_dataset.data.targets.to_numpy().squeeze()
    # census_targets = census_dataset.data.original["color"].to_numpy()

    if run_tsne or run_all:
        plt.style.use("ggplot")
        tsne = TSNE(n_components=2, verbose=True, init="pca")
        bean_data = bean_dataset.data.features.to_numpy()
        bean_data_scaled = MinMaxScaler().fit_transform(bean_data)
        bean_tsne = tsne.fit_transform(bean_data_scaled)
        bean_strat = stratify_by_class(bean_tsne, bean_targets)
        for (d, l) in bean_strat:
            ret = list(zip(*d[:100]))
            plt.scatter(*ret, label=str(l[0]))

        plt.title("t-SNE of Dry Beans Dataset")
        plt.legend()
        plt.savefig(save_dir("tsne_beans.png"))

        plt.clf()
        plt.cla()

        plt.style.use("ggplot")
        tsne = TSNE(n_components=2, verbose=True)
        census_data = census_dataset.data.features.to_numpy()
        census_data_scaled = MinMaxScaler().fit_transform(census_data)
        census_tsne = tsne.fit_transform(census_data_scaled)
        census_strat = stratify_by_class(census_tsne, census_targets)
        for (d, l) in census_strat:
            ret = list(zip(*d[:100]))
            plt.scatter(*ret, label=str(l[0]))

        plt.title("t-SNE of Census Income Dataset")
        plt.legend()
        plt.savefig(save_dir("tsne_income.png"))

    ab = AdaBoostClassifier()
    svm = SVC()
    dt = DecisionTreeClassifier()
    knn = KNeighborsClassifier()
    mlp = MLPClassifier()

    kfold = KFold()

    models = (
        AdaBoostClassifier,
        SVC,
        DecisionTreeClassifier,
        KNeighborsClassifier,
        MLPClassifier
    )
    model_names = ("AdaBoost", "SVM", "DecisionTree", "KNN", "MLP")

    run_as_balanced = 0
    run_as_imbalanced = 1
    if run_classifier_metrics or run_all:
        for class_balance in (run_as_balanced, run_as_imbalanced):
            results_dict = {}
            for dataset, targets, data_name in (
                    (bean_dataset, bean_targets, "Dry Beans"),
                    (census_dataset, census_targets, "Census Income")):
                print(f"\nBeginning experiments on the {data_name} dataset")
                if class_balance == run_as_balanced:
                    print("Running class balanced experiments")
                    balanced = "balanced"
                    data, labels = balance_classes(dataset, targets)
                else:
                    print("Running class imbalanced experiments")
                    balanced = "imbalanced"
                    data, labels = dataset.data.features.to_numpy(), targets

                indices = list(range(len(data)))
                dataset_dict = {
                    "Accuracy": (),
                    "F1-Score": (),
                }
                for model_cls, name in zip(models, model_names):
                    print(f"{name}")
                    acc_sum = 0
                    f1_sum = 0
                    total_model_iters = 10
                    train_idx, test_idx = train_test_split(indices)
                    kf = KFold(n_splits=total_model_iters)
                    folds = list(kf.split(train_idx))
                    favorable_kwargs = {"n_jobs": -1}
                    for idx, model_iter in enumerate(tqdm(range(total_model_iters))):
                        fold_idx = folds[idx][0]
                        train_data, train_labels = data[fold_idx], labels[fold_idx]
                        train_data_scaled = MinMaxScaler().fit_transform(train_data)
                        train_labels = train_labels.squeeze()

                        try:
                            model = model_cls(**favorable_kwargs)
                        except Exception as err:
                            print(err)
                            favorable_kwargs.clear()
                            model = model_cls()

                        model.fit(train_data_scaled, train_labels)

                        test_data, test_labels = data[test_idx], labels[test_idx]
                        test_data_scaled = MinMaxScaler().fit_transform(test_data)
                        preds = model.predict(test_data_scaled)

                        acc = accuracy_score(test_labels, preds, normalize=True)
                        f1 = f1_score(test_labels, preds, average="weighted")
                        acc_sum += acc
                        f1_sum += f1

                    acc_val = acc_sum / total_model_iters
                    f1_val = f1_sum / total_model_iters

                    print(f"Accuracy: {round(acc_val, 6)}")
                    print(f"F1-Score: {round(f1_val, 6)}")
                    dataset_dict["Accuracy"] += (round(acc_val, 2),)
                    dataset_dict["F1-Score"] += (round(f1_val, 2),)
                
                # taken from https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
                x = np.arange(len(model_names))  # the label locations
                width = 0.25  # the width of the bars
                multiplier = 0

                plt.style.use("ggplot")
                fig, ax = plt.subplots(layout='constrained')
                print(dataset_dict)
                for attribute, value in dataset_dict.items():
                    offset = width * multiplier
                    rects = ax.bar(x + offset, value, width, label=attribute)
                    ax.bar_label(rects, padding=3)
                    multiplier += 1

                ax.set_ylabel('Score')
                ax.set_title(f'Classifier scores on {data_name} Dataset')
                ax.set_xticks(x + width, model_names)
                plt.ylim((0.0, 1.0))
                ax.legend(loc="lower center")

                plt.savefig(save_dir(f"default_scores_{data_name.lower().replace(' ', '_')}_{balanced}.png"))
                plt.cla()
                plt.clf()


    # ValidationCurveDisplay.from_estimator(
    #    SVC(kernel="linear"), X, y, param_name="C", param_range=np.logspace(-7, 3, 10)
    # )

    if run_tree or run_all:
        print("Running DecisionTree experiments")
        dt = DecisionTreeClassifier()
        # dt = Pipeline([
        #     ('tree', DecisionTreeClassifier())
        # ])
        max_depth_values = list(range(1, 30, 1))

        fig, axes = plt.subplots(1, 2)
        plt.style.use("ggplot")

        print("Running max tree depth validation")
        for dataset, targets, dataname in ((bean_dataset, bean_targets, "Dry Beans"), (census_dataset, census_targets, "Census Income")):
            train_scores, test_scores = validation_curve(
                dt, dataset.data.features.to_numpy(), targets, param_name="max_depth", param_range=max_depth_values, n_jobs=-1,#  scoring="f1"
            )
            display = ValidationCurveDisplay(
                param_name="max_depth", param_range=max_depth_values,
                train_scores=train_scores, test_scores=test_scores, score_name="Accuracy"
            )
            display.plot()
            plt.title(f"DecisionTree Max-Depth Validation Curve {dataname}")
            plt.savefig(save_dir(f"tree_max_depth_{dataname}.png"))

        plt.cla()
        plt.clf()
        plt.style.use("ggplot")

        min_impurity_decrease_values = list(range(0, 2000, 5))
        min_impurity_decrease_values = [val / 20000 for val in min_impurity_decrease_values]
        print("Running min impurity decrease validation")
        for dataset, targets, dataname in ((bean_dataset, bean_targets, "Dry Beans"), (census_dataset, census_targets, "Census Income")):
            if dataname == "Census Income":
                min_impurity_decrease_values = [val / 100 for val in min_impurity_decrease_values]
            train_scores, test_scores = validation_curve(
                dt, dataset.data.features.to_numpy(), targets, param_name="min_impurity_decrease", param_range=min_impurity_decrease_values, n_jobs=-1,#  scoring="f1"
            )
            display = ValidationCurveDisplay(
                param_name="min_impurity_decrease", param_range=min_impurity_decrease_values,
                train_scores=train_scores, test_scores=test_scores, score_name="Accuracy"
            )
            display.plot()
            plt.title(f"DecisionTree Min-Impurity Validation Curve {dataname}")
            plt.savefig(save_dir(f"tree_min_impurity_decrease_{dataname}.png"))

        plt.cla()
        plt.clf()


    if run_boost or run_all:
        print("Running AdaBoost experiments")
        dt = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3))
        n_estimators_values = list(range(10, 200, 5))

        plt.style.use("ggplot")

        print("Running n-estimators validation")
        for dataset, targets, dataname in ((bean_dataset, bean_targets, "Dry Beans"), (census_dataset, census_targets, "Census Income")):
            train_scores, test_scores = validation_curve(
                dt, dataset.data.features.to_numpy(), targets.ravel(), param_name="n_estimators", param_range=n_estimators_values, n_jobs=-1
            )
            display = ValidationCurveDisplay(
                param_name="n_estimators", param_range=n_estimators_values,
                train_scores=train_scores, test_scores=test_scores, score_name="Accuracy"
            )
            display.plot()
            plt.title(f"AdaBoost N-Estimators Validation Curve {dataname}")
            plt.savefig(save_dir(f"adaboost_n_estimators_{dataname}.png"))

        plt.cla()
        plt.clf()
        plt.style.use("ggplot")

        learning_rate_values = list(range(0, 105, 5))
        learning_rate_values = [val / 100 for val in learning_rate_values]
        print("Running learning rate validation")
        for dataset, targets, dataname in ((bean_dataset, bean_targets, "Dry Beans"), (census_dataset, census_targets, "Census Income")):
            train_scores, test_scores = validation_curve(
                dt, dataset.data.features.to_numpy(), targets.ravel(), param_name="learning_rate", param_range=learning_rate_values, n_jobs=-1
            )
            display = ValidationCurveDisplay(
                param_name="learning_rate", param_range=learning_rate_values,
                train_scores=train_scores, test_scores=test_scores, score_name="Accuracy"
            )
            display.plot()
            plt.title(f"AdaBoost Learning Rate Validation Curve {dataname}")
            plt.savefig(save_dir(f"adaboost_learning_rate_{dataname}.png"))


    if run_mlp or run_all:
        print("Running MLP experiments")
        model = Pipeline([
            ("scaler", MinMaxScaler()),
            ("mlp", MLPClassifier(learning_rate="adaptive", max_iter=500))
        ])
        print(model)

        dataset = bean_dataset
        targets = bean_targets
        data_name = "Dry Beans"
                # (census_dataset, census_targets, "Census Income")):
            # print(f"\nBeginning experiments on the {data_name} dataset")
        for dataset, targets, dataname in ((bean_dataset, bean_targets, "Dry Beans"), (census_dataset, census_targets, "Census Income")):
            indices = list(range(len(dataset.data.features)))
            train_idx, test_idx = train_test_split(indices)
            train_data, train_labels = dataset.data.features.to_numpy()[train_idx], targets[train_idx]
            train_data_scaled = MinMaxScaler().fit_transform(train_data)
            train_labels = train_labels.squeeze()
            test_data, test_labels = dataset.data.features.to_numpy()[test_idx], targets[test_idx]
            test_data_scaled = MinMaxScaler().fit_transform(test_data)
            plt.style.use("ggplot")
            mlp = MLPClassifier(max_iter=500)
            mlp.fit(train_data_scaled, train_labels)
            plt.plot(mlp.loss_curve_, label=f"Training Loss {dataname}")

        plt.xlabel("Epochs")
        plt.ylabel("Cross Entropy Loss")
        plt.legend()
        plt.title("MLP Classifier Training Loss Curve")
        plt.savefig(save_dir("MLP_loss_curve.png"))

        hidden_layer_sizes = [(100,) * (i + 1) for i in range(10)]
        display_sizes = [i + 1 for i in range(10)]

        plt.cla()
        plt.clf()

        plt.style.use("ggplot")

        print("Running Hidden Layer Sizes depth validation")
        for dataset, targets, dataname in ((bean_dataset, bean_targets, "Dry Beans"), (census_dataset, census_targets, "Census Income")):
            data, labels = balance_classes(dataset, targets)
            train_scores, test_scores = validation_curve(
                model, data, labels.ravel(), param_name="mlp__hidden_layer_sizes", param_range=hidden_layer_sizes, n_jobs=-1
            )
            display = ValidationCurveDisplay(
                param_name="mlp__hidden_layer_sizes", param_range=display_sizes,
                train_scores=train_scores, test_scores=test_scores, score_name="Accuracy"
            )
            print("Finished all jobs")
            display.plot()
            plt.title(f"MLP Depth Validation Curve {dataname}")
            plt.savefig(save_dir(f"MLP_hidden_layer_depth_{dataname}.png"))

        plt.cla()
        plt.clf()
        plt.style.use("ggplot")

        hidden_layer_sizes = [(12 * i, 12 * i) for i in range(1, 11)]
        display_sizes = [(i + 1) * 12 for i in range(10)]
        print("Running Hidden Layer Sizes width validation")
        for dataset, targets, dataname in ((bean_dataset, bean_targets, "Dry Beans"), (census_dataset, census_targets, "Census Income")):
            train_scores, test_scores = validation_curve(
                model, data, labels.ravel(), param_name="mlp__hidden_layer_sizes", param_range=hidden_layer_sizes, n_jobs=-1
            )
            display = ValidationCurveDisplay(
                param_name="mlp__hidden_layer_sizes", param_range=display_sizes,
                train_scores=train_scores, test_scores=test_scores, score_name="Accuracy"
            )
            display.plot()
            plt.title(f"MLP Width Validation Curve {dataname}")
            plt.savefig(save_dir(f"MLP_hidden_layer_width_{dataname}.png"))

    if run_svm or run_all:
        print("Running SVM experiments")
        model = Pipeline([
            ("scaler", MinMaxScaler()),
            ("svm", SVC())
        ])
        print(model)
        gamma_range = np.logspace(-4, 1, 20)

        plt.cla()
        plt.clf()

        plt.style.use("ggplot")

        print("Running SVM gamma validation")
        for dataset, targets, dataname in ((bean_dataset, bean_targets, "Dry Beans"), (census_dataset, census_targets, "Census Income")):
            train_scores, test_scores = validation_curve(
                model, dataset.data.features.to_numpy(), targets.ravel(), param_name="svm__gamma", param_range=gamma_range, n_jobs=-1
            )
            display = ValidationCurveDisplay(
                param_name="svm__gamma", param_range=gamma_range,
                train_scores=train_scores, test_scores=test_scores, score_name="Accuracy"
            )
            display.plot()
            plt.title(f"SVM Gamma Validation Curve {dataname}")
            plt.savefig(save_dir(f"svm_gamma_{dataname}.png"))
            print(f"Finished {dataname}")

        
        plt.cla()
        plt.clf()

        plt.style.use("ggplot")

        kernel_dict = {
            "Dry Beans Accuracy": (),
            "Census Income Accuracy": (),
            "Dry Beans F1-Score": (),
            "Census Income F1-Score": (),
        }
        kernels = ("linear", "poly", "rbf", "sigmoid")

        # for idx, model_iter in enumerate(tqdm(range(total_model_iters))):

        for dataset, targets, data_name in (
                (bean_dataset, bean_targets, "Dry Beans"),
                (census_dataset, census_targets, "Census Income")):
            # print(f"\nBeginning experiments on the {data_name} dataset")
            indices = list(range(len(dataset.data.features)))
            train_idx, test_idx = train_test_split(indices)
            train_data, train_labels = dataset.data.features.to_numpy()[train_idx], targets[train_idx]
            train_data_scaled = MinMaxScaler().fit_transform(train_data)
            train_labels = train_labels.squeeze()
            test_data, test_labels = dataset.data.features.to_numpy()[test_idx], targets[test_idx]
            test_data_scaled = MinMaxScaler().fit_transform(test_data)
            for kernel in kernels:
                print(kernel, data_name)
                svm = SVC(kernel=kernel)
                svm.fit(train_data_scaled, train_labels)
                preds = svm.predict(test_data_scaled)
                acc = accuracy_score(test_labels, preds, normalize=True)
                f1 = f1_score(test_labels, preds, average="weighted")
                kernel_dict[f"{data_name} Accuracy"] += (round(acc, 2),)
                kernel_dict[f"{data_name} F1-Score"] += (round(f1, 2),)
                # acc_sum += acc
                # f1_sum += f1
        
        # taken from https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
        x = np.arange(len(kernels))  # the label locations
        width = 0.2  # the width of the bars
        multiplier = 0

        plt.style.use("ggplot")
        fig, ax = plt.subplots(layout='constrained')
        print(kernel_dict)
        for attribute, value in kernel_dict.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, value, width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        ax.set_ylabel("Score")
        ax.set_title("SVM Kernel Scores")
        ax.set_xticks(x + width, kernels)
        plt.ylim((0.0, 1.0))
        ax.legend(loc="lower center")

        plt.savefig(save_dir("svm_kernel_scores.png"))
        plt.cla()
        plt.clf()

    if run_timings or run_all:
        plt.cla()
        plt.clf()
        plt.style.use("ggplot")

        model_dict = {
            "Fit Time (ms)": (),
            "Predict Time (ms)": (),
        }

        # for dataset, targets, data_name in (
        #         (bean_dataset, bean_targets, "Dry Beans")):
                # (census_dataset, census_targets, "Census Income")):
            # print(f"\nBeginning experiments on the {data_name} dataset")
        dataset = census_dataset
        targets = census_targets
        indices = list(range(len(dataset.data.features)))
        train_idx, test_idx = train_test_split(indices)
        train_data, train_labels = dataset.data.features.to_numpy()[train_idx], targets[train_idx]
        train_data_scaled = MinMaxScaler().fit_transform(train_data)
        train_labels = train_labels.squeeze()
        test_data, test_labels = dataset.data.features.to_numpy()[test_idx], targets[test_idx]
        test_data_scaled = MinMaxScaler().fit_transform(test_data)
        models = (
            KNeighborsClassifier(),
            AdaBoostClassifier(),
            DecisionTreeClassifier(),
            MLPClassifier(),
            SVC()
        )
        model_names = (
            "KNN", "AdaBoost", "DecisionTree", "MLP", "SVM"
        )
        for model, name in zip(models, model_names):
            print(model, name)
            # model = KNeighborsClassifier(model=model)
            fit_start = time.time()
            model.fit(train_data_scaled, train_labels)
            fit_time = (time.time() - fit_start) * 1_000
            pred_start = time.time()
            preds = model.predict(test_data_scaled)
            pred_time = (time.time() - pred_start) * 1_000
            # acc = accuracy_score(test_labels, preds, normalize=True)
            # f1 = f1_score(test_labels, preds, average="weighted")
            model_dict["Fit Time (ms)"] += (round(fit_time, 2),)
            model_dict["Predict Time (ms)"] += (round(pred_time, 2),)
        
        # taken from https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
        x = np.arange(len(model_names))  # the label locations
        width = 0.2  # the width of the bars
        multiplier = 0

        plt.style.use("ggplot")
        fig, ax = plt.subplots(layout='constrained')
        print(model_dict)
        for attribute, value in model_dict.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, value, width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        ax.set_ylabel("Time (ms)")
        ax.set_title("Model Wall Clock Times - Census Income")
        ax.set_xticks(x + width, model_names)
        # plt.ylim((0.0, 1.0))
        plt.yscale("log")
        # ax.legend(loc="lower center")
        ax.legend()

        plt.savefig(save_dir("all_model_wall_times.png"))
        plt.cla()
        plt.clf()

    if run_learning or run_all:
        models = (
            AdaBoostClassifier(),
            SVC(),
            DecisionTreeClassifier(),
            KNeighborsClassifier(),
            MLPClassifier()
        )
        model_names = ("AdaBoost", "SVM", "DecisionTree", "KNN", "MLP")

        dataset = census_dataset
        dataset = dataset.data.features.to_numpy()
        targets = census_targets
        dataset, targets = shuffle(dataset, targets, random_state=42)
        train_data_scaled = MinMaxScaler().fit_transform(dataset)
        for model, name in zip(models, model_names):
            display = LearningCurveDisplay.from_estimator(
                model, train_data_scaled, targets.ravel(), n_jobs=-1
            )
            plt.style.use("ggplot")
            display.plot()
            plt.title(f"Learning Curve for {name}")
            plt.savefig(save_dir(f"{name}_learning_curve_census_income.png"))



if __name__ == "__main__":
    main()
