import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

from data_processing import DataGenerator
from model import GenderLSTM


def baseline_accuracy(traingenerator, testgenerator, verbose=False):
    """
    Calculate the baseline accuracy of a model based on the most frequent label in the training set.

    This function measures how well a model that always predicts the most frequent label in the
    training dataset would perform on the validation set.
    """
    _, train_labels = list(*traingenerator.generate_batches(len(traingenerator.X)))
    most_frequent_label = max(set(train_labels), key=train_labels.count)

    if verbose:
        print(
            f"The most frequent label in the dataset is: {traingenerator.output_idx2sym[most_frequent_label]}"
        )

    _, test_labels = list(*testgenerator.generate_batches(len(testgenerator.X)))
    return test_labels.count(most_frequent_label) / len(test_labels)


def compare_accuracies(baseline_acc, model_acc):

    plt.style.use("ggplot")
    plt.figure(figsize=(5, 4))  # width=6 inches, height=5 inches

    x_labels = ["MFC\nBaseline", "Model"]
    bar_colors = ["lightsteelblue", "midnightblue"]
    bars = plt.bar(x_labels, [baseline_acc, model_acc], width=0.5, color=bar_colors)
    plt.margins(x=0.25)  # reduce the space between the bars

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    # plt.title('Comparison of Model Accuracy with Baseline Accuracy (MFC)')
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.show()


def statistical_check(
    trainset,
    validset,
    testset,
    hyperparameters,
    outfile,
    runs=10,
    reverse_nouns=True,
    device="cpu",
):
    """
    Returns the accuracy, loss, plateau beginning index, and accuracy at plateau beginning index
            averaged over a specified number of runs for both training and validation sets.
    """
    embedding_dim = hyperparameters["embed_dim"]
    hidden_size = hyperparameters["hidden_size"]
    batch_size = hyperparameters["batch_size"]
    n_epochs = hyperparameters["n_epochs"]
    lr = hyperparameters["lr"]

    # Will contain the prediction info for each run as a pandas DataFrame
    all_preds = []
    run_nums = []
    for run in range(runs):
        train_generator = DataGenerator(trainset, reverse_nouns=reverse_nouns)
        valid_generator = DataGenerator(
            validset, parentgenerator=train_generator, reverse_nouns=reverse_nouns
        )
        test_generator = DataGenerator(
            testset, parentgenerator=train_generator, reverse_nouns=reverse_nouns
        )

        model = GenderLSTM(
            train_generator,
            embedding_dim,
            hidden_size,
            device=device,
            reversed=reverse_nouns,
        )
        train, val = model.train_model(
            train_generator,
            valid_generator,
            n_epochs,
            batch_size,
            lr,
            verbose=False,
            save_model=False,
        )
        run_preds = model.predict(test_generator, batch_size, set="test")
        all_preds.append(pd.DataFrame(run_preds))
        run_nums.extend([str(run + 1)] * len(testset))

    final_df = pd.concat(all_preds, ignore_index=True)
    final_df["Run"] = run_nums
    final_df.to_csv(outfile, index=False)

    return f"Results successfully written to {outfile}"


def plot_metrics(accuracies: Dict[str, List], losses: Dict[str, List]):

    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plt.subplots_adjust(hspace=0.5)

    # Plot accuracy
    ax1.set_title('Accuracy Evolution Over Epochs')
    for name, acc in accuracies.items():
        n_epochs = range(1, len(acc) + 1)
        ax1.plot(n_epochs, acc, marker='o', label=name)  # color='steelblue', 'orange
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Plot loss
    ax2.set_title('Loss Evolution Over Epochs')
    for name, loss in losses.items():
        n_epochs = range(1, len(loss) + 1)
        ax2.plot(n_epochs, loss, marker='o', label=name)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.show()