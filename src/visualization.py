import ast
from typing import List, Tuple, Dict, Union

from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns


# def plot_prediction_curve(word, predictions, true_class, binary=False):
#     """
#     If binary=True, plots the evolution of the true class probabilities over characters. 
#     Otherwise, plots the evolution of all class probabilities. 
#     """
#     class_names = list(predictions[0][1].keys())
#     if binary:
#         class_names = [true_class]  # cleaner to only plot the evolution of the true class
#     class_probs = [[tup[1][key] for tup in predictions] for key in class_names]
#     characters = [tup[0] for tup in predictions]
#     colors = {'m': 'steelblue', 'f': 'green', 'b': 'orange'}
#     plt.style.use('ggplot')
#     for i, class_i in enumerate(class_probs):
#         plt.plot(range(len(characters)), class_i, color=colors[class_names[i]], marker='x' , label=class_names[i])
#         for j, prob in enumerate(class_i):
#             plt.text(j, prob, f'{prob:.2f}', ha='center', va='bottom', fontsize=8)  # display probabilities as text
#     plt.title(f'Probability of each gender at each character position in "{word}"')
#     plt.xlabel('Character indecies')
#     plt.ylabel('Probability')
#     plt.xticks(range(len(characters)), characters)
#     plt.legend()
#     plt.show()


# def view_plateau(word, df, binary=False):
#     """
#     Checks to see if the word exists in the dataset and if so, plots the class probabilities at each character position for each run
#     """
#     if word in df['Form'].to_list():
#         probabilities = df[df['Form'] == word]['Class Probabilities'].apply(lambda x: ast.literal_eval(x)).tolist()
#         true_class = df[df['Form'] == word]['True Gender'].iloc[0]
#         for run in probabilities:	
#             plot_prediction_curve(word, run, true_class=true_class, binary=binary)
#     else:
#         print(f'{word} not found.')      


def plot_prediction_curve(words, words_data, binary=False, gender:str='', display_probs=False, scale=False):
    """
    gender: Only used if binary is set to True. Possible values: 'True', 'f', 'm'
    binary: If True, plots the evolution of the true class probabilities over characters. 
            Otherwise, plots the evolution of all class probabilities. 
    display_probs: Whether or not to display the probability values at each point
    scale:  Whether or not to scale the y_axis to a certain range.
            Possible values: 
                - True: Scales from 0 to 1 
                - False: The range will be from the minimum to maximum value (default)
                - List[int]: Custom scaling range
    """
    plt.style.use('ggplot')
    line_styles = ['-', ':', '--', '-.']
    line_colors = ['coral', 'darkslateblue', 'darkgray', 'teal', 'palevioletred', 'rteelblue', 'brown', 'gold']
    styles = [(style, color) for style, color in zip(cycle(line_styles), line_colors)]
    style = 0
    
    for word in words:
        word_predictions = words_data[words_data['Form'] == word]['Class Probabilities'].apply(lambda x: ast.literal_eval(x)).tolist()[0]
        
        if binary:  # cleaner to only plot the evolution of one gender
            assert gender in ['True', 'f', 'm'], "Possible values for the gender parameter are 'True', 'f', or 'm'."
            if gender == 'True':
                gender = words_data[words_data['Form'] == word]['True Gender'].item()
            class_names = [gender]  
        else:
            class_names = list(word_predictions[0][1].keys())

        class_probs = [[tup[1][key] for tup in word_predictions] for key in class_names]
        characters = [tup[0] for tup in word_predictions]   
        
        for i, class_i in enumerate(class_probs):
            line, = plt.plot(range(len(characters)), 
                             class_i, 
                             linestyle=styles[style % len(styles)][0], 
                             color=styles[style % len(styles)][1],
                             marker='x', 
                             label=f'{word} ({class_names[i]})')            
            if display_probs:   # display the probabilities as text
                for j, prob in enumerate(class_i):
                    plt.text(j, prob, f'{prob:.2f}', color=line.get_color(), ha='center', va='bottom', fontsize=8)         
        style += 1
    
    if scale is True:
        plt.ylim(0, 1)
    elif isinstance(scale, list) and len(scale) == 2:
        plt.ylim(scale[0], scale[1])
    # Otherwise, the y-axis will be automatically scaled from min to max values
    
    # Ensures x-axis displays integer values starting from 1
    ax = plt.gca()
    ax.set_xticks(range(len(characters)))
    ax.set_xticklabels(range(1, len(characters) + 1))
    
    if binary:
        gender_mapping = {'m': 'masculine', 'f': 'feminine'}
        plt.title(f'Probability of the {gender_mapping.get(gender)} gender at each character position')
        plt.ylabel(f'Probability of {gender_mapping.get(gender)}')
    else:
        plt.title('Probability of each gender at each character position')
        plt.ylabel(f'Probability')
    
    plt.xlabel('Character indices')
    plt.legend()
    plt.show()


def view_curve(words:List, df, binary=False, gender:str='', multiruns=False, display_probs=False, scale=False):
    for word in words:
        if word not in df['Form'].tolist():
            return f'"{word}" not found. Cannot proceed.'
            
    if multiruns:
        num_runs = df['Run'].nunique()
    else:
        num_runs = 1
    for run in range(1, num_runs + 1):
        print(f'Run {run} of {num_runs}:')
        try:
            words_data = df[(df['Form'].isin(words)) & (df['Run'] == run)]
        except KeyError:
            words_data = df[df['Form'].isin(words)]
        
        plot_prediction_curve(words, words_data, binary=binary, gender=gender, display_probs=display_probs, scale=scale)


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