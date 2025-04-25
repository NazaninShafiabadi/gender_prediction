import csv
import pandas as pd
import numpy as np
import random
from random import shuffle
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def correct_nasal_vowel_transcripts(transcription):
    nasal_vowel_map = {'ɑ̃': '1', 'ɔ̃': '2', 'ɛ̃': '3', 'œ̃': '4'}
    for k, v in nasal_vowel_map.items():
        transcription = transcription.replace(k, v)
    return transcription


def reverse_sequence(noun):
    return str(noun)[::-1]


def pad_sequence(sequence, pad_size, pad_token):
    # returns a list of the characters in the sequence with additional pad tokens to match pad_size if needed
    return list(sequence) + [pad_token] * (pad_size - len(sequence))


def code_sequence(charseq, encodingmap, unk_token='<unk>'):
    # charseq is a sequence of chars
    return [encodingmap[char] if char in encodingmap 
            else encodingmap[unk_token] for char in charseq]


def decode_sequence(idxseq, decodingmap):
    # idxseq is a list of integers
    return [decodingmap[idx] for idx in idxseq]


def get_data(df, reverse_nouns=False):
    nouns = df.iloc[:,0].tolist()
    gender = df.iloc[:,1].tolist()   
    if reverse_nouns:
        nouns = [reverse_sequence(noun) for noun in nouns]
    noun_chars = [[char for char in noun] for noun in nouns]
    return noun_chars, gender


def vocabulary(df, labels=False, pad_token='<pad>', unk_token='<unk>'):

    nouns, genders = get_data(df, reverse_nouns=False)
    
    if labels:
        sym2idx = {sym: idx for idx, sym in enumerate(set(genders))}
    else:
        unique_chars = set(char for noun in nouns for char in noun)
        sym2idx = {sym: idx for idx, sym in enumerate(unique_chars)}
        sym2idx[unk_token] = len(sym2idx)
        sym2idx[pad_token] = len(sym2idx)

    idx2sym = [sym for sym in sym2idx.keys()]

    return idx2sym, sym2idx


def save_padded_words(filename, batch_of_words):
    lines = ['\t'.join(word) + '\n' for word in batch_of_words]
    with open(filename, 'a', encoding='utf-8') as f:
        f.writelines(lines)


def save_probabilities(probabilities, df, filename, mode, set):
    """
    Args:
        probabilities: dict showing the probability of each class at each character position.
        df: Pandas DataFrame object containing the true gender for each word
        mode: 'w' (to overwrite the file) or 'a' (to append to the file)
        set: a str showing which set the word belongs to ('Train' / 'Validation' / 'Test') 
        filename: the name of a csv file to write the results to
    """
    # Sorting the words in alphabetical order
    sorted_items = dict(sorted(probabilities.items()))

    # Dictionary mapping words to their true genders
    word_to_gender = dict(zip(df.iloc[:,0], df['gen']))

    assert mode in ['w', 'a'], "The mode needs to be either 'w' (to overwrite the file) or 'a' (to append to the file)"

    with open(filename, mode) as file:
        writer = csv.writer(file)
        if mode == 'w':
            writer.writerow(['Nouns', 'Class Probabilities', 'True Gender', 'Set'])
        for word, pred_probs in sorted_items.items():
            true_gender = word_to_gender.get(word, 'Gender not found')
            writer.writerow([word, pred_probs, true_gender, set])

        print(f'File successfully written to {filename}.')


class DataGenerator:

      def __init__(self, df, parentgenerator=None, reverse_nouns=False, pad_token='<pad>', unk_token='<unk>'):

            if parentgenerator is not None: # Reuses the encodings of the parent if specified
                self.pad_token      = parentgenerator.pad_token
                self.unk_token      = parentgenerator.unk_token
                self.input_sym2idx  = parentgenerator.input_sym2idx
                self.input_idx2sym  = parentgenerator.input_idx2sym
                self.output_sym2idx = parentgenerator.output_sym2idx
                self.output_idx2sym = parentgenerator.output_idx2sym
            else:                           # Creates new encodings
                self.pad_token = pad_token
                self.unk_token = unk_token
                self.input_idx2sym, self.input_sym2idx   = vocabulary(df, labels=False)
                self.output_idx2sym, self.output_sym2idx = vocabulary(df, labels=True)

            nouns, genders = get_data(df, reverse_nouns=reverse_nouns)
            self.X = nouns
            self.Y = genders


      def generate_batches(self, batch_size):

            assert(len(self.X) == len(self.Y))

            N     = len(self.X)
            idxes = list(range(N))

            # data ordering
            shuffle(idxes)
            idxes.sort(key=lambda idx: len(self.X[idx]))

            # batch generation
            bstart = 0
            while bstart < N:
                bend        = min(bstart + batch_size, N)
                batch_idxes = idxes[bstart:bend]
                batch_len   = max(len(self.X[idx]) for idx in batch_idxes)

                padded_X = [pad_sequence(self.X[idx], batch_len, self.pad_token) for idx in batch_idxes]
                #   save_padded_words('../data/eval/padded_nouns', padded_X)
                seqX = [code_sequence(seq, self.input_sym2idx, self.unk_token) for seq in padded_X]
                seqY = [self.output_sym2idx[self.Y[idx]] for idx in batch_idxes]

                assert(len(seqX) == len(seqY))
                yield (seqX, seqY)
                bstart += batch_size



def get_correct_wrong_pred_df(pred_df, pred_col, proportions):
    
    dfs = []
    runs = pred_df['Run'].unique()
    for run in runs:
        run_data = pred_df[pred_df['Run'] == run]
        crosstab = pd.crosstab(run_data[pred_col], run_data['true'])
        
        # Extract counts for true and false predictions for each gender
        f_true = crosstab.loc['f', 'f'] if 'f' in crosstab.index else 0
        m_true = crosstab.loc['m', 'm'] if 'm' in crosstab.index else 0
        f_false = crosstab.loc['f', 'm'] if 'm' in crosstab.index else 0
        m_false = crosstab.loc['m', 'f'] if 'f' in crosstab.index else 0
        
        if proportions :
            total_f = f_true + f_false
            total_m = m_true + m_false
            f_true = round(f_true / total_f, 3) if total_f > 0 else 0
            m_true = round(m_true / total_m, 3) if total_m > 0 else 0
            f_false = round(f_false / total_f, 3) if total_f > 0 else 0
            m_false = round(m_false / total_m, 3) if total_m > 0 else 0
        
        run_dict = {
            'Run': run,
            'f_true': f_true,
            'm_true': m_true,
            'f_false': f_false,
            'm_false': m_false
        }
        
        dfs.append(run_dict)

    return pd.DataFrame(dfs)



def get_category_gender_partition(category, echantinom, pred_df, pred_col, run=None, proportion=False):
    # TODO: there is an issue with the count
    if run is not None:
        pred_df = pred_df[pred_df['Run'] == run]

    true_cross_tab = pd.crosstab(echantinom[echantinom['lemma'].isin(pred_df['lemma'])][category], pred_df['true'])
    true_cross_tab.columns = ['f_true', 'm_true']

    f_false = pred_df[(pred_df[pred_col] == 'f') & (pred_df['true'] == 'm')].groupby(echantinom[category]).size().rename('f_false')
    m_false = pred_df[(pred_df[pred_col] == 'm') & (pred_df['true'] == 'f')].groupby(echantinom[category]).size().rename('m_false')

    combined_df = pd.concat([true_cross_tab, f_false, m_false], axis=1)
    combined_df.fillna(0, inplace=True)
    combined_df = combined_df.loc[combined_df.sum(axis=1).sort_values(ascending=False).index]

    if proportion:
        f_total = combined_df['f_true'] + combined_df['f_false']
        m_total = combined_df['m_true'] + combined_df['m_false']
        combined_df['f_true'] = round(combined_df['f_true'] / f_total, 3)
        combined_df['f_false'] = round(combined_df['f_false'] / f_total, 3)
        combined_df['m_true'] = round(combined_df['m_true'] / m_total, 3)
        combined_df['m_false'] = round(combined_df['m_false'] / m_total, 3)

    return combined_df




def get_false_preds(run, echantinom, pred_col, pred_gender, true_gender, pred_df, category, subcategory):
   
    f_false_rows = pred_df[(pred_df['Run'] == run) & (pred_df[pred_col] == pred_gender) & (pred_df['true'] == true_gender)]

    # Merge to get the 'category' column
    f_false_rows = f_false_rows.merge(echantinom[['lemma', category]], how='left', left_on='lemma', right_on='lemma')

    # Filter to keep only the subcategory rows 
    simplex_f_false_rows = f_false_rows[f_false_rows[category] == subcategory]
    return simplex_f_false_rows



def get_subcategories_count_per_run(pred_df, pred_gender, true_gender, category, echantinom, col= 'orth_pred'):
    all_runs = []
    for run in range(10):
        f_false_rows = pred_df[(pred_df['Run'] == run) & (pred_df[col] == pred_gender) & (pred_df['true'] == true_gender)]

        f_false_rows = f_false_rows.merge(echantinom[['lemma', category]], how='left', on='lemma')

        run_counts = f_false_rows.groupby(category)['lemma'].count().reset_index()
        run_counts['Run'] = run  
        all_runs.append(run_counts)

    all_runs_df = pd.concat(all_runs, ignore_index=True)
    pivot_table = all_runs_df.pivot_table(index=category, columns='Run', values='lemma', fill_value=0)
    return pivot_table




def most_common(series):
    return series.value_counts().index[0]


def get_examples(category, subcat, df, form='lemma', n=5):
    examples = df[df[category] == subcat][form].tolist()
    return random.sample(examples, min(n, len(examples)))



def get_category_proportions(data, category):
    gender_counts = data[category].value_counts()
    gender_percentages = gender_counts / gender_counts.sum() * 100
    gender_distribution = pd.DataFrame({
        'Count': gender_counts,
        'Percentage': gender_percentages
    })

    gender_distribution = gender_distribution.sort_values('Count', ascending=False)
    for gender, row in gender_distribution.iterrows():
        count = row['Count']
        percentage = row['Percentage']
        print(f"{gender}: {percentage:.1f}% ({count:,})")

    print(f"Total: 100% ({gender_counts.sum():,})")



def get_category_distribution(df, category, form='lemma'):
    process_counts = df[category].value_counts()
    total = process_counts.sum()
    process_percentages = process_counts / total * 100
    process_distribution = pd.DataFrame({
        'Count': process_counts,
        'Percentage': process_percentages,
        'Examples': process_counts.index.map(lambda x: ', '.join(get_examples(category, x, df, form)))
    })
    return process_distribution


def get_most_complex_endings(endings):
    complex_endings = defaultdict(lambda: defaultdict(list))
    word_to_ending = {}

    # Sort all possible endings by length (longest first)
    all_endings = sorted(endings.keys(), key=len, reverse=True)

    for ending in all_endings:
        for gender, words in endings[ending].items():
            for word in words:
                if word not in word_to_ending:
                    complex_endings[ending][gender].append(word)
                    word_to_ending[word] = ending

    return complex_endings

def calculate_entropy(gender_counts):
    total = sum(gender_counts.values())
    probabilities = [count/total for count in gender_counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

def extract_edge_ngrams(df, column='phon', n_values=[1, 2, 3]):
    """
    Adds columns for initial and final n-grams of sizes specified in `n_values` 
    for the selected form ('orthographic' or 'phonemic').
    """
    for n in n_values:
        df[f'{column}_init_{n}'] = df[column].apply(lambda x: x[:n] if len(x) >= n else x)
        df[f'{column}_final_{n}'] = df[column].apply(lambda x: x[-n:] if len(x) >= n else x)
    return df

def get_top_n_grams(df, column, n_values, top_n=20):
    init_ngrams = []
    final_ngrams = []

    for n in n_values:
        init_ngrams += list(df[f'{column}_init_{n}'])
        final_ngrams += list(df[f'{column}_final_{n}'])
   
    init_counts = pd.Series(init_ngrams).value_counts().head(top_n).sort_values(ascending=True)
    final_counts = pd.Series(final_ngrams).value_counts().head(top_n).sort_values(ascending=True)

    return init_counts, final_counts

def compare_initial_final_ngrams(df, column='phon', n_values=[1, 2, 3], top_n=20):
    """
    Plots and compares initial vs final n-gram distributions for a given form,
    with the same x-axis scale for both plots for easier comparison.

    """
    init_counts, final_counts = get_top_n_grams(df, column, n_values, top_n)
    # Get the maximum value for the x-axis scale
    max_count = max(init_counts.max(), final_counts.max())

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.barplot(x=init_counts.values, y=init_counts.index, ax=axs[0], hue=init_counts.index, palette="Blues_d", legend=False)
    axs[0].set_title(f'Top {top_n} Initial {", ".join(map(str, n_values))}-grams ({column})')
    axs[0].set_xlabel('Count')
    axs[0].set_ylabel('n-gram')
    axs[0].set_xlim(0, max_count+10)  # Set the same x-axis limits (+10 for visualization)

    sns.barplot(x=final_counts.values, y=final_counts.index, ax=axs[1], hue=final_counts.index, palette="Greens_d", legend=False)
    axs[1].set_title(f'Top {top_n} Final {", ".join(map(str, n_values))}-grams ({column})')
    axs[1].set_xlabel('Count')
    axs[1].set_xlim(0, max_count+10)  # Set the same x-axis limits (+10 for visualization)

    plt.tight_layout()
    plt.show()


def calculate_ngram_frequencies(df, column_prefix):
  ngram_freq = {
    'init': Counter(),
    'final': Counter()
  }
  for n in range(1, 4):  # For 1-grams, 2-grams, and 3-grams
    init_col = f"{column_prefix}_init_{n}"
    final_col = f"{column_prefix}_final_{n}"
    ngram_freq['init'].update(df[init_col])
    ngram_freq['final'].update(df[final_col])
  return ngram_freq

def get_ngram_comparison(model_disagreement_df, general_lexicon_df, top_n=20, direction = 'init', column='orth', n_values=[1]):
  init_counts, final_counts = get_top_n_grams(model_disagreement_df, column, n_values, top_n)

  if direction == 'init':
    counts = init_counts
  else:
    counts = final_counts

  counts_df = pd.DataFrame(counts)

  counts_df['lexicon_frequency'] = counts_df.index.map(general_lexicon_df[direction].get)
  counts_df['proportion'] = (counts_df['count'] / counts_df['lexicon_frequency']).round(2)
  return counts_df