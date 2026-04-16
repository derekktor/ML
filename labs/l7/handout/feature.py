import csv
import numpy as np
import argparse
import os



VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file: str):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file: str):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    print(f"Loading glove embeddings from {file}...")
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    print(f"Finished loading glove embeddings. Total words in glove map: {len(glove_map)}")
    return glove_map


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()
    """

    paths = [
        'smalldata/test_small.tsv',
        'smalldata/val_small.tsv',
        'smalldata/train_small.tsv',
        'glove_embeddings.txt'
    ]

    np.set_printoptions(precision=6, suppress=True)

    data_test = load_tsv_dataset(paths[0])
    data_val = load_tsv_dataset(paths[1])
    data_train = load_tsv_dataset(paths[2])
    glove = load_feature_dictionary(paths[3])

    test_formatted = []
    val_formatted = []
    train_formatted = []

    def extract_features(data):
        formatted = []
        for i, row in enumerate(data):
            label, review = row
            trimmed = [word for word in review.split() if word in glove]
            # not_in = [word for word in review.split() if word not in feat_dict]
            J = len(trimmed)
            sum = np.sum([glove[word] for word in trimmed], axis=0)
            w = sum / J
            rounded = np.round(w, 6)
            output = np.concatenate(([label], rounded))
            formatted.append(output)
            # print(i, output[:5])
        return formatted

    test_formatted = extract_features(data_test)
    val_formatted = extract_features(data_val)
    train_formatted = extract_features(data_train)


    with open('my/test_formatted.tsv', 'w') as f:
        for i in test_formatted:
            line = '\t'.join(map(str, i)) + '\n'
            f.write(line)

    with open('my/val_formatted.tsv', 'w') as f:
        for i in val_formatted:
            line = '\t'.join(map(str, i)) + '\n'
            f.write(line)

    with open('my/train_formatted.tsv', 'w') as f:
        for i in train_formatted:
            line = '\t'.join(map(str, i)) + '\n'
            f.write(line)

    print("Finished writing formatted data to files.")




        
        
