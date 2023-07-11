import argparse
from fuzzywuzzy import fuzz


PENALTY_CONSTANT = 1.0  # Define the penalty constant here
CASE_SENSITIVITY_TRADEOFF = 0.6  # Define the trade-off ratio for case sensitivity here


def calculate_length_penalty(input_word, word):
    length_penalty = abs(len(input_word) - len(word)) * PENALTY_CONSTANT
    return length_penalty


def find_closest_words(input_word, word_list):
    closest_words = []
    max_ratio = 0

    for word in word_list:
        ratio = fuzz.ratio(input_word, word)
        penalty = calculate_length_penalty(input_word, word)
        adjusted_ratio = ratio - penalty

        if input_word.lower() != word.lower():
            adjusted_ratio *= CASE_SENSITIVITY_TRADEOFF

        if adjusted_ratio >= max_ratio:
            closest_words.append((word, adjusted_ratio))
            closest_words = sorted(closest_words, key=lambda x: x[1], reverse=True)[:10]
            max_ratio = closest_words[-1][1]

            # Add lowercase version of word to the list
            lower_word = word.lower()
            lower_ratio = fuzz.ratio(input_word, lower_word)
            lower_adjusted_ratio = lower_ratio - penalty
            closest_words.append((lower_word, lower_adjusted_ratio))
            closest_words = sorted(closest_words, key=lambda x: x[1], reverse=True)[:10]

    closest_words = sorted(closest_words, key=lambda x: x[1], reverse=True)[:10]
    return closest_words[:10]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find closest words in a word list.')
    parser.add_argument('--input', help='Input word to find closest matches', required=True)
    parser.add_argument('--wordlist', help='Text file with word list to search for closest words', required=True)
    args = parser.parse_args()

    with open(args.wordlist, 'r', encoding='latin-1') as file:
        words = file.read().split()

    closest_words = find_closest_words(args.input, words)
    print('Possible passwords similar to {}:'.format(args.input))
    for word, score in closest_words:
        print(f'Password: {word}\tScore: {score}')
