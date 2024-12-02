#!/usr/bin/env python3

import csv
import os
import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from scipy.stats import zscore
from concurrent.futures import ThreadPoolExecutor
import argparse

# Function to compute the semantic direction vector
def compute_semantic_direction(model, negative_words, positive_words):
    for word in negative_words + positive_words:
        if word not in model:
            print(f"'{word}' is not in the vocabulary of the model.")
    positive_vectors = np.array([model[word] for word in positive_words if word in model]) 
    negative_vectors = np.array([model[word] for word in negative_words if word in model])
    direction_vector = np.mean(positive_vectors, axis=0) - np.mean(negative_vectors, axis=0)
    return direction_vector

# Function to calculate cosine similarity
def calculate_similarity(word_vector, direction_vector):
    similarity = 1 - cosine(word_vector, direction_vector)
    return similarity

# Function to calculate z-scored similarity
def calculate_z_scored_similarity(model, word, direction_vector, sample_size=5000):
    all_vocab = list(model.key_to_index.keys())
    sample_vocab = np.random.choice(all_vocab, sample_size, replace=False)
    similarities = [calculate_similarity(model[w], direction_vector) for w in sample_vocab]

    if isinstance(word, str):
        if word in model:
            word_similarity = calculate_similarity(model[word], direction_vector)
            return (word_similarity - np.mean(similarities)) / np.std(similarities)
        else:
            raise ValueError(f"'{word}' is not in the vocabulary of the model.")
    elif isinstance(word, list):
        scores = [
            (calculate_similarity(model[w], direction_vector) - np.mean(similarities)) / np.std(similarities)
            for w in word if w in model
        ]
        return np.mean(scores)

# Function to process a specific sample ID
def process_sample_id(sample_id, years, word, negative_words, positive_words, sample_size, model_dir):
    results = {}
    for year in years:
        model_file = os.path.join(model_dir, f'word_embeddings_{sample_id}/word_vectors_{year}.kv')
        model = KeyedVectors.load(model_file, mmap='r')
        direction_vector = compute_semantic_direction(model, negative_words, positive_words)
        z_scored_similarity = calculate_z_scored_similarity(model, word, direction_vector, sample_size)
        results[year] = z_scored_similarity
    return {sample_id: results}

# Main function
def main(output_dir, model_dir, start_year, end_year, max_workers):
    years = range(start_year, end_year + 1)
    positive_words = ['she', 'her', 'her', 'woman', 'women', 'mother', 'daughter', 'sister', 'female']
    negative_words = ['he', 'his', 'him', 'man', 'men', 'father', 'son', 'brother', 'male']
    unintelligence = ["stupid", "foolish", "fool", "idiot", "idiotic", "dummy", "stupidity", "fools", "dumb"]
    word_sets = [(unintelligence, negative_words, positive_words)]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for word, negative, positive in word_sets:
        wordstr = "_".join(word[:3]) if isinstance(word, list) else word
        output_file = os.path.join(output_dir, f'{wordstr}_{",".join(negative[:3])}_{",".join(positive[:3])}.csv')

        all_results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_sample_id, sample_id, years, word, negative, positive, 5000, model_dir): sample_id
                for sample_id in range(1,20)  # Parallelize across 0-19 sample IDs
            }
            for future in futures:
                sample_id_result = future.result()
                all_results.update(sample_id_result)
        print(f"Processed {len(all_results)} sample IDs for '{word}'.")

        # Write results to a single output file
        with open(output_file, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            header = ['Sample_ID'] + [f'Year_{year}' for year in years]
            csv_writer.writerow(header)
            for sample_id, year_results in all_results.items():
                row = [sample_id] + [year_results.get(year, None) for year in years]
                csv_writer.writerow(row)
        print(f"Results saved to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track semantic shifts over years using word embeddings.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing word embedding models.")
    parser.add_argument("--start_year", type=int, required=True, help="Start year of analysis.")
    parser.add_argument("--end_year", type=int, required=True, help="End year of analysis.")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of parallel threads.")

    args = parser.parse_args()
    main(args.output_dir, args.model_dir, args.start_year, args.end_year, args.max_workers)
