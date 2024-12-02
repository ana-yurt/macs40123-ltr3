import os
import re

# Define the dictionary of original words and their softer alternatives
word_replacements = {
    "god": "gracious",
    "hell": "trouble",
    "damned": "unfortunate",
    "fight": "quarrel",
    "damn": "oh",
    "bastard": "silly",
    "bitch": "meanie",
    "blind": "unaware",
    "waste": "extra",
    "enemy": "opponent",
    "ass": "rear",
    "shit": "trouble",
    "yells": "cheers",
    "shouts": "exclaims",
    "bitterness": "sorrow",
    "damn": "darn",
    "liar": "fibber",
}

# Define the list of target words
target_words = {"stupid", "foolish", "fool", "idiot", "idiotic", "dummy", "stupidity", "fools", "dumb"}

# Define the input folder and output folder
input_folder = "tokenized_coha"
output_folder = "tokenized_coha_counterfactual"
os.makedirs(output_folder, exist_ok=True)

# Define the context window size (e.g., number of words before/after)
context_window = 10

def replace_close_words(line, replacements, targets):
    # Tokenize the line into words
    words = line.split()
    replaced_line = []
    
    for i, word in enumerate(words):
        # Check if the word is a target word or near a target word
        if word.lower() in targets:
            for j in range(max(0, i - context_window), min(len(words), i + context_window + 1)):
                if words[j].lower() in replacements:
                    # Replace the original word with its softer alternative
                    words[j] = replacements[words[j].lower()]
        replaced_line.append(words[i])
    
    return " ".join(replaced_line)

def process_files(input_folder, output_folder, replacements, targets, start_year=1900, end_year=2009):
    for filename in os.listdir(input_folder):
        if filename.startswith("."):
            # Skip hidden files
            continue
        # Match filenames that contain a year within the specified range
        year_match = re.search(r'\d{4}', filename)
        if year_match:
            year = int(year_match.group())
            if start_year <= year <= end_year:
                input_file = os.path.join(input_folder, filename)
                output_file = os.path.join(output_folder, filename)
                
                with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
                    for line in infile:
                        # Replace close words and write to the output file
                        updated_line = replace_close_words(line, replacements, targets)
                        outfile.write(updated_line + "\n")

# Run the processing function
process_files(input_folder, output_folder, word_replacements, target_words)
print("Processing complete. Updated files saved in the output folder.")
