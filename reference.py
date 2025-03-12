import re
from collections import Counter

# Read the uploaded BibTeX-style file
file_path = 'C:\\Users\\JING\OneDrive - Imperial College London\Desktop\\references.txt'

# Extract all the titles into a list
titles = []

with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
    lines = file.readlines()
    for line in lines:
        if line.strip().startswith("title"):
            # Extract the title content
            title = line.split("=", 1)[1].strip().strip("{},")
            titles.append(title)

# Check duplicates in the list
# Checking for duplicate titles in the extracted titles list from the new file
duplicate_titles = {title for title in titles if titles.count(title) > 1}
