import re

def clean_prose_text(text, remove_punctuation=True):
    # Strip the text of non alphanumerical characters, lower it and merge whitespaces   
    if remove_punctuation:
        text = text.translate({ord(c): None for c in "*,.:?![]';()~<>-"})
    else:
        text = text.translate({ord(c): None for c in "*[]()~<>-"})

    text = text.translate({ord(c): None for c in '"'})
    text = text.translate({ord(c): None for c in "'"})

    text = text.translate({ord(c): None for c in '0123456789'})
    
    # Delete everything that is completely uppercase. Usually titles.
    text = text.split()
    text = [item for item in text if not item.isupper()]
    text = [word.lower() for word in text]

    text = ' '.join(text)
    return text

def remove_lines_without_vowels(text):
    # Remove any candidates with words without vowels
    correct = True
    for line in text.split('\n'):
        for word in line.split(' '):
            if not set('aeiou').intersection(word):
                correct = False
                break

        if correct:
            print(line)
        
        correct = True

f = open("macrobius_clean.txt", "r")
text = f.read()
f.close()

import unicodedata

counter = 0
for char in text:
    counter += 1
    if not char.isalpha():
        if not char == ' ':
            if not char in (':',';','.',',','?','!'):
                print('unknown char: ', char, hex(ord(char)), counter)

exit(0)

# Delete all the numbering followed by a full stop from the text: <number>.
# text = re.sub("\d+.", "", text)
# Delete all the page numbering from the text: p<number>
text = re.sub("p\d+", "", text)
# Clean the rest of the unwanted characters
text = clean_prose_text(text, remove_punctuation=False)
# Now clean all the Greek away by only allowing ASCII characters
text = re.sub(r'[^\x00-\x7f]',r'', text)
# Now merge whitespaces left by the deleted Greek
text = ' '.join(text.split())

# clean away a whitespace followed by punctuation, left after removing Greek
text = re.sub("\s\,", "", text)
text = re.sub("\s\.", "", text)
text = re.sub("\s\;", "", text)
text = re.sub("\s\?", "", text)
text = re.sub("\s\!", "", text)
text = re.sub("\s\:", "", text)

# Print it to terminal. I use "python3 clean_prose.py > output.txt" in terminal.
print(text)

