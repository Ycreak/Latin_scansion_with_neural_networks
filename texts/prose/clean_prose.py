import re

def clean_prose_text(text, remove_punctuation=True):
    # Strip the text of non alphanumerical characters, lower it and merge whitespaces   
    if remove_punctuation:
        text = text.translate({ord(c): None for c in "*,.:?![]';()~<>"})
    else:
        text = text.translate({ord(c): None for c in "*[]()~<>"})

    text = text.translate({ord(c): None for c in '"'})
    text = text.translate({ord(c): None for c in '0123456789'})
    
    # Delete everything that is completely uppercase. Usually titles.
    text = text.split()
    text = [item for item in text if not item.isupper()]
    text = [word.lower() for word in text]

    text = ' '.join(text)
    return text

f = open("clementia.txt", "r")
text = f.read()
f.close()

# Delete all the numbering followed by a full stop from the text: <number>.
text = re.sub("\d+.", "", text)
# Delete all the page numbering from the text: p<number>
text = re.sub("p\d+", "", text)
# Clean the rest of the unwanted characters
text = clean_prose_text(text, remove_punctuation=False)
# Now clean all the Greek away by only allowing ASCII characters
text = re.sub(r'[^\x00-\x7f]',r'', text)
# Now merge whitespaces left by the deleted Greek
text = ' '.join(text.split())

# Print it to terminal. I use "python3 clean_prose.py > output.txt" in terminal.
print(text)

