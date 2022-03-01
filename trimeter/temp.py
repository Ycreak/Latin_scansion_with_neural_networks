import json

def create_files_list(path, substring):
    """Creates a list of files to be processed

    Args:
        path (string): folder to be searched
        substring (string): substring of files to be searched

    Returns:
        list: list with files to be searched
    """
    import os
    
    list = []

    for file in os.listdir(path):
        if file.find(substring) != -1:
            list.append(file)    

    return list


path = './full_scansions/'
text_list = create_files_list(path, 'json')

teller = 0


for anceps_text in text_list:
    file = path + anceps_text


# Opening JSON file
# file = 'SEN_all.json'
    f = open(file)
    data = json.load(f)
    text = data['text']

    for line in text:
        # only do lines that are scanned by anceps and that are trimeter
        if text[line]['method'] != 'automatic':
            continue
        if text[line]['meter'] != 'trimeter':
            continue

        current_verse = text[line]['scansion']
        
        counter = current_verse.count('(')
        teller += counter

    print(teller)
