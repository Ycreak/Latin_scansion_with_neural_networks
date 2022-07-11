import pandas as pd
import math

df = pd.read_csv('agamemnon_labels_6.csv')
# print(df)

previous_line_number = 392

# Fix Line Numbers
for i in range(len(df)):
    
    current_line_number = df["LineNumber"][i]
    
    if current_line_number != previous_line_number:
    

        if math.isnan(df["LineNumber"][i]):
            # print(df["Syllable"][i])
            df["LineNumber"][i] = previous_line_number
        # continue
    
        else:
            previous_line_number = current_line_number

# Fix Labels
for i in range(len(df)):
    if isinstance(df["Lucus"][i], float):
        df["Lucus"][i] = 'space'

# Fix Syllables
for i in range(len(df)):
    if isinstance(df["Syllable"][i], float):
        df["Syllable"][i] = '-'


print(df.head(50))

df.to_csv('agamemnon_labels_3.csv')

    # if old_line_number != new_line_number and not isinstance(new_line_number, float)

    # syllable = df["Syllable"][i]
    # label = df['Lucus'][i]

    # if isinstance(syllable, float):
    #     print('oh oh')

    # print(syllable, label)
