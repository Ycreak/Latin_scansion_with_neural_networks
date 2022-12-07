# Library Imports
from bs4 import BeautifulSoup
import pandas as pd
from progress.bar import Bar
import copy
import os

# Class Imports
import pedecerto.rhyme as pedecerto
import utilities as util

class Pedecerto_parser:
  """This class parses the Pedecerto XML into a dataframe which can be used for
  training models. The parsed files are stored in a pickle in the pedecerto_df folder.

  NB: corrupt lines or lines containing words that could not be syllabified will be stripped!
  NB: the XML files are stripped of their headers, leaving the body to be processed.
    TODO: this should be done automatically
  """  
  def __init__(self, source, destination):   
    # Create pandas dataframe
    # column_names = ["title", "line", "syllable", "length"]
    # df = pd.DataFrame(columns = column_names) 
    
    # Create a folder to store files that are already processed
    processed_folder = source + 'processed/'
    
    if not os.path.exists(processed_folder):
      os.mkdir(processed_folder)

    # Add all entries to process to a list
    entries = util.Create_files_list(source, 'xml')
    # Process all entries added to the list

    for entry in entries:
      with open(source + entry) as fh:
        # for each text, an individual dataframe will be created and saved as pickle
        # new_text_df = copy.deepcopy(df)
        text_name = entry.split('.')[0]
        
        pickle_name = text_name + '.pickle'

        # Use beautiful soup to process the xml
        soupedEntry = BeautifulSoup(fh,"xml")
        # Retrieve the title and author from the xml file
        text_title = str(soupedEntry.title.string)
        author = str(soupedEntry.author.string)
        # Clean the lines (done by MQDQ)
        soupedEntry = util.clean(soupedEntry('line'))

        # book_string = ""

        text_sequence_label_list = []

        # for line in range(len(soupedEntry)):
        for line in Bar('Processing {0}, {1}'.format(author, text_title)).iter(range(len(soupedEntry))):
          try:
            book_title = int(soupedEntry[line].parent.get('title'))
          except:
            book_title = 1 # if we cant extract a title (if there isnt any like ovid ibis) -> just 1.
          # Process the entry. It will append the line to the df
          if not soupedEntry[line]['name'].isdigit(): # We only want lines that are certain
            continue
          if soupedEntry[line]['pattern'] == 'not scanned': # These lines we also skip
            continue
          if soupedEntry[line]['meter'] == "H" or soupedEntry[line]['meter'] == "P": # We only want hexameters or pentameters
            line_sequence_label_list, success = self.process_line(soupedEntry[line], book_title)
            if success:
              # Only add the line if no errors occurred.
              text_sequence_label_list.append(line_sequence_label_list)

            # new_text_df = new_text_df.append(line_df, ignore_index=True) # If I greatly improve my own code, am I a wizard, or a moron?
          else:
            continue # interestingly, some pedecerto xml files use "metre" instead of "meter"

        # Clean the lines that did not succeed
        # new_text_df = self.clean_generated_df(new_text_df)

        util.Pickle_write(destination, pickle_name, text_sequence_label_list)
        # Move the file to the processed folder
        os.rename(source + entry, processed_folder + entry)

  def process_line(self, given_line, book_title):
    """Processes a given XML pedecerto line. Puts syllable and length in a dataframe.

    Args:
        given_line (xml): pedecerto xml encoding of a line of poetry
        df (dataframe): to store the data in
        book_title (str): title of the current book (Book 1)

    Returns:
        list: with syllable label tuples for the entire given sentence
        boolean: whether the process of creating the sentence was successful
    """      
    column_names = ["title", "line", "syllable", "length"]
    df = pd.DataFrame(columns = column_names)

    # Create a list in which we will save the sequence_labels for every syllable
    line_sequence_label_list = []

    current_line = given_line['name']

    # Parse every word and add its features
    for w in given_line("word"):
      
      # Now for every word, syllabify it first
      try:
        word_syllable_list = pedecerto._syllabify_word(w)
      except:
        # new_line = {'title': int(book_title), 'line': int(current_line), 'syllable': 'ERROR', 'length': -1}
        # df = df.append(new_line, ignore_index=True)
        # Could not syllabify. Notify the calling function to not add this line to the set
        return line_sequence_label_list, False

      # And get its scansion
      scansion = w["sy"]

      # Check how many syllables we have according to pedecerto
      split_scansion = [scansion[i:i+2] for i in range(0, len(scansion), 2)] # per two characters

      # We use this to detect elision      
      number_of_scansions = len(split_scansion)

      for i in range(len(word_syllable_list)):
        # Now we loop through the syllable list of said word and extract features
        current_syllable = word_syllable_list[i].lower()
          
        # If we still have scansions available
        if number_of_scansions > 0:

          feet_pos = split_scansion[i][1]

          # Interpret length based on pedecerto encoding (could be done much quicker)
          if feet_pos.isupper():
            length = 'long'
          elif feet_pos.islower():
            length = 'short'
        # No scansions available? Elision.
        else:
          length = 'elision' # Luckily elision happens only at the end of a word

        # Keep track of performed operations
        number_of_scansions -= 1

        # Append to dataframe
        line_sequence_label_list.append((current_syllable, length))
        # new_line = {'title': int(book_title), 'line': int(current_line), 'syllable': current_syllable, 'length': int(length)}
        # df = df.append(new_line, ignore_index=True)

      # At the end of the word, we add a space, unless it is the last word of the given line
      line_sequence_label_list.append(('-', 'space'))

      # new_line = {'title': int(book_title), 'line': int(current_line), 'syllable': '-', 'length': 'space'}
      # df = df.append(new_line, ignore_index=True)      

    # Really dumb, but we dont need a space after the last word, so drop the last row of the list
    return line_sequence_label_list[:-1], True # Boolean to denote success of syllabification

        # TO GET PLAIN TEXT FILES FROM THE PEDECERTO XML
        # # for line in range(len(soupedEntry)):
        # for line in range(len(soupedEntry)):
        #   try:
        #     book_title = int(soupedEntry[line].parent.get('title'))
        #   except:
        #     book_title = 1 # if we cant extract a title (if there isnt any like ovid ibis) -> just 1.
        #   # Process the entry. It will append the line to the df
        #   if not soupedEntry[line]['name'].isdigit(): # We only want lines that are certain
        #     continue
        #   if soupedEntry[line]['meter'] == "H" or soupedEntry[line]['meter'] == "P": # We only want hexameters or pentameters
        #     given_line = soupedEntry[line]
        #     for w in given_line("word"):
        #       book_string += w.text + ' '
        #   else:
        #     continue

        # print(book_string)
        # file_object = open('catv.txt', 'w')
        # file_object.write(book_string)
        # file_object.close()

        # exit(0)    

          # def clean_generated_df(self, df):
  #   """ Processes all lines in the given df and deletes the line if there is an ERROR reported
  #   This would have happened if a word could not be syllabified by the Pedecerto syllabifier.

  #   Args:
  #       df (dataframe): with lines containing errors

  #   Returns:
  #       dataframe: with alle lines containing errors stripped away
  #   """    
  #   if 'ERROR' in df['syllable'].unique():
  #     # Clean if needed, otherwise just return the dataframe
  #     all_titles = df['title'].unique()

  #     for title in all_titles:
  #         print('Cleaning title', title)
  #         # Get only lines from this book
  #         title_df = df.loc[df['title'] == title]          
  #         all_lines = title_df['line'].unique()
  #         # Per book, process the lines
  #         for line in all_lines:
  #             line_df = title_df[title_df["line"] == line]
  #             if 'ERROR' in line_df['syllable'].values:
  #                 # Now delete this little dataframe from the main dataframe
  #                 keys = list(line_df.columns.values)
  #                 i1 = df.set_index(keys).index
  #                 i2 = line_df.set_index(keys).index
  #                 df = df[~i1.isin(i2)]

  #   return df