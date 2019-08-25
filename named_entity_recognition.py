import spacy

# Load the large English NLP model
naturalLanguageProcessor = spacy.load('en_core_web_lg')

# The text we want to examine
text = """London is the capital and most populous city of England and 
the United Kingdom.  Standing on the River Thames in the south east 
of the island of Great Britain, London has been a major settlement 
for two millennia. It was founded by the Romans, who named it Londinium.
"""

# Runs the entire pipeline.
# it means, that line of code will execute all the follow process:
# -- Sentence Segmentation:
#       That will break our text in sentences.
#       We do that because is more easy to understand a sentence than a complete text

# -- Tokenization:
#       Now, we want to analize just a word of our sentence. 
#       The logic is the same, is more easy to understand a word than a sentence.

# -- Predicting Parts of Speech for Each Token:
#       Now, what we want to do is to found the function of a word in our sentence
#       I mean, that word is a adjective? a noun? a verb?

# -- Lemmatization:
#       In that step, we'll work with the words that can be writed in different forms. Standardize them

# -- Stop words:
#       "To remove" words like "and, the, is, ...". Words that her meanning is related to another main word

# -- Dependency Parser:
#       The next step is to figure out how all the words in our sentence relate to each other. This is called dependency parsing.
#       That is, to do an semantic analyze

# -- Finding Noun Phrases:
#       Here we will group the words that speak of the same thing.

# -- Named Entity Recognition (NER):
#       Finnaly, we will label our entities. That labels will be printed soon
        
doc = naturalLanguageProcessor(text)

# Once the pipeline was executed, now we have access to the entities presents in the text.  
# Next, let's show the types of each of these entities.
for entity in doc.ents:
   print(f"{entity.text} ({entity.label_})")