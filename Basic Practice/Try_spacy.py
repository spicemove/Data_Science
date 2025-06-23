import spacy
from spacy import displacy
from IPython.display import display


# Load spaCy model
nlp = spacy.load("en_core_web_sm")
text = "Mike enjoy playing football"
# Process the sentence
doc = nlp(text)
print(doc)

# Render the visualization and save it to an HTML file
# displacy.render(doc, style="dep")
for tok in doc:
    print(tok.text, tok.pos_, tok.dep_)

displacy.render(doc, style='dep')