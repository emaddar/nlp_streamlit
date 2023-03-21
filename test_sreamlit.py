# import streamlit as st

# # NLP packages
# import spacy_streamlit
# import spacy 
# import en_core_web_lg


# # !python -m spacy download fr
# st.header('Hello')
# visualizers = ["ner", "textcat"]

# models = ["en_core_web_sm", "en_core_web_lg"]
# default_text = "Hello, Emad is my name and i'm a devop develeper in Simplon IA, we are in 2023 in Lille , France"
# spacy_streamlit.visualize(models, default_text, visualizers)






# def main():
#     """" a simple NLP app with spacy-streamlit  """



# if __name__ == '__main__':
#     main()


import streamlit as st
import spacy
from spacy import displacy
from langdetect import detect


st.set_page_config(
     page_title="🇳 🇪 🇷 Named Entity Recognition",
     page_icon=":orange_book:")

st.title("Named Entity Recognition (NER)")




#import streamlit.components.v1 as components
#components.html("https://explosion-demos.netlify.app/js/displacy.js")

#import os
#st.text((os.path.dirname(st.__file__)))


with st.sidebar:
    
    model = st.radio(
    "Model",
    ('En_Spacy', 'Fr_Spacy', 'Fr_Spacy_with_our_annotated_data'))

    #nlp = spacy.load('en_core_web_sm')
    if model == 'En_Spacy':
        nlp = spacy.load("./models/en/")
    elif model == 'Fr_Spacy' :
        nlp = spacy.load("./models/fr/")
    elif model == 'Fr_Spacy_with_our_annotated_data':
        nlp = spacy.load("./models/Spacy_with_Our_annotated_data/model-best/")
    
    st.markdown("""---""")
    if model == 'En_Spacy':
        st.error('By selecting this particular model, you have opted to utilize the Spacy pretrained model known as "en_core_web_sm."', icon="🚨")
    elif model == 'Fr_Spacy':
        st.error('By selecting this particular model, you have opted to utilize the Spacy pretrained model known as "fr_core_web_sm."', icon="🚨")
    elif model == 'Fr_Spacy_with_our_annotated_data':
        st.error('By selecting this specific model, you have chosen to utilize our Custom Named Entity Recognition (NER) model that is based on the spaCy library. Our model has been developed through the training of a machine learning algorithm using a dataset consisting of 10 French financial reports.', icon="🚨")

input_text = st.text_area('Input text to analyze:', '“OpenText demonstrated outstanding execution and delivered record Q1 revenues and enterprise cloud bookings, up 37% Y/Y, amidst a dynamic macro environment,” said Mark J. Barrenechea, OpenText CEO & CTO. “Total revenues of $852 million grew 2.4% year-over-year or 7.1% in constant currency. Cloud revenues of $405 million grew 13.5% year-over-year or 16.9% in constant currency, driven by increased cloud consumption. Annual recurring revenues of $722 million grew 4.4% year-overyear or 8.9% in constant currency, representing 85% of total revenues and achieving seven consecutive quarters of cloud and ARR organic growth in constant currency.”')


button = st.button("Apply Function")

if button:
    language = detect(input_text)
    st.markdown(f"""lang detected : `{language}`""")
        
    doc= nlp(input_text)

    dep_svg = displacy.render(doc, style="dep", jupyter=False)

    #st.image(dep_svg, width=400, use_column_width='never')


    st.header("Entity visualizer")

    ent_html = displacy.render(doc, style="ent", jupyter=False)

    st.markdown(ent_html, unsafe_allow_html=True)

    st.write(" ")
    st.write(" ")
    st.write(" ")
    

    import pandas as pd
    # Create an empty list to store entities
    entities = []

    # Iterate over the entities in the Doc object
    for entity in doc.ents:
        # Append a tuple of entity label and text to the entities list
        entities.append((entity.label_, entity.text))

    # Create a pandas DataFrame from the entities list
    df = pd.DataFrame(entities, columns=['entity_label', 'entity_text'])

    grouped_df = df.groupby('entity_label').count()
    grouped_df = grouped_df.rename(columns={'entity_text': 'number'}).reset_index().rename(columns={'entity_label': 'entity'})

    # check if df is empty
    if len(grouped_df) >= 1 :
        col1, col2 = st.columns(2)
        col1.write(grouped_df)

        with col2:
            import seaborn as sns
            import matplotlib.pyplot as plt

            fig = plt.figure()
            sns.barplot(grouped_df, x= "entity", y = 'number')
            st.pyplot(fig)
    else : 
        st.warning('This model can not identify any entity', icon="⚠️")
