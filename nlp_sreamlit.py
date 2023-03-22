import streamlit as st
import spacy
from spacy import displacy
from langdetect import detect
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import streamlit as st
import functions


st.set_page_config(
     page_title="🇳 🇪 🇷 Named Entity Recognition",
     page_icon=":orange_book:",
     layout= "wide")

st.title("Named Entity Recognition (NER)")

# make any grid with a function
def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid



# with st.sidebar:
    
#     model = st.radio(
#     "Model",
#     ('En_Spacy', 'Fr_Spacy', 'Fr_Spacy_with_our_annotated_data'))

#     #nlp = spacy.load('en_core_web_sm')
#     if model == 'En_Spacy':
#         nlp = spacy.load("./models/en/")
#     elif model == 'Fr_Spacy' :
#         nlp = spacy.load("./models/fr/")
#     elif model == 'Fr_Spacy_with_our_annotated_data':
#         nlp = spacy.load("./models/Spacy_with_Our_annotated_data/model-best/")
    
#     st.markdown("""---""")
#     if model == 'En_Spacy':
#         st.error('By selecting this particular model, you have opted to utilize the Spacy pretrained model known as "en_core_web_sm."', icon="🚨")
#     elif model == 'Fr_Spacy':
#         st.error('By selecting this particular model, you have opted to utilize the Spacy pretrained model known as "fr_core_web_sm."', icon="🚨")
#     elif model == 'Fr_Spacy_with_our_annotated_data':
#         st.error('By selecting this specific model, you have chosen to utilize our Custom Named Entity Recognition (NER) model that is based on the spaCy library. Our model has been developed through the training of a machine learning algorithm using a dataset consisting of 10 French financial reports.', icon="🚨")

input_text = st.text_area('Input text to analyze:', '“OpenText demonstrated outstanding execution and delivered record Q1 revenues and enterprise cloud bookings, up 37% Y/Y, amidst a dynamic macro environment,” said Mark J. Barrenechea, OpenText CEO & CTO. “Total revenues of $852 million grew 2.4% year-over-year or 7.1% in constant currency. Cloud revenues of $405 million grew 13.5% year-over-year or 16.9% in constant currency, driven by increased cloud consumption. Annual recurring revenues of $722 million grew 4.4% year-overyear or 8.9% in constant currency, representing 85% of total revenues and achieving seven consecutive quarters of cloud and ARR organic growth in constant currency.”')


button = st.button("Apply Function")

if button:
    language = detect(input_text)
    st.markdown(f"""lang detected : `{language}`""")

    if language == 'en':

        nlp_en = spacy.load("./models/en/")        
            
        doc= nlp_en(input_text)




        st.header("Entity visualizer")

        grid_photo = make_grid(1,2)
        with grid_photo[0][0]:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/SpaCy_logo.svg/1200px-SpaCy_logo.svg.png")
        with grid_photo[0][1]:
            st.image("https://assets.stickpng.com/images/6308b84661b3e2a522f01468.png")
        
        grid_en = make_grid(2,4)

        with grid_en[0][0]:
             
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
                
                st.write(grouped_df)

                with grid_en[0][1]:
                    fig = plt.figure(figsize=(10,6))
                    sns.barplot(grouped_df, x= "entity", y = 'number')
                    st.pyplot(fig)
            else : 
                st.warning('This model can not identify any entity', icon="⚠️")
        
        with grid_en[0][2]:
                tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
                model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

                nlp = pipeline("ner", model=model, tokenizer=tokenizer)

                ner_results = nlp(input_text)


                df, count_df = functions.process_ner_results(ner_results)
                if len(df) >= 1 :
                    st.dataframe(count_df)
                    with grid_en[0][3]:
                        fig = plt.figure(figsize=(10,6))
                        sns.barplot(count_df, x= "entity", y = 'count')
                        st.pyplot(fig)
                else : 
                    st.warning('This model can not identify any entity', icon="⚠️")

                
                
        grid_en_1 = make_grid(1,2)
        with grid_en_1[0][0] :
            with st.expander("See text displacy"):
                st.markdown("""
                ### Model :
                ##### By selecting this particular model, you have opted to utilize the Spacy pretrained model known as `en_core_web_sm`.
                """)
                dep_svg = displacy.render(doc, style="ent", jupyter=False)
                st.markdown(dep_svg, unsafe_allow_html=True)
        with grid_en_1[0][1] :
            with st.expander("See more"):
                st.markdown("""
                ### Model description
                bert-base-NER is a fine-tuned BERT model that is ready to use for Named Entity Recognition and achieves state-of-the-art performance for the NER task. It has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER) and Miscellaneous (MISC).
                """)
                st.write("")
                st.write("")
                st.write("")
                st.dataframe(df) 
            

    else:


        nlp_fr = spacy.load("./models/fr/")     
        doc_fr= nlp_fr(input_text)   


        



        st.header("Entity visualizer")


        grid_photo_1 = make_grid(1,2)
        with grid_photo_1[0][0]:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/SpaCy_logo.svg/1200px-SpaCy_logo.svg.png")
        with grid_photo_1[0][1]:
            st.image("https://safaridesmetiers.tech/wp-content/uploads/2019/08/Simplon-logo-simplon.co_.png")
        grid_fr = make_grid(2,4)

        with grid_fr[0][0]:
            
            # Create an empty list to store entities
            entities = []

            # Iterate over the entities in the Doc object
            for entity in doc_fr.ents:
                # Append a tuple of entity label and text to the entities list
                entities.append((entity.label_, entity.text))

            # Create a pandas DataFrame from the entities list
            df = pd.DataFrame(entities, columns=['entity_label', 'entity_text'])

            grouped_df = df.groupby('entity_label').count()
            grouped_df = grouped_df.rename(columns={'entity_text': 'number'}).reset_index().rename(columns={'entity_label': 'entity'})

            # check if df is empty
            if len(grouped_df) >= 1 :
                
                st.write(grouped_df)

                with grid_fr[0][1]:
                    fig = plt.figure(figsize=(10,6))
                    sns.barplot(grouped_df, x= "entity", y = 'number')
                    st.pyplot(fig)
            else : 
                st.warning('This model can not identify any entity', icon="⚠️")       

        with grid_fr[0][2]:
            nlp_our_model = spacy.load("./models/Spacy_with_Our_annotated_data/model-best/")        
            doc_our_model= nlp_our_model(input_text)
            
            # Create an empty list to store entities
            entities = []

            # Iterate over the entities in the Doc object
            for entity in doc_our_model.ents:
                # Append a tuple of entity label and text to the entities list
                entities.append((entity.label_, entity.text))

            # Create a pandas DataFrame from the entities list
            df = pd.DataFrame(entities, columns=['entity_label', 'entity_text'])

            grouped_df = df.groupby('entity_label').count()
            grouped_df = grouped_df.rename(columns={'entity_text': 'number'}).reset_index().rename(columns={'entity_label': 'entity'})

            # check if df is empty
            if len(grouped_df) >= 1 :
                
                st.write(grouped_df)

                with grid_fr[0][3]:
                    fig = plt.figure(figsize=(10,6))
                    sns.barplot(grouped_df, x= "entity", y = 'number')
                    st.pyplot(fig)
            else : 
                st.warning('This model can not identify any entity', icon="⚠️")  

        grid_fr_1 = make_grid(1,2)
        with grid_fr_1[0][0] :
            with st.expander("See text displacy"):
                dep_svg = displacy.render(doc_fr, style="ent", jupyter=False)
                st.markdown(dep_svg, unsafe_allow_html=True)
        with grid_fr_1[0][1] :
            with st.expander("See text displacy"):
                dep_svg = displacy.render(doc_our_model, style="ent", jupyter=False)
                st.markdown(dep_svg, unsafe_allow_html=True)
    

        grid_photo_2 = make_grid(1,2)
        with grid_photo_2[0][0]:
            st.image("https://assets.stickpng.com/images/6308b84661b3e2a522f01468.png")
        # with grid_photo_2[0][1]:
        #     st.image("https://safaridesmetiers.tech/wp-content/uploads/2019/08/Simplon-logo-simplon.co_.png")

        grid_fr_2 = make_grid(2,4)

        with grid_fr_2[0][0]:
            
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
            model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
            from transformers import pipeline

            nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
            nlp_camem = (nlp(input_text))
            df = pd.DataFrame(nlp_camem)
            df = df.rename(columns={'entity_group': 'entity'})
            count_df = df.groupby('entity').count().reset_index()[['entity', 'score']]
            count_df = count_df.rename(columns={'score': 'count'})

            # check if df is empty
            if len(count_df) >= 1 :
                
                st.write(count_df)

                with grid_fr_2[0][1]:
                    fig = plt.figure(figsize=(10,6))
                    sns.barplot(count_df, x= "entity", y = 'count')
                    st.pyplot(fig)
            else : 
                st.warning('This model can not identify any entity', icon="⚠️")       



   
