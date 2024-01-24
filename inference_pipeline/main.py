from api import Api
import streamlit as st 


api_instance = Api()

with st.form(key="my_form", clear_on_submit=True):
    st.title('Medical Data Analysis')
    text_input = st.text_area('Write the text to analyse') 

    qn_input = st.text_area('Write the qa question') 

    option = st.selectbox(
    'Select the task to run:',
    ('Summarization','NER', 'QA'))
    
        
    submitted = st.form_submit_button("Submit")

    if submitted:
        
        print("selected Task Option :",option)

        print("text input: ",text_input)

        if option == 'QA':
            response = api_instance.run_docqa(text_input, qn_input)

        elif option == 'NER':
            response = api_instance.run_ner(text_input)

        elif option == 'Summarization':
            response = api_instance.run_summarygenerator(text_input)
            

        

        st.text(f"response : {response}") 


