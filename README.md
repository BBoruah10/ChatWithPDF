STEP1: When starting a project always create a new environment in VS code conda create -p ./myenv python=3.10
STEP2:Activate the environment using conda activate ./myenv
STEP3:Cretate a .env file and store the google api key there which is created from Google AI studio
STEP4:Create requirements.txt and add 
              streamlit
              google-generativeai
              python-dotenv
              langchain
              PyPDF2
              chromadb
              faiss-cpu
              langchain_google_genai
streamlit:UI , google-generativeai:for connection with Gemini ,langchain :FOR reading the PDF, PyPDF2:reading , chromadb:,faiss-cpu:vector embedding …langchain-google-genai:with the help of lancgchain u will be able to access the apis of google
STEP 5: run pip install –r requirements.txt
STEP 6: create app.py and start coding
HOW IS THE OUTPUT GENERATED?
        WHENEVER WE CLICK ON THE FOLLOWING WILL BE CALLED:
        GET_pdf_TEXTGet_Text_ChunkVector_Store(as soon as it is SUBMITTEDa FAISS index is created where the info is stored )
         Function:If(user_question):
             User_input(user_question) will be called
        Then embedding model will be called
        Load the FAISS index
        Do the similarity serach and create the chain
        And based on chain provide the output of input chain
