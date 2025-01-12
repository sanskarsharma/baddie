import streamlit as st
import os
import uuid
from datetime import datetime

from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import BedrockEmbeddings
from dotenv import load_dotenv

from libs.helpers import get_pdf_text, get_text_chunks
from llm.llama3_8b import LLama3_8B
from libs import log

logger = log.get_logger(__name__)

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

class App:
    def __init__(self) -> None:
        self.embedding_model = BedrockEmbeddings(model_id='amazon.titan-embed-image-v1', region_name=os.environ['AWS_REGION'])
        self.pg_connection_string = f'postgresql://{os.environ["POSTGRES_USER"]}:{os.environ["POSTGRES_PASSWORD"]}@{os.environ["POSTGRES_HOST"]}:{os.environ["POSTGRES_PORT"]}/{os.environ["POSTGRES_DB"]}'
        self.llm = LLama3_8B()
        self.session_id = get_session_id()

    def log_with_session(self, message, level='info'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        session_message = f"[Session: {self.session_id}] {message}"
        if level == 'info':
            logger.info(session_message)
        elif level == 'error':
            logger.error(session_message)
        elif level == 'warning':
            logger.warning(session_message)

    def run(self, stream_response=False):
        self.log_with_session("New session started")
        
        pdf_docs = st.file_uploader("Upload your PDFs here:", type="pdf", accept_multiple_files=True)
        process_files = st.button("Process files")
        if process_files:
            self.log_with_session("Processing PDF files")
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            self.log_with_session("Indexing started")
            PGVector.from_texts(texts=text_chunks,
                                embedding=self.embedding_model,
                                connection_string=self.pg_connection_string)
            self.log_with_session("Indexing completed")
            st.success('Done!')

        st.markdown("""# Add PDFs and ask questions.<br/>""",unsafe_allow_html=True)

        user_input = st.text_input(label='Ssup', placeholder="How can I commit tax fraud?")
        ask_button = st.button("Ask")
        st.markdown("""<br />""", unsafe_allow_html=True)

        if ask_button:
            try:
                self.log_with_session(f"New question received: {user_input}")
                vectorstore = PGVector(
                    connection_string=self.pg_connection_string,
                    embedding_function=self.embedding_model)
                with st.spinner('Processing ...'):
                    resp = self.llm.get_RAG_response(vectorstore=vectorstore, user_prompt=user_input, stream_response=stream_response)
                    st.write(resp)
                    self.log_with_session("Response generated successfully")
            except Exception as e:
                self.log_with_session(f"Error occurred: {str(e)}", level='error')
                st.warning("Something went wrong, Please try again.")
                return


if __name__ == '__main__':
    load_dotenv()
    app = App()
    app.run(stream_response=True)
