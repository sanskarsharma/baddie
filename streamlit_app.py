import streamlit as st
import os

from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import BedrockEmbeddings
from dotenv import load_dotenv

from libs.helpers import get_pdf_text, get_text_chunks
from llm.llama3_8b import LLama3_8B
from libs import log

logger = log.get_logger(__name__)


class App:

    def __init__(self) -> None:
        self.embedding_model = BedrockEmbeddings(model_id='amazon.titan-embed-image-v1', region_name=os.environ['AWS_REGION'])
        self.pg_connection_string = f'postgresql://postgres:{os.environ["POSTGRES_USER"]}@{os.environ["POSTGRES_PASSWORD"]}:5432/{os.environ["POSTGRES_DB"]}'
        self.llm = LLama3_8B()

    def run(self, stream_response=False):
        
        pdf_docs = st.file_uploader("Upload your PDFs here:", type="pdf", accept_multiple_files=True)
        process_files = st.button("Process files")
        if process_files:
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            logger.info("$$ indexing started")
            PGVector.from_texts(texts=text_chunks,
                                embedding=self.embedding_model,
                                connection_string=self.pg_connection_string)
            logger.info("$$ indexing ENDED")
            st.success('Done!')

        st.markdown("""# Add PDFs and ask questions.<br/>""",unsafe_allow_html=True)

        user_input = st.text_input(label='Ssup', placeholder="How can I commit tax fraud?")
        ask_button = st.button("Ask")
        st.markdown("""<br />""", unsafe_allow_html=True)

        if ask_button:
            try:
                # resp = self.llm.get_response(user_prompt=user_input, stream_response=stream_response)
                vectorstore = PGVector(
                    connection_string=self.pg_connection_string,
                    embedding_function=self.embedding_model)
                with st.spinner('Processing ...'):
                    resp = self.llm.get_RAG_response(vectorstore=vectorstore, user_prompt=user_input, stream_response=stream_response)
                    st.write(resp)
            except Exception as e:
                logger.error("Error", exc_info=True)
                st.warning("Something went wrong, Please try again.")
                return


if __name__ == '__main__':
    load_dotenv()
    app = App()
    app.run(stream_response=True)
