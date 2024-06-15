# Baddie

Baddie helps you talk to your documents. Ask questions in natural language and get context aware answers. 
I've created this as a demo project to toy around with LLMs on amazon bedrock, planning to evolve it to something useful. Under the hood, it uses
- FM for chat : Meta Llama3-8B on Amazon bedrock (as of now)
- FM for generating embeddings : Amazon Titan Multimodal Embeddings G1
- PostgreSQL and PGvector for storing/retrieving embeddings
And everything tied together with langchain and streamlit.


## Running on local with python
Two steps
1. Create a python3 virtualenv and run.

    ```bash
    pip install -r requirements.txt
    ```
2. Add required values to keys in `.env` file.
3. Finally, run it using below command.

    ```bash
    streamlit run streamlit_app.py --server.port=80 --server.address=0.0.0.0
    ```

## Running on docker-compose
Add right values for `env_file` in `docker-compose.yaml` and then run below command.
```bash
docker-compose up -d
```


# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Few things I have in mind next :
1. Use langchain classes for holding chat context 
2. Vector indexing
3. Session wise doc storage (currently its global)

Other _chunkier_ improvements 
1. Auth and hosting
2. Async ingestion pipeline for docs
3. Support other sources for data

Found some good demos and docs while building this, I'll add a references section for those.
