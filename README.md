# local-rag-project

A project I made to learn more about using AI through code, and to expand my knowledge of LLMs and what they can do. 

## Usage

**Getting started**

Once you have cloned or forked this repo as a local copy onto your machine, run `python3 -m venv .venv`, then `source .venv/bin/activate` on Mac, or just `venv/bin/activate` on Windows to activate the virtual environment.
Once you have the venv activated, run `pip install -r requirements.txt` to download the required packages.


- To run the program and interact with the stored files, just run `python3 query.py`. 

- To update the database, add whatever txt files you want to the docs directory, and run `python3 populate_database.py`

- To reset the database, run `python3 populate_database.py --reset`, using the `--reset` flag to reset. 