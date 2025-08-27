import os, glob, chromadb
from chromadb.utils import embedding_functions

def ingest(kb_dir:str, persist_dir:str):
    client = chromadb.PersistentClient(path=persist_dir)
    coll = client.get_or_create_collection(
        "knowledgebase", embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    docs, ids = [], []
    for path in glob.glob(os.path.join(kb_dir, "**/*.*"), recursive=True):
        if "chroma" in path: continue
        try:
            txt = open(path,"r",encoding="utf-8",errors="ignore").read()
        except Exception:
            continue
        docs.append(txt[:4000])              # small chunk per file
        ids.append(os.path.basename(path))
    if docs:
        existing = set(coll.get()["ids"])
        new_docs = [d for i,d in enumerate(docs) if ids[i] not in existing]
        new_ids  = [i for i in ids if i not in existing]
        if new_docs:
            coll.add(documents=new_docs, ids=new_ids)
    return True

def search(query:str, persist_dir:str, k:int=4):
    client = chromadb.PersistentClient(path=persist_dir)
    coll = client.get_or_create_collection(
        "knowledgebase", embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    res = coll.query(query_texts=[query], n_results=k)
    docs = res["documents"][0] if res["documents"] else []
    return "\n\n".join(docs[:k])
