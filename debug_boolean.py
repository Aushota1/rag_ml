from dotenv import load_dotenv
load_dotenv('.env')
from config import config
from indexer import HybridIndexer
from reranker import Reranker

indexer = HybridIndexer(config.EMBEDDING_MODEL, config.INDEX_PATH)
indexer.load_index()
reranker = Reranker(config.RERANKER_MODEL)

question = "Was the main claim approved by the court?"
print(f"Question: {question}")
print(f"RELEVANCE_THRESHOLD: {config.RELEVANCE_THRESHOLD}")
print(f"RELEVANCE_CLASSIFIER_THRESHOLD: {config.RELEVANCE_CLASSIFIER_THRESHOLD}")
print()

results = indexer.hybrid_search(question, 20)
print(f"Hybrid search: {len(results)} candidates")

reranked = reranker.rerank(question, results, 10)
print(f"After rerank: {len(reranked)} chunks")
print()
for r in reranked[:5]:
    score = r['rerank_score']
    text = r['text'][:120].replace('\n', ' ')
    print(f"  score={score:.4f} | {text}")

print()
if reranked:
    max_score = reranked[0]['rerank_score']
    print(f"Max score: {max_score:.4f}")
    print(f"Passes threshold ({config.RELEVANCE_THRESHOLD}): {max_score >= config.RELEVANCE_THRESHOLD}")
else:
    print("No chunks after rerank!")
