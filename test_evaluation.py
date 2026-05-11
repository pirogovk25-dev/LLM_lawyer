"""
Evaluation: Baseline RAG vs Enhanced RAG
(Query Rewriting + Multi-Query Retrieval)

Запуск:
  cd "c:\\Users\\asus\\Desktop\\LLM law"
  python test_evaluation.py
"""
import sys
import os
import io

# UTF-8 вывод на Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Обход прокси для localhost — иначе qdrant-client и ollama идут через 10809
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Патч transformers ОБЯЗАТЕЛЬНО до импорта FlagEmbedding
from core.patch import apply_transformers_patch
apply_transformers_patch()

import json
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from langchain_openai import ChatOpenAI

# ── LLM: используем Ollama (OpenAI-совместимый API на порту 11434) ────────────
print("Connecting to Ollama (qwen2.5:7b)...")
ollama_llm = ChatOpenAI(
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="ollama",
    model_name="qwen2.5:7b",
    temperature=0,
    max_tokens=2048,
)

# Патчим llm_client ДО импорта graph — get_llm() вернёт наш Ollama-клиент
import services.llm_client as llm_module
llm_module.llm = ollama_llm

# ── Загрузка моделей и графа ─────────────────────────────────────────────────
print("Loading BGE-M3 embedding model (may take 1-2 min)...")
from services.search_service import ultimate_hybrid_search, multi_query_hybrid_search
print("Loading graph pipeline...")
from workflow.graph import app_workflow
import config

print("All models loaded.\n")

# ── Датасеты ─────────────────────────────────────────────────────────────────
HIT_RATE_DATASET = [
    {
        "question": "Какой максимальный срок может длиться проверка организации в обычных условиях?",
        "expected_source": "Закон Российской Федерации от 21.07.1993 №5485-1.docx"
    },
    {
        "question": "Какую информацию, касающуюся экологии и здравоохранения, закон запрещает засекречивать?",
        "expected_source": "Закон Российской Федерации от 21.07.1993 №5485-1.docx"
    },
    {
        "question": "Что является основанием для прекращения допуска сотрудника к гостайне по статье 23?",
        "expected_source": "Закон Российской Федерации от 21.07.1993 №5485-1.docx"
    },
    {
        "question": "Куда может обратиться иностранная некоммерческая организация для обжалования действий госорганов?",
        "expected_source": "Федеральный закон от 12.01.1996 №7-ФЗ"
    },
    {
        "question": "Как осуществляется мониторинг в сфере отношений с соотечественниками за рубежом согласно статье 25?",
        "expected_source": "Федеральный закон от 24.05.1999 №99-ФЗ"
    },
]

with open('ground_truth_dataset.json', 'r', encoding='utf-8') as f:
    RAGAS_DATASET = json.load(f)


# ── Part 1: Baseline Hit Rate / MRR ──────────────────────────────────────────
print("=" * 60)
print("PART 1: BASELINE (прямой поиск, без LLM улучшений)")
print("=" * 60)

baseline_hits = 0
baseline_mrr = 0.0
for item in tqdm(HIT_RATE_DATASET, desc="Baseline search"):
    results = ultimate_hybrid_search(item["question"], final_limit=3)
    for rank, chunk in enumerate(results):
        if item["expected_source"] in chunk.payload.get('source', ''):
            baseline_hits += 1
            baseline_mrr += 1.0 / (rank + 1)
            break

baseline_hr = baseline_hits / len(HIT_RATE_DATASET) * 100
baseline_mrr_val = baseline_mrr / len(HIT_RATE_DATASET)
print(f"\nBaseline  ->  Hit Rate: {baseline_hr:.1f}%  |  MRR: {baseline_mrr_val:.3f}\n")


# ── Part 2: Enhanced Hit Rate / MRR (полный граф) ────────────────────────────
print("=" * 60)
print("PART 2: ENHANCED (rewrite -> multi-query -> retrieve)")
print("=" * 60)

enhanced_hits = 0
enhanced_mrr = 0.0

for item in tqdm(HIT_RATE_DATASET, desc="Enhanced search"):
    try:
        state = app_workflow.invoke({"question": item["question"]})
        docs = state.get("documents", [])
        rewritten = state.get("rewritten_question", "")
        expanded = state.get("expanded_queries", [])

        print(f"\n  Q: {item['question'][:70]}")
        print(f"  Rewritten: {rewritten[:80]}" if rewritten else "  Rewritten: (fallback)")
        print(f"  Expanded:  {expanded}" if expanded else "  Expanded:  (fallback)")

        for rank, doc in enumerate(docs):
            if item["expected_source"] in doc:
                enhanced_hits += 1
                enhanced_mrr += 1.0 / (rank + 1)
                print(f"  HIT at rank {rank + 1}")
                break
        else:
            print("  MISS")
    except Exception as e:
        print(f"  ERROR: {e}")

enhanced_hr = enhanced_hits / len(HIT_RATE_DATASET) * 100
enhanced_mrr_val = enhanced_mrr / len(HIT_RATE_DATASET)
print(f"\nEnhanced  ->  Hit Rate: {enhanced_hr:.1f}%  |  MRR: {enhanced_mrr_val:.3f}\n")


# ── Part 3: Ragas (Enhanced pipeline, 10 вопросов) ───────────────────────────
print("=" * 60)
print("PART 3: RAGAS EVALUATION (10 вопросов, Enhanced pipeline)")
print("=" * 60)

RESULTS_CACHE = "eval_results_cache.json"
results = []

if os.path.exists(RESULTS_CACHE):
    print("Loading cached answers from eval_results_cache.json...")
    with open(RESULTS_CACHE, 'r', encoding='utf-8') as f:
        results = json.load(f)
else:
    for entry in tqdm(RAGAS_DATASET, desc="Generating answers"):
        try:
            response = app_workflow.invoke({"question": entry["question"]})
            answer = response.get("generation", "")
            if hasattr(answer, "content"):
                answer = answer.content
            results.append({
                "question": entry["question"],
                "answer": str(answer),
                "contexts": response.get("documents", []),
                "ground_truth": entry.get("ground_truth", "")
            })
        except Exception as e:
            print(f"  ERROR for '{entry['question'][:50]}': {e}")
    with open(RESULTS_CACHE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings

print("\nLoading HuggingFace embeddings for Ragas...")
hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

dataset = Dataset.from_list(results)
ragas_llm = LangchainLLMWrapper(ollama_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
run_config = RunConfig(max_workers=1, timeout=1600)

print("Running Ragas evaluation...")
score = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, answer_correctness],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=run_config,
    raise_exceptions=False
)

# ── Итоговое сравнение ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ИТОГОВОЕ СРАВНЕНИЕ")
print("=" * 60)
df = score.to_pandas()
df.to_csv("rag_evaluation_enhanced_report.csv", index=False, encoding='utf-8-sig')

# Извлекаем средние значения — в новых версиях Ragas возвращает список
def mean_score(val):
    if isinstance(val, list):
        valid = [v for v in val if v is not None]
        return sum(valid) / len(valid) if valid else 0.0
    return float(val) if val is not None else 0.0

faith   = mean_score(score['faithfulness'])
relev   = mean_score(score['answer_relevancy'])
correct = mean_score(score['answer_correctness'])

new_hr_str      = f"{enhanced_hr:.1f}%"
new_mrr_str     = f"{enhanced_mrr_val:.3f}"
new_faith_str   = f"{faith:.4f}"
new_relev_str   = f"{relev:.4f}"
new_correct_str = f"{correct:.4f}"

print(f"\n{'Metric':<30} {'Previous':>15} {'Enhanced':>15}")
print("-" * 62)
print(f"{'Hit Rate':.<30} {'100.0%':>15} {new_hr_str:>15}")
print(f"{'MRR':.<30} {'1.000':>15} {new_mrr_str:>15}")
print(f"{'Faithfulness':.<30} {'1.0000':>15} {new_faith_str:>15}")
print(f"{'Answer Relevancy':.<30} {'0.6895':>15} {new_relev_str:>15}")
print(f"{'Answer Correctness':.<30} {'0.3557':>15} {new_correct_str:>15}")
print("\nReport saved: rag_evaluation_enhanced_report.csv")
