# financial_index_llm_evaluation

## TOC-guided LangGraph pipeline

The `toc_langgraph_app.py` module assembles a LangGraph workflow that:
- detects table-of-contents pages in a report JSON export,
- ranks TOC entries against a question,
- fetches the most promising sections, and
- asks an LLM whether each page answers the question.

### Quick start

```python
from pathlib import Path

from langchain_openai import ChatOpenAI

from schemas import QuestionSpec
from toc_langgraph_app import (
    TOCSearchConfig,
    build_toc_qa_graph,
    load_report,
)

report = load_report(Path("../../data/datasets/stoxx_600/JSONs/AAK AB1.json"))
question = QuestionSpec(
    id="ad-hoc",
    question="Does the page describe the company's business activities?",
    include=["business model", "company overview"],
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

graph = build_toc_qa_graph(TOCSearchConfig()).compile()
result = graph.invoke({"report": report, "question": question, "llm": llm})
print(result["final_decision"], result["final_summary"])
```

Use `run_toc_qa(report_path, question, llm)` for a convenience wrapper that handles loading and compilation.
