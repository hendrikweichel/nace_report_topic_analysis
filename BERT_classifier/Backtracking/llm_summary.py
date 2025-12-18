from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

def summarize_text(
    text: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> str:
    """
    Summarize a given text using LangChain + OpenAI (no prompt templates).

    Args:
        text: Input text to summarize
        model: OpenAI chat model
        temperature: Sampling temperature

    Returns:
        Concise summary as a string
    """

    if text.strip() == "":
        return ""

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
    )

    messages = [
        SystemMessage(
            content="You summarize texts concisely without using additional information."
        ),
        HumanMessage(
            content=f"Please summarize the business model in 4 sentences:\n\n{text}"
        ),
    ]

    response = llm.invoke(messages)
    return response.content

    
if __name__ == "__main__":

    summary = summarize_text("## CHAPTER 1.0 INTRODUCTION ## 1.1 ABOUT FRONTKEN GROUP We build technology and provide services that enable our customers to be more sustainable and do more for our environment, community, and society. We integrate our technology, business practices, partnerships, supply chain and production processes around a single mission - to build sustainability through actionable technology and make more positive impact towards the environment and society together with our customers, employees and shareholders and stakeholders. Our foundation is built on our core values, which distinguish us and guide our actions and the way we conduct our business in a socially responsible and ethical manner. We are committed to delivering value to all our stakeholders including customers, employees and shareholders through sustaining growth in our businesses, protecting the environment, empowering lives of people and nurturing communities where we operate. We will also continue to build the company on the foundation of: - (a) Responsible management; - (b) Responsible innovation and service; - (c) Responsible green production; - (d) Responsible workplace; - (e) Responsible inclusion and diversity; - (f) Responsible supply chain; - (g) Responsible Climate Change. We want to make it easy to be more sustainable, by building technology and providing services including training to help people to better understand their impact and actions. <!-- image -->")
    print(summary)