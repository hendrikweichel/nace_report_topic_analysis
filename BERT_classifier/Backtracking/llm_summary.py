from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

def summarize_text(
    text: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> str:
    """
    Summarize a given text using LangChain + OpenAI.

    Args:
        text: Input text to summarize
        model: OpenAI chat model
        temperature: Sampling temperature

    Returns:
        Concise summary as a string
    """

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You summarize texts concisely without using additional information."),
        ("human", "Please summarize the business model in 4 sentences: :\n\n{text}")
    ])

    chain = prompt | llm

    response = chain.invoke({"text": text})
    return response.content

if __name__ == "__main__":

    summary = summarize_text("MMS  ANNUAL REPORT 2022 B ## Annual General Meeting The Annual General Meeting of the members of McMillan Shakespeare Limited A.B.N. 74 107 233 983 will be held virtually and in person on 28 October 2022 at 10.00am. Please refer to the AGM notice for further details. mmsg.com.au The McMillan Shakespeare Group is a provider of salary packaging, novated leasing, disability plan management and support co-ordination, asset management and related financial products and services. Through its subsidiaries, it offers a breadth of services and expertise, designed to responsibly deliver longterm value to its customers. The Group employs a highly committed team of c.1,300 people across Australia, New Zealand and the United Kingdom and domestically manages programs for some of the largest public sector, corporate and charitable organisations. Header <!-- image --> SUBHEADER <!-- image --> <!-- image --> <!-- image --> <!-- image --> <!-- image --> <!-- image --> <!-- image --> <!-- image -->")
    print(summary)