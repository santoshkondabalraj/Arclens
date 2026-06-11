from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

_SYSTEM_TEMPLATE = """\
You are a precise research assistant answering questions about YouTube podcast content.

RULES:
1. Answer ONLY using information present in the CONTEXT passages below.
2. Do NOT use any prior knowledge, training data, or external information.
3. Every factual claim MUST be followed immediately by an inline citation in this \
exact format: [Source: "{{title}}", {{channel}}, ~{{timestamp}}s]
   - Use the exact episode title, channel name, and nearest timestamp from the source metadata.
4. Write in clear, concise prose. Do not pad the answer.
5. If the context is only partially relevant, answer what you can from it and note any gaps \
in plain language — do NOT refuse or output structured data.

CONTEXT:
{context}
"""

_HUMAN_TEMPLATE = "Question: {question}"


def build_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(_SYSTEM_TEMPLATE),
            HumanMessagePromptTemplate.from_template(_HUMAN_TEMPLATE),
        ]
    )
