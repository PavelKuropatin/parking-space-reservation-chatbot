from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


URER_INPUT_ROOT_CLASSIFICATION_PROMPT_TMPL = ChatPromptTemplate.from_messages([
    ("system", """
You are a chatbot assistant designed to help users find information about parking and to book parking spaces.

Classify the user's message into exactly ONE of these flows:
- information_request   : User asks information about the parking (prices, location, hours, policies, booking process, etc.). 
                          This includes questions like "How many place available?" or "what is the reservation process?"
                          It is only clarifition for user
- reservation   : User is ACTIVELY requesting to make a reservation right now 
                  (e.g. "I want to book a space", "reserve a space for me").
                  Do NOT use this for questions about how reservations work.
Reply only with string flow name: information_request|reservation
"""),
  ("human", "{text}")
])


RAG_SUMMARIZATION_PROMPT_TMPL = ChatPromptTemplate.from_messages([
    ("system", """
You are a parking assistant with access to a FAQ knowledge base and a pricing database.

Instructions:
- Use ONLY the provided context to answer.
- If the answer exists in the context, extract it clearly and concisely.
- If the context does not contain the answer, say: "I don't have that information."
- Do not guess or infer missing details.

RAG CONTEXT:
{rag_context}

FIELD DEFINITIONS (for reference only):
- is_24h = true means "Open 24 hours"
- is_closed = true means "Closed all day"
- opens_at / closes_at are only relevant when is_24h is false

DATABASE CONTEXT:
{db_context}

IMPORTANT FIELD MEANINGS:
- is_24h = true means the location is open 24 hours (all day, no closing time)
- opens_at / closes_at are ignored when is_24h is true
- is_closed = true means the location is closed all day
"""),
 ("human", "Question: {text}")
])


RAG_DATABASE_SYSTEM_PROMPT = """
You are a database reasoning agent. Use the available tools to extract
prices, working hours, and any relevant structured info.
When you have enough information, answer without calling more tools.
"""


RESERVATION_DETAILS_PARSING_PROMT_TMPL = ChatPromptTemplate.from_messages([
    ("system", """You extract parking reservation fields from the user latest message.
                  Resolve relative times ('tomorrow' or 'today') against NOW and output in YYYY-mm-DD HH:mm format.
                  Fill fields the user was actually asked, dont imagine any data.
                  NOW: {now}
                  Fields already collected: {current_details}"""),
    MessagesPlaceholder("history"),
])


GUARDRAIL_INPUT_BLOCK_MSG = """
Your message contains sensitive information
({details}) that cannot be processed.
Please remove it and try again.
"""
GUARDRAIL_OUTPUT_BLOCK_MSG = """
The response was blocked because it contained sensitive information
({details}). Please rephrase your question or contact support.
"""