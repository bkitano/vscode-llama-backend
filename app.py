from fastapi import FastAPI
import openai
import asyncio

app = FastAPI()

SYSTEM_PROMPT = f"""INSTRUCTIONS: 
                The user asks the assistant about a health related question, 
                the assistant provides a response, which is a rich response that has a direct answer, 
                as well as recommended topics to read about and actions to take. The assistant is
                also provided with information about the user's health and wellness preferences,
                including their health goals, and their current health status. The assistant should
                use this information to provide a response that is tailored to the user's needs. 

                In addition, the assistant should also specifically look for personal details that
                can be inferred from the user's question, and return those as well. For example, if
                the user asks a question about being an athlete, the assistant should return the
                personal detail 'is an athlete'. Similarly, if the user asks a question about a chronic
                condition like diabetes, the assistant should return the personal detail 'may have diabetes'.

                Only use the functions you have been provided with, and only output valid JSON.
                """

SYSTEM_MESSAGE = {
    "role": "system",
    "content": SYSTEM_PROMPT,
}


def format_user_question(user_question):
    return f"""
                USER QUESTION: {user_question}
                USER PREFERENCES: []""".strip()

def get_response_completion(user_question, chat_history=[], stream=False, n=1):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        stream=stream,
        n=n,
        functions=[CratesMessage.openai_schema],
        function_call={"name": CratesMessage.openai_schema["name"]},
        messages=[SYSTEM_MESSAGE]
        + chat_history
        + [
            {"role": "user", "content": format_user_question(user_question)},
        ],
    )

def get_response_endpoint(chat_payload, streaming=False):
    user_question = chat_payload[-1]["content"]
    chat_history = []
    for chat_item in chat_payload[:-1]:
        if chat_item["role"] == "user":
            message = {
                "role": "user",
                "content": format_user_question(chat_item["content"]),
            }
            chat_history.append(message)
        else:  # role is crates
            try:
                formatted_function_response = CratesMessage(
                    **chat_item
                ).model_dump_json()
                message = {
                    "role": "function",
                    "name": CratesMessage.openai_schema["name"],
                    "content": formatted_function_response,
                }
                chat_history.append(message)
            except:
                print("failed to parse function response")
                print(chat_item)
                message = {
                    "role": "assistant",
                    "content": chat_item["content"],
                }
                chat_history.append(message)

    if streaming:

        def delta_stream():
            completion = get_response_completion(
                user_question, chat_history=chat_history, stream=True
            )

            for chunk in completion:
                if "function_call" in chunk["choices"][0]["delta"]:
                    if "arguments" in chunk["choices"][0]["delta"]["function_call"]:
                        delta = chunk["choices"][0]["delta"]["function_call"][
                            "arguments"
                        ]
                        # yield delta
                        yield bytes(delta, "utf-8")

        # it may be the fact that the deta_stream generator is only created
        # once, and so the delta_stream itself doesn't have enough info inside it.
        def stream_json():
            stream = json_stream.load(delta_stream(), persistent=True)
            g = json_stream_generator(stream)
            while True:
                try:
                    yield bytes(next(g), "utf-8")
                except StopIteration:
                    return

        return stream_json