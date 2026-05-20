from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

client = OpenAI()

tools=[
    {"type":"function",
    "function":{
        "name":"get_weather",
        "description":"Get the weather in a city",
        "parameters": {
    "type": "object",
    "properties": {
        "city": {"type": "string"}
    },
    "required": ["city"]
}}},
    {"type":"function",
    "function":{
        "name":"get_time",
        "description":"Get the time in a city",
        "parameters": {
    "type": "object",
    "properties": {
        "city": {"type": "string"}
    },
    "required": ["city"]
}}},
    {"type":"function",
    "function":{
        "name":"convert_currency",
        "description":"Convert an amount from one currency to another",
        "parameters": {
    "type": "object",
    "properties": {
        "amount": {"type": "number"},
        "from_currency": {"type": "string"},
        "to_currency": {"type": "string"}
    },
    "required": ["amount", "from_currency", "to_currency"]
}}}
]


def get_weather(city: str) -> str:
    return f"Il fait 22°C et ensoleillé à {city}."

def get_time(city: str) -> str:
    return f"Il est 14h32 à {city}."

def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    return f"{amount} {from_currency} est équivalent à {amount * 3.9} {to_currency}."

messages = [
    {"role": "user", "content": "Quel temps fait-il à Tel Aviv, et combien font 100 EUR en ILS ?"}
]

tool_functions={
    "get_weather": get_weather,
    "get_time": get_time,
    "convert_currency": convert_currency
}

while True:

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    messages.append(response.choices[0].message)
    
    finish_reason = response.choices[0].finish_reason
    if finish_reason == "stop":
        print(response.choices[0].message.content)
        break

    if finish_reason == "tool_calls":

        for tool_call in response.choices[0].message.tool_calls:
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments

            if function_name in tool_functions:
                args_dict = json.loads(function_args)
                function_response = tool_functions[function_name](**args_dict)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": function_response})
            else:
                break


print(messages)
