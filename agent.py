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
}
    }}

]


def get_weather(city: str) -> str:
    return f"Il fait 22°C et ensoleillé à {city}."

messages = [
    {"role": "user", "content": "Quel temps fait-il à Paris ?"}
]

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
            if function_name == "get_weather":
                args_dict = json.loads(function_args)
                function_response = get_weather(args_dict["city"])
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": function_response})
            else:
                break
    

print(messages)
