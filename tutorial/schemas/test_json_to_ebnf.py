import json

from xgrammar import Grammar

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time",
            "parameters": {
                # json schema here
                "type": "object",
                "properties": {
                    "time": {"type": "string", "description": "The current time", "format": "date"},
                    "timezone": {"type": "string", "description": "The timezone"},
                },
            },
            "strict": True,
        },
    }
]
if __name__ == "__main__":
    print(" ENBF ".center(80, "="))
    ebnf = Grammar.from_json_schema(
        json.dumps(tools[0]["function"]["parameters"]), print_converted_ebnf=True
    )
