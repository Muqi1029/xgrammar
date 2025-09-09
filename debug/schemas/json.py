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
                "required": ["time"],
            },
        },
    }
]
