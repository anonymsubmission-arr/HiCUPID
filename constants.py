HF_MODELS = [
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Qwen2.5-7B-Instruct",
]

OPENAI_MODELS = [
    "GPT-4o-mini-2024-07-18",
]

EVAL_MODELS = [
    "GPT-4o-2024-08-06",
    "Llama-3.2-3B-HiCUPID",
]

MODEL_NAME = {
    "gpt-4o": "GPT-4o",
    "gpt-4o-2024-08-06": "GPT-4o-2024-08-06",
    "gpt-4o-2024-05-13": "GPT-4o-2024-05-13",
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-4o-mini-2024-07-18": "GPT-4o-mini-2024-07-18",
}


def model_name(model_id):
    return MODEL_NAME.get(model_id, model_id.split("/")[-1])


DIALOGUE_SPLIT = {
    "test_1": "train",
    "test_2": "test",
}

PROMPT_MESSAGES = {
    "vanilla_v1": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer within a few sentences.",
        },
    ],
    "vanilla_v2": [
        {
            "role": "user",
            "content": "You are a helpful assistant. Answer within a few sentences.",
        },
        {
            "role": "assistant",
            "content": "Of course! How can I assist you today?",
        },
    ],
    "zero_v1": [
        {
            "role": "system",
            "content": "You are a personalized assistant for the user. When answering the user's questions, consider the user's characteristics based on the previous conversation history. Answer within a few sentences.",
        },
    ],
    "zero_v2": [
        {
            "role": "user",
            "content": "You are a personalized assistant for the user. When answering the user's questions, consider the user's characteristics based on the previous conversation history. Answer within a few sentences.",
        },
        {
            "role": "assistant",
            "content": "Of course! How can I assist you today?",
        },
    ],
    "few_v1": [
        {
            "role": "system",
            "content": """You are a personalized assistant for the user. When answering the user's questions, consider the user's characteristics based on the previous conversation history. Below are examples of personalized answers.

[Example 1]
Question: What martial art would you recommend for someone looking to improve their fitness?
Personalized answer: Given that you don't enjoy Muay Thai, you might want to try Brazilian Jiu-Jitsu. It's a great workout and focuses on grappling, which can be a fun and effective way to improve fitness without the striking aspect of Muay Thai.
General answer: I would recommend Muay Thai as it's an excellent full-body workout that improves strength, endurance, and flexibility.

[Example 2]
Question: What type of workout shoes would you recommend?
Personalized answer: For someone who does HIIT, I recommend shoes that offer excellent support and cushioning, with a focus on stability and flexibility. Look for cross-training shoes that can handle lateral movements and provide good grip for quick changes in direction.
General answer: Generally, it's best to choose workout shoes that are comfortable, fit well, and are appropriate for the type of exercise you plan to do, whether it's running, weightlifting, or general gym workouts.

Like the personalized answers in the examples above, consider the user's characteristics based on the previous conversation history when answering the user's questions. Answer within a few sentences.""",
        },
    ],
    "few_v2": [
        {
            "role": "user",
            "content": """You are a personalized assistant for the user. When answering the user's questions, consider the user's characteristics based on the previous conversation history. Below are examples of personalized answers.

[Example 1]
Question: What martial art would you recommend for someone looking to improve their fitness?
Personalized answer: Given that you don't enjoy Muay Thai, you might want to try Brazilian Jiu-Jitsu. It's a great workout and focuses on grappling, which can be a fun and effective way to improve fitness without the striking aspect of Muay Thai.
General answer: I would recommend Muay Thai as it's an excellent full-body workout that improves strength, endurance, and flexibility.

[Example 2]
Question: What type of workout shoes would you recommend?
Personalized answer: For someone who does HIIT, I recommend shoes that offer excellent support and cushioning, with a focus on stability and flexibility. Look for cross-training shoes that can handle lateral movements and provide good grip for quick changes in direction.
General answer: Generally, it's best to choose workout shoes that are comfortable, fit well, and are appropriate for the type of exercise you plan to do, whether it's running, weightlifting, or general gym workouts.

Like the personalized answers in the examples above, consider the user's characteristics based on the previous conversation history when answering the user's questions. Answer within a few sentences.""",
        },
        {
            "role": "assistant",
            "content": "Of course! How can I assist you today?",
        },
    ],
}

METADATA = "{idx}. The user {verb} {entity}."

EVAL_MESSAGES = {
    "a": [
        {
            "role": "user",
            "content": """The user asked the assistant a question, and the assistant provided two different answers. The user wants to evaluate the two answers.

[Question]
{question}

[Answer (a)]
{model_answer}

[Answer (b)]
{personalized_answer}

[User's Characteristics]
{metadata}

[Evaluation Criteria]
1. Personalization: Does the answer consider the [User's Characteristics] well?
2. Logical Validity: Is the answer logically valid as a response to the question?

For both answers, provide comments within a few sentences based on each evaluation criterion. After the evaluation, specify the better response by writing 'Better Response:' followed by 'Answer (a)' or 'Answer (b)'. If both answers are equally good, write 'Tie' after 'Better Response:'. Do not use bold font.""",
        },
    ],
    "b": [
        {
            "role": "user",
            "content": """The user asked the assistant a question, and the assistant provided two different answers. The user wants to evaluate the two answers.

[Question]
{question}

[Answer (a)]
{personalized_answer}

[Answer (b)]
{model_answer}

[User's Characteristics]
{metadata}

[Evaluation Criteria]
1. Personalization: Does the answer consider the [User's Characteristics] well?
2. Logical Validity: Is the answer logically valid as a response to the question?

For both answers, provide comments within a few sentences based on each evaluation criterion. After the evaluation, specify the better response by writing 'Better Response:' followed by 'Answer (a)' or 'Answer (b)'. If both answers are equally good, write 'Tie' after 'Better Response:'. Do not use bold font.""",
        },
    ],
}

MISTRAL_CHAT_TEMPLATE = """{%- if messages[0]["role"] == "system" %}
    {%- set system_message = messages[0]["content"] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}
{%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}

{#- This block checks for alternating user/assistant messages, skipping tool calling messages #}
{%- set ns = namespace() %}
{%- set ns.index = 0 %}
{%- for message in loop_messages %}
    {%- if not (message.role == "tool" or message.role == "tool_results" or (message.tool_calls is defined and message.tool_calls is not none)) %}
        {%- if (message["role"] == "user") != (ns.index % 2 == 0) %}
            {{- raise_exception("After the optional system message, conversation roles must alternate user/assistant/user/assistant/...") }}
        {%- endif %}
        {%- set ns.index = ns.index + 1 %}
    {%- endif %}
{%- endfor %}

{{- bos_token }}
{%- for message in loop_messages %}
    {%- if message["role"] == "user" %}
        {%- if tools is not none and (message == user_messages[-1]) %}
            {{- "[AVAILABLE_TOOLS] [" }}
            {%- for tool in tools %}
                {%- set tool = tool.function %}
                {{- '{"type": "function", "function": {' }}
                {%- for key, val in tool.items() if key != "return" %}
                    {%- if val is string %}
                        {{- '"' + key + '": "' + val + '"' }}
                    {%- else %}
                        {{- '"' + key + '": ' + val|tojson }}
                    {%- endif %}
                    {%- if not loop.last %}
                        {{- ", " }}
                    {%- endif %}
                {%- endfor %}
                {{- "}}" }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- else %}
                    {{- "]" }}
                {%- endif %}
            {%- endfor %}
            {{- "[/AVAILABLE_TOOLS]" }}
            {%- endif %}
        {%- if (message == user_messages[-1]) and system_message is defined %}
            {{- "[INST] " + system_message + "\n\n" + message["content"] + "[/INST]" }}
        {%- else %}
            {{- "[INST] " + message["content"] + "[/INST]" }}
        {%- endif %}
    {%- elif message.tool_calls is defined and message.tool_calls is not none %}
        {{- "[TOOL_CALLS] [" }}
        {%- for tool_call in message.tool_calls %}
            {%- set out = tool_call.function|tojson %}
            {{- out[:-1] }}
            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}
                {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
            {%- endif %}
            {{- ', "id": "' + tool_call.id + '"}' }}
            {%- if not loop.last %}
                {{- ", " }}
            {%- else %}
                {{- "]" + eos_token }}
            {%- endif %}
        {%- endfor %}
    {%- elif message["role"] == "assistant" %}
        {{- " " + message["content"]|trim + eos_token}}
    {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}
        {%- if message.content is defined and message.content.content is defined %}
            {%- set content = message.content.content %}
        {%- else %}
            {%- set content = message.content %}
        {%- endif %}
        {{- '[TOOL_RESULTS] {"content": ' + content|string + ", " }}
        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}
            {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
        {%- endif %}
        {{- '"call_id": "' + message.tool_call_id + '"}[/TOOL_RESULTS]' }}
    {%- else %}
        {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}
    {%- endif %}
{%- endfor %}"""
