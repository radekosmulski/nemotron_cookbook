function GetCompletion(message: string) -> string {
    client "openai/gpt-5-nano"
    prompt #"
        {{ _.role('user') }}
        {{ message }}
    "#
}

class LibrariesAndLanguages {
    used_libraries string[]
    supported_languages string[]
}

function ExtractLibrariesAndLanguages(description: string) -> LibrariesAndLanguages {
    client "openai/gpt-5-nano"
    prompt #"
        {{ _.role('user') }}
        The following is a description of a large language model:

        ''' 
        {{ description }}
        '''

        Extract the following information from the description:
        1. The libraries the model was trained on.
        2. The supported languages.

        {{ ctx.output_format }}
    "#
}

class MyUserMessage {
  role "user" | "assistant"
  content string
}
function ChatWithLLM(messages: MyUserMessage[]) -> string {
  client "openai/gpt-5-nano"
  prompt #"
    Answer the user's questions based on the chat history:
    {% for message in messages %}
      {{ _.role(message.role) }} 
      {{ message.content }}
    {% endfor %}
    Answer:
  "#
}

test TestName {
  functions [ChatWithLLM]
  args {
    messages [
      {
        role "user"
        content "Hello!"
      }
      {
        role "assistant"
        content "Hi!"
      }
    ]
  }
}

class GetWeatherAPI {
    location string
}

class Message {
    content string
}

function ToolCallOrMessage(message: string) -> Message | GetWeatherAPI {
  client "openai/gpt-5-nano"
  prompt #"
    {{ _.role('user') }}
    You are an agent that can either answer the user's question directly or call a tool to get more information.

    Here is the message from the user:
    {{ message }}

    Decide whether to answer the user's question directly or call the tool.

    {{ ctx.output_format }}
  "#
}