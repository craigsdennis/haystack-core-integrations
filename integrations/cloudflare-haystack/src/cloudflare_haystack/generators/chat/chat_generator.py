import dataclasses
import json
from typing import Callable, List, Dict, Optional

from cloudflare import Cloudflare
from haystack import component
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage, StreamingChunk


class BaseCloudflareGenerator:

    def __init__(
        self,
        account_id: Secret = Secret.from_env_var("CLOUDFLARE_ACCOUNT_ID"),
        api_token: Secret = Secret.from_env_var("CLOUDFLARE_API_TOKEN"),
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        model: str = "@cf/meta/llama-3.1-8b-instruct",
    ):
        self.client = Cloudflare(api_token=api_token.resolve_value())
        self.account_id = account_id.resolve_value()
        self.model = model
        self.streaming_callback = streaming_callback


    def _convert_to_message_json(
        self, messages: List[ChatMessage]
    ) -> List[Dict[str, str]]:
        valid_keys = {"role", "content"}
        converted = []
        for m in messages:
            message_dict = dataclasses.asdict(m)
            filtered_message = {
                k: v for k, v in message_dict.items() if k in valid_keys and v
            }
            converted.append(filtered_message)
        return converted


@component
class CloudflareChatGenerator(BaseCloudflareGenerator):

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage]):
        messages_as_json = self._convert_to_message_json(messages)
        replies: List[ChatMessage] = []
        if self.streaming_callback is None:
            result = self.client.workers.ai.run(
                self.model,
                account_id=self.account_id,
                messages=messages_as_json,
            )
            replies.append(ChatMessage.from_assistant(result["response"]))
        else:
             with self.client.workers.ai.with_streaming_response.run(
                account_id=self.account_id,
                model_name=self.model_name,
                messages=messages_as_json,
                stream=True,
            ) as response:
                # The response is an EventSource object that looks like so
                # data: {"response": "Hello "}
                # data: {"response": ", "}
                # data: {"response": "World!"}
                # data: [DONE]
                # Create a token iterator
                def iter_tokens(r):
                    for line in r.iter_lines():
                        if line.startswith("data: ") and not line.endswith("[DONE]"):
                            entry = json.loads(line.replace("data: ", ""))
                            yield entry["response"]
                
                message_chunks = []
                for token in iter_tokens(response):
                    self.streaming_callback(StreamingChunk(content=token))
                    message_chunks.append(token)
                replies.append(ChatMessage.from_assistant("".join(message_chunks)))
        return {"replies": replies}
