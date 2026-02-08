import asyncio
import aiohttp
import os
from collections import deque

# Retry when rate-limited (429) or server busy (503)
MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 2

# API key from file (same path as deepseek-ai.py)
directory = "/home/tien/"
filename = "api-openrouter.txt"
API_PATH = os.path.join(directory, filename)

# Load API key from file
def load_api_key(filepath):
    with open(filepath, "r") as f:
        return f.read().strip()

# Set API key and model
API_KEY = load_api_key(API_PATH)
DEEPSEEK_MODEL = "tngtech/deepseek-r1t2-chimera:free"

# Class for DeepSeek chat application
class DeepSeekChatApp:
    def __init__(self):
        # Initialize session and conversation history
        self.session = None
        # Conversation history is a queue of max length 30
        self.conversation_history = deque(maxlen=30)#
        # Semaphore is a synchronization primitive that limits the number of concurrent operations
        self.semaphore = asyncio.Semaphore(10)
    
    # Initialize session
    async def init_session(self):
        self.session = aiohttp.ClientSession()

    # Close session
    async def close_session(self):
        await self.session.close()

    # Add to conversation history
    def add_to_conversation_history(self, role, content):
        self.conversation_history.append({"role": role, "content": content})

    async def get_chat_completion(self, user_input):
        async with self.semaphore:
            self.add_to_conversation_history("user", user_input)
            api_url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            }
            data = {
                "model": DEEPSEEK_MODEL,
                "messages": list(self.conversation_history),
            }
            last_error = None
            for attempt in range(MAX_RETRIES):
                try:
                    async with self.session.post(
                        url=api_url, headers=headers, json=data
                    ) as response:
                        if response.status in (429, 503):
                            await response.read()  # drain so connection can be reused
                            wait_sec = INITIAL_BACKOFF_SEC * (2 ** attempt)
                            if attempt < MAX_RETRIES - 1:
                                print(
                                    f"Rate limited ({response.status}). Retrying in {wait_sec}s..."
                                )
                                await asyncio.sleep(wait_sec)
                                continue
                            response.raise_for_status()
                        response.raise_for_status()
                        api_result = await response.json()
                        message = api_result.get("choices", [{}])[0].get(
                            "message", {}
                        )
                        content = message.get("content", "")
                        self.add_to_conversation_history("assistant", content)
                        return content
                except aiohttp.ClientResponseError as e:
                    last_error = e
                    if e.status in (429, 503) and attempt < MAX_RETRIES - 1:
                        wait_sec = INITIAL_BACKOFF_SEC * (2 ** attempt)
                        print(f"Rate limited ({e.status}). Retrying in {wait_sec}s...")
                        await asyncio.sleep(wait_sec)
                        continue
                    raise
            if last_error is not None:
                raise last_error

    async def main(self):
        print("DeepSeek chat (continuous conversation with context)")
        print(f"Model: {DEEPSEEK_MODEL}")
        await self.init_session()

        try:
            while True:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting. Goodbye!")
                    break
                print("Processing...")
                assistant_response = await self.get_chat_completion(user_input)
                print(f"Assistant: {assistant_response}\n")
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
        finally:
            await self.close_session()


if __name__ == "__main__":
    asyncio.run(DeepSeekChatApp().main())
