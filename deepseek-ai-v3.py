import asyncio
import aiohttp
import json
import os
from collections import deque

# Retry when rate-limited (429) or server busy (503)
MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 2

# API key from file (same path as deepseek-ai.py)
directory = "/home/tien/"
filename = "api-openrouter.txt"
API_PATH = os.path.join(directory, filename)


def load_api_key(filepath):
    with open(filepath, "r") as f:
        return f.read().strip()


API_KEY = load_api_key(API_PATH)
DEEPSEEK_MODEL = "tngtech/deepseek-r1t2-chimera:free"


class DeepSeekChatApp:
    def __init__(self):
        self.session = None
        self.conversation_history = deque(maxlen=30)
        self.semaphore = asyncio.Semaphore(10)

    async def init_session(self):
        self.session = aiohttp.ClientSession()

    async def close_session(self):
        await self.session.close()

    def add_to_conversation_history(self, role, content):
        self.conversation_history.append({"role": role, "content": content})

    def load_conversation_history(self, filepath):
        """Load history from a JSON array file or JSONL file (one JSON object per line)."""
        path = os.path.expanduser(filepath)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        messages = []
        if not raw:
            return
        # Try JSON array first
        if raw.startswith("["):
            messages = json.loads(raw)
        else:
            # Try JSONL (one {"role":..., "content":...} per line)
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                self.conversation_history.append(
                    {"role": msg["role"], "content": str(msg["content"])}
                )

    def save_conversation_history(self, filepath):
        """Save history to a JSON file (array of {role, content})."""
        path = os.path.expanduser(filepath)
        data = list(self.conversation_history)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

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
        load_path = input(
            "Load previous conversation? (path to .json/.jsonl file or Enter for new): "
        ).strip()
        if load_path:
            try:
                self.load_conversation_history(load_path)
                n = len(self.conversation_history)
                print(f"Loaded {n} message(s) from {load_path}")
            except FileNotFoundError:
                print(f"File not found: {load_path}. Starting new conversation.")
            except (json.JSONDecodeError, OSError) as e:
                print(f"Could not load: {e}. Starting new conversation.")
        await self.init_session()
        save_path = load_path if load_path else ""

        try:
            while True:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                print("Processing...")
                assistant_response = await self.get_chat_completion(user_input)
                print(f"Assistant: {assistant_response}\n")
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            await self.close_session()
            if self.conversation_history:
                prompt = "Save conversation? (path to file or Enter to skip): "
                if save_path:
                    prompt = f"Save conversation? (path or Enter for same file [{save_path}]): "
                path = input(prompt).strip() or save_path
                if path:
                    try:
                        self.save_conversation_history(path)
                        print(f"Saved {len(self.conversation_history)} message(s) to {path}")
                    except OSError as e:
                        print(f"Could not save: {e}")
            print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(DeepSeekChatApp().main())
