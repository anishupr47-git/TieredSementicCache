# TieredSemanticCache ⚡

[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13+-blue.svg)](https://pypi.org/project/tiered-semantic-cache/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-orange.svg)](https://pypi.org/project/tiered-semantic-cache/)
[![Offline Vectors](https://img.shields.io/badge/AI%20API%20Key-Not%20Required-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()

> **The fastest, simplest semantic cache for AI chatbots, LLM prompts, and Python apps.**  
> Cut your OpenAI / database bills by 80%+ and serve answers in **sub-milliseconds** instead of seconds.

---

## 📖 What is TieredSemanticCache? (The 30-Second Explanation)

Every time someone asks your AI chatbot or web app a question, calling an LLM (like OpenAI, Claude, or DeepSeek) or querying a large database takes **2 to 3 seconds** and costs money.

### The Problem with Normal Caches (like basic Redis):
* User 1 asks: `"What are your bank opening hours?"` $\rightarrow$ Saved in Redis.
* User 2 asks: `"What time does the bank open?"`
* **Regular Redis fails!** It says: *"The letters don't match word-for-word! CACHE MISS!"* $\rightarrow$ You pay OpenAI again and your user waits 2 seconds.

### The Solution with `TieredSemanticCache`:
* It converts questions into mathematical **meaning vectors** (arrows in space).
* It realizes that both questions mean the exact same thing ($>70\%$ similarity).
* **Instant Cache Hit!** The saved answer is returned in **0.01 milliseconds** for **$0.00**!

---

## 🌟 Why "Two-Tiered"? (The Desk and The Filing Cabinet)

Most caches either run out of RAM memory or are too slow on disk. We solve this with two smart tiers:

1. **L1 RAM (The Clean Office Desk):**
   - Keeps your most frequently requested answers in lightning-fast computer memory.
   - Answers return in less than a microsecond!

2. **L2 Disk (The Metal Filing Cabinet):**
   - When your desk gets full, the oldest, least-used answers automatically slide into a file on your hard drive.
   - Uses zero-copy memory mapping (`mmap`), allowing you to store **millions of answers** without running out of RAM!

3. **Strict Exclusive Sizing:**
   - Every answer lives in only one place at a time. If an item on disk is used again, it gets promoted back to RAM automatically.

---

## 🚀 Quick Start in 3 Lines of Code

### 1. Installation

Works on **Python 3.9, 3.10, 3.11, 3.12, 3.13+** on Windows, macOS, and Linux:

```bash
pip install tiered-semantic-cache
```

### 2. Basic Python Example

```python
from semantic_cache import TieredSemanticCache

# Create your cache (works 100% offline out-of-the-box!)
cache = TieredSemanticCache()

# Save an answer
cache.set(
    query="What is the capital of France?",
    answer="The capital of France is Paris."
)

# Ask with different wording -> Instant Match!
result = cache.get("Tell me France's capital city")

if result:
    print(result.value)       # "The capital of France is Paris."
    print(result.similarity)  # e.g., 0.82 (High Match!)
    print(result.tier)        # "L1_SEMANTIC"
```

---

## 🤖 Real-World Example: FastAPI / Flask Chatbot

Here is how you use it in your web backend to save money and speed up user replies:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from semantic_cache import TieredSemanticCache, CacheConfig

app = FastAPI()
cache = TieredSemanticCache(config=CacheConfig(similarity_threshold=0.70))

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    # 1. Check cache first!
    hit = cache.get(req.question)
    if hit is not None:
        # ⚡ Instant response (~0ms), $0.00 spent on OpenAI!
        return {
            "answer": hit.value,
            "source": "cache",
            "similarity": hit.similarity
        }

    # 2. Cache Miss -> Call your expensive LLM or database
    llm_answer = call_openai_gpt4(req.question)

    # 3. Save to cache so future users get it instantly!
    cache.set(req.question, llm_answer)

    return {"answer": llm_answer, "source": "openai"}
```

---

## 🔒 Multi-Tenant Privacy (Ram vs Shyam)

If you are building an app with multiple users, you never want User A (Ram) to see private answers cached for User B (Shyam).

Use **`NamespacedSemanticCache`** to give every user their own isolated room:

```python
from semantic_cache import TieredSemanticCache, NamespacedSemanticCache

global_cache = TieredSemanticCache()

# Ram's private cache view
ram_cache = NamespacedSemanticCache(global_cache, namespace="user_ram")

# Shyam's private cache view
shyam_cache = NamespacedSemanticCache(global_cache, namespace="user_shyam")

# Ram stores his balance
ram_cache.set("What is my current balance?", "$5,000")

# Ram can see his balance
print(ram_cache.get("What is my current balance?").value)  # "$5,000"

# Shyam asks the same question -> SECURITY GUARD BLOCKS IT!
assert shyam_cache.get("What is my current balance?") is None
```

---

## ⏱️ Expiration (TTL) & Group Tag Invalidation

### Automatic Expiration (TTL)
Keep answers fresh by giving them an expiration countdown in seconds:

```python
# Expires in 10 minutes (600 seconds)
cache.set("Stock price for AAPL", "$180.50", ttl=600)

# Check remaining seconds
remaining = cache.ttl("Stock price for AAPL")
```

### Active Background Cleaning
Dead answers never clog your RAM or disk. A quiet background thread automatically cleans out expired answers every 30 seconds.

### Tag Invalidation
Label answers with tags so you can delete whole categories with one command:

```python
# Cache with tags
cache.set("Who won the 2026 World Cup?", "Team A", tags=["sports", "football"])
cache.set("Who won the 2026 Champions League?", "Team B", tags=["sports", "football"])

# When new tournament starts, delete all sports answers at once:
cache.invalidate_tag("sports")
```

---

## 🌐 Run as a Background Redis Server

`TieredSemanticCache` speaks standard **RESP (Redis Protocol)**! You can run it as a 24/7 background daemon and talk to it from any programming language (Python, JavaScript, Go, Rust, Java, or PHP).

### 1. Start the Server:
```bash
semantic-cache-server --port 6380 --ram-capacity 5000
```

### 2. Connect with Standard Redis CLI:
```bash
redis-cli -p 6380
```
```text
127.0.0.1:6380> SET "greeting" "Hello World"
OK
127.0.0.1:6380> GET "greeting"
"Hello World"

127.0.0.1:6380> SEMANTIC.SET "How to learn Python?" "Start with official tutorials!"
OK
127.0.0.1:6380> SEMANTIC.GET "best way to learn python"
"Start with official tutorials!"
```

### 3. Python Client SDK with Crash-Proof Fallback:
```python
from semantic_cache import SemanticCacheClient, TieredSemanticCache

# Connects to server, with automatic local fallback if the server ever goes offline!
client = SemanticCacheClient(
    host="127.0.0.1",
    port=6380,
    fallback_cache=TieredSemanticCache()
)

client.set("hello", "world")
print(client.get("hello"))  # "world"
```

---

## 📊 Speed & Complexity Guarantees

| Operation | Speed | How It Works |
|---|---|---|
| **Exact Lookups** (`get_exact`) | $\mathcal{O}(1)$ | Instant 1-step hash lookup. Skips vector math completely! |
| **Meaning Match** (`find_semantic`) | $\mathcal{O}((N+M)d)$ | Vectorized dot-product matrix multiplication using fast hardware math. |
| **Insert / Eviction** | $\mathcal{O}(1)$ | Instant card-swap replacement. Zero array resizing overhead. |
| **Tag Deletion** | $\mathcal{O}(K)$ | Deletes only the tagged items without searching through unrelated data. |

---

## 🛠️ Configuration Options

```python
from semantic_cache import CacheConfig, TieredSemanticCache

config = CacheConfig(
    ram_capacity=1000,           # Max items in fast RAM desk
    similarity_threshold=0.70,   # 70% meaning match dial (0.0 to 1.0)
    disk_path="cache.db",        # Hard drive storage file
    vector_dim=384,              # Coordinate arrow size
    default_ttl=3600,            # Default expiration (1 hour)
    enable_active_sweep=True,    # Automatic background cleaner
    sweep_interval_sec=30.0,     # Clean every 30 seconds
)

cache = TieredSemanticCache(config=config)
```

---

## 📄 License & Open Source

This project is licensed under the permissive **MIT License** — you are free to use it for personal projects, commercial products, startups, or academic research without any restrictions.

Contributions, feature requests, and bug reports are welcome on [GitHub](https://github.com/anishupr47-git/TieredSementicCache)!
