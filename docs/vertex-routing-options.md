# Vertex routing

How `VertexAIProvider` sends a request to Vertex AI, shown as diagrams. The prose
reference is [providers/vertexai.md](./providers/vertexai.md).

`VertexAIProvider` is one entry point that routes by publisher. It holds one set of
Google credentials, picks a route from the instance model at construction, and
sends each request over that route:

- `google/gemini*` goes to Gemini's native API.
- `anthropic/*` goes to the Anthropic Messages API on Vertex.
- Everything else, and every self-deployed endpoint, goes to Vertex's
  OpenAI-compatible Chat Completions endpoint.

Legend used in the diagrams:

- Blue: native protocol (one translation, nothing lost)
- Orange: OpenAI-compatible protocol (Google translates again on its side)
- Grey: shared between routes

---

## Request flow

```mermaid
flowchart LR
    EX["AgentExecutor('vertexai', provider)"] --> VP["VertexAIProvider(model, project, location)"]
    VP --> AUTH["Shared credentials<br/>credentials= / JSON env var / ADC"]
    VP --> R{"Route fixed at construction<br/>(routing='native' by default)"}

    R -->|"google/gemini*"| G1["GeminiProvider converters<br/>(reused, internal instance)"]
    G1 --> G2["genai.Client(vertexai=True)<br/>generateContent"]

    R -->|"anthropic/*"| C1["AnthropicProvider converters<br/>(reused, internal instance)"]
    C1 --> C2["AsyncAnthropicVertex<br/>rawPredict, SDK retries off"]

    R -->|"other publisher, endpoint_id,<br/>or routing='openai'"| O1["to_vertexai_messages()"]
    O1 --> O2["AsyncOpenAI<br/>endpoints/openapi or endpoints/{id}<br/>/chat/completions"]

    AUTH -.-> G2
    AUTH -.-> C2
    AUTH -.-> O2

    TF["provider.format_tools(tools)<br/>tool format of the active route"]
    R -.-> TF

    classDef native fill:#9ccfd8,stroke:#286983,color:#000
    classDef compat fill:#f6c177,stroke:#b5651d,color:#000
    classDef same fill:#dfdad9,stroke:#797593,color:#000
    class G1,G2,C1,C2 native
    class O1,O2 compat
    class AUTH,TF same
```

- All routes share the resolved credentials, project and location.
- Dobby's `with_retries` is the only retry layer on every route.
- `provider.name` is `"vertexai"` and `provider.model` is the id as passed, on
  every route. `provider.client` is the `AsyncOpenAI` client; the native routes
  use their own SDK clients.

---

## How the router picks a route

The route is chosen once, from the instance model, and reads the publisher only.
Nothing is validated against a model list.

```mermaid
flowchart TB
    START["VertexAIProvider(...)"] --> E{"endpoint_id set?"}
    E -->|yes| OAI["openai route<br/>(Chat Completions)"]
    E -->|no| RT{"routing='openai'?"}
    RT -->|yes| OAI
    RT -->|"no (default 'native')"| P{"publisher/name of model"}
    P -->|"google/ and name starts with 'gemini'"| GEM["gemini route"]
    P -->|"anthropic/"| ANT["anthropic route"]
    P -->|"anything else<br/>(google/gemma-*, meta/*, unprefixed ids)"| OAI

    classDef native fill:#9ccfd8,stroke:#286983,color:#000
    classDef compat fill:#f6c177,stroke:#b5651d,color:#000
    class GEM,ANT native
    class OAI compat
```

A per-call `model=` override is always sent over the instance's route:

```mermaid
flowchart LR
    CALL["chat(..., model='google/gemini-2.5-pro')"] --> ROUTE{"instance route"}
    ROUTE -->|"gemini"| STRIP["strip 'google/'<br/>send 'gemini-2.5-pro'"]
    ROUTE -->|"anthropic"| STRIP2["strip 'anthropic/' only<br/>send 'google/gemini-2.5-pro'<br/>Vertex returns its own error"]
    ROUTE -->|"openai"| VERB["forward verbatim"]

    classDef native fill:#9ccfd8,stroke:#286983,color:#000
    classDef bad fill:#eb6f92,stroke:#9b2335,color:#fff
    classDef compat fill:#f6c177,stroke:#b5651d,color:#000
    class STRIP native
    class STRIP2 bad
    class VERB compat
```

---

## History batching

`AgentExecutor` keeps each tool turn whole on every route. Events are unchanged;
only the history the executor builds for the next request changes.

```mermaid
flowchart LR
    subgraph BEFORE["Before: one turn split per call"]
        direction TB
        a1["assistant: tool_use A"] --> u1["user: result A"] --> a2["assistant: tool_use B"] --> u2["user: result B"]
        lost["reasoning + text dropped"]
    end
    subgraph AFTER["Now: one turn kept whole"]
        direction TB
        b1["assistant: every StreamEndEvent part<br/>(reasoning + text + tool_use A + tool_use B)"] --> b2["user: result A + result B<br/>(in call order)"]
    end
    BEFORE --> AFTER

    classDef bad fill:#eb6f92,stroke:#9b2335,color:#fff
    classDef native fill:#9ccfd8,stroke:#286983,color:#000
    class lost bad
    class b1,b2 native
```

Hosts that persist history should store it in the same shape: `StreamEndEvent.parts`
as one assistant message and every tool result as one user message. Old split-shape
histories are still accepted.

---

## Gemini thought signatures on Chat Completions

On the OpenAI-compatible route, Vertex returns Gemini thought signatures on
`tool_calls[].extra_content`. The provider keeps them and sends them back.

```mermaid
flowchart LR
    RESP["Vertex response<br/>tool_calls[].extra_content"] --> READ["ToolUsePart.metadata['extra_content']"]
    READ --> HIST["History"]
    HIST --> SEND["to_vertexai_messages()<br/>tool_calls[].extra_content, unchanged"]
    SEND --> OK["Next request carries the signature"]

    classDef compat fill:#f6c177,stroke:#b5651d,color:#000
    class READ,SEND,OK compat
```

Missing, empty or non-dict values are dropped rather than replayed.

---

## Gemini tool-call IDs

Gemini often returns function calls with no ID. Dobby mints one per call.

```mermaid
flowchart TB
    FC["Gemini function call"] --> HAS{"Gemini gave an id?"}
    HAS -->|yes| USE["use it as is"]
    HAS -->|no| BASE["candidate = call_{name}"]
    BASE --> TAKEN{"candidate taken?"}
    TAKEN -->|no| DONE["use candidate"]
    TAKEN -->|yes| NEXT["try call_{name}_2, _3, ..."]
    NEXT --> TAKEN

    subgraph T["An id is taken when"]
        T1["used earlier in the same response"]
        T2["used in history by a call that has a real result"]
        T3["NOT taken: a call answered only by an<br/>approval_required or cancelled placeholder"]
    end
```

- A single call to a tool keeps `call_{name}`, as before.
- A call paused for approval keeps its ID when the model regenerates it, so it
  still matches the host's `approved_tool_calls`.
- A later, separate call to the same tool gets a suffix, so it cannot reuse an
  earlier approval.
