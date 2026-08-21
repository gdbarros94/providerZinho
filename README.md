# EdgeAI Micro-Provider

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![OpenAI Spec](https://img.shields.io/badge/OpenAI_API-100%25_Compatible-412991.svg)](https://platform.openai.com/docs/api-reference)

API REST de alta performance e baixíssimo consumo de memória, 100% compatível com a especificação da OpenAI (`/v1/chat/completions`), desenvolvida para servir modelos minúsculos de IA em dispositivos Edge (como Raspberry Pi, servidores locais ou instâncias Docker).

Suporta **Needle 2** (para *Tool Calling* com gramática restrita) e modelos de texto minúsculos (ex: `SmolLM-135M`, `Qwen-1.5B`) via `llama.cpp` com suporte a **Server-Sent Events (SSE) streaming**.

---

## ✨ Principais Funcionalidades

- **100% Compatível com a OpenAI:** Funciona diretamente como drop-in replacement com a SDK oficial da OpenAI, LangChain, LlamaIndex, n8n ou 9router.
- **Tool Calling Estruturado (Needle 2):** Registra e executa chamadas de função usando a biblioteca `cactus-needle` com decodificação restrita por gramática.
- **Modelos de Texto GGUF:** Carrega modelos `.gguf` minúsculos via `llama-cpp-python` com baixíssimo uso de RAM (< 500 MB).
- **Streaming de Resposta (SSE):** Streaming de texto token-a-token em tempo real via Server-Sent Events, além de streaming simulado para chamadas de ferramentas.
- **Proteção contra Sobrecarga (Hardware-Aware Admission):** Monitora uso em tempo real de RAM e CPU via `psutil`. Se o servidor estiver sobrecarregado (ex: RAM > 85%), rejeita novas inferências com erro HTTP 503 padronizado para evitar travamentos ou OOM.
- **Detecção e Offload Automático para GPU:** Proba GPUs NVIDIA via `nvidia-smi`. Se houver VRAM livre suficiente, realiza offload automático das camadas (`n_gpu_layers`).
- **Pronto para Docker & Tailnet:** Facilmente implantável localmente ou em redes privadas via Docker Compose.

---

## 🏗️ Arquitetura

```text
edge_ai_provider/
├── main.py                 # Ponto de entrada da aplicação FastAPI
├── api/
│   ├── routes.py           # Endpoints (/v1/chat/completions, /v1/models, /v1/health)
│   └── schemas.py          # Modelos Pydantic v2 (OpenAI spec)
├── core/
│   ├── config.py           # Gerenciamento de configurações (pydantic-settings)
│   ├── hardware_monitor.py # Controle de admissão baseado em RAM/CPU/Semáforo
│   └── security.py         # Validação de API Key (Bearer token)
├── models/
│   ├── base.py             # Interface abstrata BaseModelAdapter
│   ├── needle_adapter.py   # Adaptador do Needle 2 para Tool Calling
│   ├── registry.py         # ModelRegistry (Factory Pattern)
│   └── text_adapter.py     # Adaptador de modelos de texto via llama.cpp
└── utils/
    ├── gpu_detector.py     # Detecção automática de VRAM/GPU via nvidia-smi
    └── stream_parser.py    # Utilitários para formatação de SSE deltas
```

---

## 🚀 Como Executar

### Opção 1: Via Docker Compose (Recomendado)

1. Clone o repositório e crie o arquivo `.env`:
   ```bash
   cp .env.example .env
   ```

2. Coloque seus modelos `.gguf` na pasta `./models`:
   ```bash
   cp /caminho/para/seu_modelo.gguf ./models/
   ```

3. Suba o container com o Docker Compose:
   ```bash
   docker compose up -d --build
   ```

4. Verifique os logs e a saúde da API:
   ```bash
   docker compose logs -f
   curl http://localhost:9880/v1/health
   ```

---

### Opção 2: Execução Local (Python)

1. Requisitos: Python 3.11+

2. Instale as dependências:
   ```bash
   pip install -e .
   ```

3. Defina as variáveis de ambiente ou edite o `.env`:
   ```bash
   export MODELS_DIR=~/.edge_ai_models
   export PORT=9880
   ```

4. Inicie o servidor:
   ```bash
   python -m edge_ai_provider.main
   ```

---

## ⚙️ Variáveis de Ambiente

| Variável | Padrão | Descrição |
| :--- | :--- | :--- |
| `PORT` | `9880` | Porta onde o servidor aceitará conexões |
| `HOST` | `0.0.0.0` | Endereço de bind do servidor |
| `API_KEY` | `sk-edge-dev-local` | Chave de API opcional para autenticação `Bearer` |
| `MODELS_DIR` | `/app/models` | Diretório contendo os arquivos `.gguf` |
| `MAX_RAM_PERCENT` | `85.0` | Limite de uso de RAM (%) antes de rejeitar requisições |
| `MAX_CPU_PERCENT` | `90.0` | Limite de uso de CPU (%) antes de rejeitar requisições |
| `MAX_CONCURRENT_INFERENCES` | `2` | Número máximo de inferências simultâneas |
| `GPU_MODE` | `auto` | Modo GPU: `auto` (detecta NVIDIA), `none` (CPU) ou número de layers |
| `NEEDLE_ENABLED` | `true` | Ativa/desativa o modelo de tool-calling `needle2-edge` |

---

## 💻 Exemplo de Uso (Python SDK OpenAI)

Você pode usar o SDK oficial da OpenAI normalmente apontando a `base_url`:

### 1. Geração de Texto com Streaming
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9880/v1",
    api_key="sk-edge-dev-local"
)

response = client.chat.completions.create(
    model="smollm-135m-instruct-v0.2.Q4_K_M",
    messages=[{"role": "user", "content": "Explique o que é Edge AI em poucas palavras."}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 2. Chamada de Ferramentas (Tool Calling com Needle 2)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9880/v1",
    api_key="sk-edge-dev-local"
)

response = client.chat.completions.create(
    model="needle2-edge",
    messages=[{"role": "user", "content": "Ligue as luzes da sala com brilho 80"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "set_lights",
            "description": "Altera o estado das luzes de um cômodo",
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {"type": "string"},
                    "brightness": {"type": "integer"}
                },
                "required": ["room", "brightness"]
            }
        }
    }]
)

print(response.choices[0].message.tool_calls)
```

---

## 📡 Endpoints Disponíveis

- **`GET /v1/health`**: Estado do servidor, métricas de hardware (RAM, CPU, GPU) e modelos ativos.
- **`GET /v1/models`**: Lista de modelos registrados.
- **`POST /v1/chat/completions`**: Endpoint principal compatível com OpenAI.

---

## 📄 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais detalhes.
