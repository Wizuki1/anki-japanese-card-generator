# 🌸 AI Anki Word Generator

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Ollama|112](https://img.shields.io/badge/Ollama-Local_LLM-black.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Diffusers-yellow.svg)
![VITS2](https://img.shields.io/badge/Style--BERT--VITS2-Audio-ff69b4.svg)

An end-to-end, **100% local** AI pipeline that automatically extracts, ranks, and generates highly contextualized Anki flashcards from Japanese text. Built with offline LLMs, Stable Diffusion (SDXL), and a custom PyTorch frequency regressor.

> **Note:** This project was developed as a personal portfolio piece to demonstrate applied Machine Learning, NLP, and MLOps skills. 

---

## 🎥 Demo



https://github.com/user-attachments/assets/822116d3-11ce-499b-bf48-79498fedcc89

**Example of the card:**

<img width="1429" height="1306" alt="image" src="https://github.com/user-attachments/assets/75a59cba-4a11-4e64-9f0d-aa840d2dfa5a" />


---

## 🧠 Core Architecture & ML Pipeline

This application isn't just an API wrapper; it's a complex orchestration of several local ML models working together under a unified Flet UI.

1. **Morphological Analysis (NLP):** Parses raw Japanese text and lemmatizes words.
2. **Frequency Ranking (trained XGBoost regressor):** Instead of static dictionaries, a trained XGBoost regressor evaluates extracted words. It concatenates LaBSE embeddings with morphological features (kanji/kana ratio, length) to predict word familiarity and filters out rare/unnecessary words.
3. **Word Sense Disambiguation (Ollama + Qwen):** Resolves meaning ambiguity. Uses a 4B parameter LLM (`Qwen3`) with strict Pydantic JSON schemas to select the contextually accurate translation, generate a Japanese example sentence, and create visual tags.
4. **Image Generation (Diffusers + SDXL):** Uses `NoobAI` combined with a LoRA adapter via HuggingFace `diffusers`. Automatically renders memory-anchoring illustrations based on the LLM's visual tags.
5. **Speech Synthesis (Style-BERT-VITS2):** Generates Japanese audio for both the target word and the example sentence. It leverages the `ku-nlp/deberta-v2` contextual language model for accurate Japanese pitch accent prediction and injects emotion embeddings (Style Vectors) for expressive playback.
6. **Anki Integration:** Sends the final compiled data (Word, Reading, Translation, Context Sentence, Generated Image, and Audio files) directly to the local Anki database via `AnkiConnect`.

---

## 🚀 Key Engineering Highlights

* **Zero Cloud Dependency:** Runs entirely on local hardware. Overcame VRAM limitations by utilizing GGUF quantization for the LLM and specific GPU offloading strategies for the SDXL pipeline.
* **Hybrid Fallback System:** Implemented a robust fallback mechanism for Out-of-Vocabulary (OOV) words. If the offline dictionary fails, the LLM dynamically shifts from a formatting role to a zero-shot translation role.
* **Prompt Chaining & Constraint Engineering:** Solved "Small LLM Drifting" and "Attention Hijacking" by architecting a strict Task-Data-Rules prompt structure, guaranteeing 100% valid JSON outputs from a 4B model at `0.1` temperature.
* **Native Deployment MLOps:** Built an interactive CLI setup script (`first_launch.py`) that dynamically fetches multi-gigabyte weights via `huggingface_hub`, handling path states and creating isolated virtual environments automatically.
 * **Sequential VRAM Orchestration:** Running an LLM (Qwen), a Diffusion model (SDXL), and a TTS architecture (VITS2) simultaneously exceeds standard consumer GPU limits (8GB VRAM). The pipeline implements strict lifecycle management: dynamically loading models, executing mixed-precision inference (`torch.amp.autocast`), and explicitly flushing memory via Python's `gc.collect()` and `torch.cuda.empty_cache()` to prevent Out-of-Memory exceptions.

---

## 🛠 Installation & Usage

**Prerequisites:**
- Windows OS with an NVIDIA GPU (Minimum 8GB VRAM recommended for fast image generation).
- [Ollama](https://ollama.com/) installed and running.
- Anki desktop app with the [AnkiConnect](https://ankiweb.net/shared/info/2055492159) add-on installed.

**Setup:**
Instead of a complex Docker setup that complicates GPU passthrough on Windows, this project uses native batch scripts for an automated, one-click installation.

1. Clone the repository:
   ```bash
   git clone https://github.com/Wizuki1/anki-japanese-card-generator.git
   cd anki-japanese-card-generator
   ```
2. Run the installer:
   Double-click `install.bat`. This will automatically create a `venv`, install CUDA-enabled PyTorch, and download all required pip dependencies.
3. Download [LoRA](https://civitai.com/models/1234435/noobai-touhou-memories-of-phantasm-style) and place it in the `model` folder.
4. Launch the app:
   Double-click `start.bat`. On the first launch, the setup script will ask which models you want to use and automatically download the required `.safetensors` and `.gguf` files.

---

## 📈 Known Limitations & Future Improvements

As a solo developer building an MVP, I had to make several architectural trade-offs:
* **UI/Business Logic Coupling:** The ML pipeline orchestration are currently somewhat tightly coupled. A refactoring towards Clean Architecture (MVC/MVVM) would improve testability.
* **Audio Generation Tuning:** While sentence-level TTS synthesis is natural thanks to BERT's contextual understanding, isolated word pronunciation occasionally suffers from pitch accent ambiguity. Future updates will involve hyperparameter tuning (noise_scale, length) or migrating to a fine-tuned dictionary-based vocoder for single-word edge cases.
* There are also some minor issues with recognising words that are already in the user’s flashcards. But I’ve already got an idea on how to make word recognition more accurate, as well as more user-friendly

---

## ⚖️ Disclaimer & Licensing

This application is an open-source educational tool, created strictly for non-commercial, personal use (Fair Use). 
* The AI model weights, LoRA adapters, and embeddings dynamically downloaded by this software remain the property of their respective creators and are subject to their original licenses. 
* This repository acts only as an orchestrator and **does not distribute proprietary model weights**.

*Note: This is an experimental MVP. If you encounter bugs during installation or generation, please open an Issue!*

## The project is licensed under the MIT licence, with the exception of the components listed below

https://huggingface.co/Laxhar/noobai-XL-1.1/tree/main

https://civitai.com/models/1234435/noobai-touhou-memories-of-phantasm-style

https://github.com/litagin02/Style-Bert-VITS2

https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/main

https://huggingface.co/sentence-transformers/LaBSE
