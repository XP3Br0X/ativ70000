# ativ70000
made for academic evaluation

"""
===============================================================================
SISTEMA LOCAL DE TRANSCRIÇÃO E TRADUÇÃO CONTEXTUAL DE ÁUDIO
===============================================================================

Este script automatiza a conversão de arquivos de áudio em texto (Speech-to-Text) 
e realiza a tradução interpretativa do contexto para a língua portuguesa.
Todo o processamento é executado 100% LOCALMENTE, garantindo privacidade total.

-------------------------------------------------------------------------------
📋 REQUISITOS E INSTALAÇÃO DO SISTEMA
-------------------------------------------------------------------------------
1. Python instalado (versão 3.9 ou superior)

2. Instalação do FFmpeg (Obrigatório para o Whisper processar áudio):
   - Linux (Ubuntu/Debian): sudo apt update && sudo apt install ffmpeg
   - macOS (Homebrew): brew install ffmpeg
   - Windows: Instale via Chocolatey (choco install ffmpeg) ou baixe o 
     executável oficial e adicione-o às Variáveis de Ambiente (PATH).

3. Instalação das dependências do Python:
   pip install openai-whisper ollama

4. Instalação do modelo de linguagem no Ollama (rode no terminal):
   ollama run llama3.2:1b

-------------------------------------------------------------------------------
⚙️ ESPECIFICAÇÕES TÉCNICAS
-------------------------------------------------------------------------------
- Whisper ('base'): 74 Milhões de parâmetros. Modelo ideal para balancear
  velocidade e precisão em transcrições.
- Llama 3.2 (1B): 1 Bilhão de parâmetros. Modelo leve e eficiente para rodar
  em computadores locais sem exigir GPUs potentes.
===============================================================================
"""

import whisper
import ollama

# 1. CONFIGURAÇÃO DE ENTRADA
# Defina aqui o nome ou o caminho do arquivo de áudio que deseja processar
audio_data = "namesake.mp3"

print("[*] Iniciando transcrição de áudio...")

# 2. PROCESSO DE TRANSCRIÇÃO (WHISPER)
# Carrega o modelo "base" do OpenAI Whisper
whisper_model = whisper.load_model("base")

# Executa a transcrição. O parâmetro 'fp16=False' previne avisos e garante
# a execução correta caso você esteja rodando o script usando apenas a CPU.
result = whisper_model.transcribe(audio_data, fp16=False)
transcribed = result['text']

print("[*] Transcrição concluída com sucesso.")
print("[*] Enviando texto para extração e tradução de contexto via LLM...")

# 3. ENGENHARIA DE PROMPT E TRADUÇÃO (OLLAMA)
# Cria a instrução para o modelo Llama focar no significado contextual do texto
prompt = f"""
Based on the transcribed text below, translate its context to portuguese:
Transcription: {transcribed}
"""

# Dispara a requisição para o Llama 3.2 (versão de 1 bilhão de parâmetros)
response = ollama.generate(model="llama3.2:1b", prompt=prompt)

# 4. EXIBIÇÃO DO RESULTADO FINAL
print("\n" + "="*30)
print("=== RESULTADO DA TRADUÇÃO CONTEXTUAL ===")
print("="*30)
print(response['response'])
