# ai-cv-people-detection-and-alarm
people detection and alarm


# Projeto de Visão Computacional

Projeto criado para disciplina Fundamentos de Inteligência Artificial (FIA) - Graduação.
Alunos:
- Renato Cardozo
- Vinicius Lemos
- Victor Xavier


Este projeto foi desenvolvido utilizando visão computacional para a criação de um sistema de segurança utilizando detecção de identificação de pessoas e rostos em áreas pré determinadas. Ao detectar pessoas ou rostos na área pré determinada, emite um alerta sonoro e registra evidências e inormações da invasão.


Este projeto configura um ambiente virtual Python e instala as bibliotecas necessárias para um projeto de Visão Computacional.

## Configuração do Ambiente Virtual

### Passos para criar e ativar um ambiente virtual:

1. **Criar o ambiente virtual:**

   ```bash
   cd trabalho-final
   ```

   ```bash
   python -m venv env-visao
   ```

2. **Ativar o ambiente virtual:**

   No macOS e Linux:

   ```bash
   source ./env-visao/bin/activate
   ```

   No Windows:

   ```bash
   .\env-visao\Scripts\activate
   ```

## Instalação de Dependências

Certifique-se de que seu ambiente virtual esteja ativado. Instale as dependências listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Conteúdo do arquivo `requirements.txt`:

```text
numpy==2.0.0
opencv-python==4.10.0.84
pyttsx3
mtcnn
tensorflow
```

## Modelo Pré-Treinado

O modelo SSD MobileNet V2 usado para a detecção de pessoas pode ser baixado através do seguinte link:

- [Download frozen_inference_graph.pb](https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)

Extraia o arquivo `frozen_inference_graph.pb` do arquivo tar.gz baixado e coloque-o no diretório `trabalho-final` do projeto.


## Executando o Projeto

Para executar o rastreio e alarme de pessoas e rostos, simplesmente execute o script `main.py` com Python. Certifique-se de que todos os arquivos necessários estão na mesma pasta que o script.

```bash
python main.py
```

## Controles

### Passo 1 - Definir área de interesse

- Pressionar 'q' para sair do aplicativo.
- Para definir a área de interesse, clique e arraste o quadrado azul até onde desejar.
- Para cancelar a seleção da área de interesse pressione a tecla 'C'.
- Para confirmar a área de interesse pressione a tecla 'ESPAÇO'.
- Ao definir a área de interesse, aparece a mensagem 'Áreas de interesse definidas: + tamanho e posição da área de interesse' no terminal.


### Passo 2 - Durante a execução do projeto, você pode:

- Pressionar 'i' para iniciar a vigilância.
- Pressionar 'p' para pausar a vigilância.
- Pressionar 'e' para encerrar a vigilância.

*Obs...: Ao detectar uma pessoa ou rosto na área de interesse, vai emitir um alarme sonoro e sistema de identificação.

### Passo 3 - Funcionamento esperado:

- Sem pessoa ou rosto na imagem: 
Não aciona o alarme sonoro e sistema de identificação.

- Rosto ou pessoa na imagem, nenhuma pessoa ou rosto na área de interesse e vigilância ativa: 
Não aciona o alarme sonoro e sistema de identificação.

- Rosto ou pessoa na imagem, pessoa ou rosto na área de interesse e vigilância ativa: 
Aciona o alarme sonoro e sistema de identificação.

*...: obs: O alarme sonoro e sistema de identificação são acionado quando a borda verde de detecção entra na área de interesse.

### Passo 4 - Detecção de invasores

- Cria a pasta invasores se não existir.
- Registra frame inteiro de imagem do momento da invasão.
- Registra recorte do rosto identificado.
- Registra log com horário, data e número de pessoas invasoras.

## Desativação do Ambiente Virtual

Quando terminar de trabalhar no projeto, você pode desativar o ambiente virtual com o comando:

```bash
deactivate
```

