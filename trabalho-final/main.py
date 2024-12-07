import cv2
import numpy as np
import os
from datetime import datetime
import pyttsx3
from mtcnn import MTCNN
import time

ARQUIVO_MODELO = "frozen_inference_graph.pb"
ARQUIVO_CFG = "ssd_mobilenet_v2_coco.pbtxt"
CLASSE_PESSOA = 1
PASTA_INVASORES = "invasores"
ARQUIVO_VIDEO = 0

def inicializar_motor_voz():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    return engine

def falar_alerta(engine):
    engine.say("Alerta de invasão detectada")
    engine.runAndWait()

def criar_pasta_invasores():
    if not os.path.exists(PASTA_INVASORES):
        os.makedirs(PASTA_INVASORES)
    return PASTA_INVASORES

def salvar_foto(quadro, pasta, nome_arquivo):
    caminho = os.path.join(pasta, nome_arquivo)
    cv2.imwrite(caminho, quadro)

def salvar_log(numero_pessoas, hora_invasao):
    caminho_log = os.path.join(PASTA_INVASORES, f"log_invasao_{hora_invasao}.txt")
    with open(caminho_log, 'w') as f:
        f.write(f"Hora da invasão: {hora_invasao}\n")
        f.write(f"Número de pessoas detectadas: {numero_pessoas}\n")
        if numero_pessoas > 1:
            f.write("Alerta: Mais de uma pessoa detectada.\n")
        else:
            f.write("Uma única pessoa detectada.\n")

def carregar_modelo():
    try:
        net = cv2.dnn.readNetFromTensorflow(ARQUIVO_MODELO, ARQUIVO_CFG)
        return net
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        exit()

# Função para selecionar múltiplas regiões de interesse (ROIs)
def selecionar_rois(imagem):
    rois = []
    while True:
        img = imagem.copy()
        cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
        cv2.imshow('ROI', img)
        roi = cv2.selectROI('ROI', img, fromCenter=False, showCrosshair=True)
        if roi == (0, 0, 0, 0):
            cv2.destroyWindow('ROI')
            break
        rois.append(roi)
        cv2.destroyWindow('ROI')
        print("Pressione 'q' para parar de selecionar regiões ou qualquer outra tecla para selecionar outra região.")
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return rois

def dentro_de_alguma_roi(x1, y1, x2, y2, rois):
    # Verifica se a caixa [x1,y1,x2,y2] intercepta alguma ROI definida
    caixa = (x1, y1, x2, y2)
    for roi in rois:
        rx, ry, rlarg, ralt = roi
        if (x1 < rx + rlarg and x2 > rx) and (y1 < ry + ralt and y2 > ry):
            return True
    return False

def main():
    motor_voz = inicializar_motor_voz()
    criar_pasta_invasores()

    modelo = carregar_modelo()
    detector_rostos = MTCNN()
    captura = cv2.VideoCapture(ARQUIVO_VIDEO)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not captura.isOpened():
        print("Erro ao acessar a câmera. Verifique sua conexão.")
        return

    # Captura um frame inicial para seleção de ROIs
    ret, frame_inicial = captura.read()
    if not ret:
        print("Não foi possível capturar frame inicial para seleção de áreas de interesse.")
        return

    print("Selecione as áreas de interesse (se desejar). Ao terminar, pressione 'q'.")
    areas_interesse = selecionar_rois(frame_inicial)
    if len(areas_interesse) > 0:
        print("Áreas de interesse definidas:")
        for i, roi in enumerate(areas_interesse):
            x, y, w, h = roi
            print(f"ROI {i+1}: x={x}, y={y}, largura={w}, altura={h}")
    else:
        print("Nenhuma área de interesse selecionada. O sistema funcionará normalmente sem áreas de restrição.")

    print("Sistema de detecção de pessoas iniciado.")
    print("Pressione 'i' para iniciar vigilância, 'p' para pausar, 'e' para encerrar.")

    primeiro_alerta = False
    numero_pessoas_anterior = 0
    monitorando = False 
    alerta_ativo = False

    while True:
        ret, quadro = captura.read()
        if not ret:
            print("Falha ao capturar quadro da câmera.")
            break

        # Se estiver monitorando, realiza detecção
        if monitorando:
            blob = cv2.dnn.blobFromImage(quadro, size=(300, 300), swapRB=True, crop=False)
            modelo.setInput(blob)
            deteccoes = modelo.forward()

            numero_pessoas = 0
            rostos_capturados = []
            pessoas_na_roi = False

            for i in range(deteccoes.shape[2]):
                confianca = deteccoes[0, 0, i, 2]
                if confianca > 0.5:
                    classe_id = int(deteccoes[0, 0, i, 1])
                    if classe_id == CLASSE_PESSOA:
                        numero_pessoas += 1

                        (altura, largura) = quadro.shape[:2]
                        caixa = deteccoes[0, 0, i, 3:7] * np.array([largura, altura, largura, altura])
                        (inicioX, inicioY, fimX, fimY) = caixa.astype("int")

                        inicioX = max(0, inicioX)
                        inicioY = max(0, inicioY)
                        fimX = min(largura, fimX)
                        fimY = min(altura, fimY)

                        cv2.rectangle(quadro, (inicioX, inicioY), (fimX, fimY), (0, 255, 0), 2)
                        cv2.putText(quadro, f"Pessoa: {confianca:.2f}", (inicioX, inicioY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        pessoa = quadro[inicioY:fimY, inicioX:fimX]

                        if pessoa.shape[0] > 0 and pessoa.shape[1] > 0:
                            rostos = detector_rostos.detect_faces(pessoa)

                            for rosto in rostos:
                                x, y, larg, alt = rosto['box']
                                margem = 10
                                x = max(0, x - margem)
                                y = max(0, y - margem)
                                larg += margem * 2
                                alt += margem * 2

                                rosto_detectado = pessoa[y:y+alt, x:x+larg]
                                rostos_capturados.append(rosto_detectado)

                        # Verificar se a pessoa está dentro de alguma ROI
                        if dentro_de_alguma_roi(inicioX, inicioY, fimX, fimY, areas_interesse):
                            pessoas_na_roi = True

            if numero_pessoas != numero_pessoas_anterior:
                disparar_alarme = False
                if numero_pessoas > 0:
                    # Checaa cada pessoa novamente para garantir
                    for i in range(deteccoes.shape[2]):
                        confianca = deteccoes[0, 0, i, 2]
                        if confianca > 0.5:
                            classe_id = int(deteccoes[0, 0, i, 1])
                            if classe_id == CLASSE_PESSOA:
                                (altura, largura) = quadro.shape[:2]
                                caixa = deteccoes[0, 0, i, 3:7] * np.array([largura, altura, largura, altura])
                                (inicioX, inicioY, fimX, fimY) = caixa.astype("int")
                                inicioX = max(0, inicioX)
                                inicioY = max(0, inicioY)
                                fimX = min(largura, fimX)
                                fimY = min(altura, fimY)

                                # Se não houver ROIs definidas
                                if len(areas_interesse) == 0:
                                    disparar_alarme = True
                                else:
                                    # Verifica se a pessoa está em alguma ROI
                                    if dentro_de_alguma_roi(inicioX, inicioY, fimX, fimY, areas_interesse):
                                        disparar_alarme = True
                                        break

                # Só faz o alerta e salva logs se houver disparo dentro da ROI ou sem ROIs definidas
                if disparar_alarme:
                    time.sleep(2)
                    hora_invasao = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    salvar_foto(quadro, PASTA_INVASORES, f"frame_invasao_{hora_invasao}.jpg")
                    salvar_log(numero_pessoas, hora_invasao)
                    for idx, rosto in enumerate(rostos_capturados):
                        salvar_foto(rosto, PASTA_INVASORES, f"rosto_{hora_invasao}_{idx}.jpg")

                    if numero_pessoas > 0 and not primeiro_alerta:
                        falar_alerta(motor_voz)
                        primeiro_alerta = True

            numero_pessoas_anterior = numero_pessoas

            # Disparar alerta baseado nas ROIs
            if pessoas_na_roi and not alerta_ativo:
                # Disparar alerta
                time.sleep(2)
                hora_invasao = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                salvar_foto(quadro, PASTA_INVASORES, f"frame_invasao_ROI_{hora_invasao}.jpg")
                salvar_log(numero_pessoas, hora_invasao)
                for idx, rosto in enumerate(rostos_capturados):
                    salvar_foto(rosto, PASTA_INVASORES, f"rosto_ROI_{hora_invasao}_{idx}.jpg")

                falar_alerta(motor_voz)
                alerta_ativo = True 
            elif not pessoas_na_roi and alerta_ativo:
                # Reseta o estado de alerta quando não tem mais pessoas na ROI
                alerta_ativo = False

            cv2.putText(quadro, f"Pessoas detectadas: {numero_pessoas}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Se não está monitorando, apenas mostra a imagem sem detecção
            cv2.putText(quadro, "Monitoramento pausado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Câmera", quadro)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('i'):
            monitorando = True
            print("Vigilância iniciada.")
        elif tecla == ord('p'):
            monitorando = False
            print("Vigilância pausada.")
        elif tecla == ord('e'):
            print("Vigilância encerrada.")
            break

    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
