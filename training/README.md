# Treinar os obstáculos do Visão do Agro

A estrutura está pronta; **não há imagens anotadas nem um modelo treinado incluídos**.
O treinamento ocorre separado do servidor. O modelo ativo não muda automaticamente.

## 1. Classes

Mantenha esta ordem ao criar as classes na ferramenta de anotação:

| ID | Nome exato | O que marcar |
|---|---|---|
| 0 | wire | Fio/cabo aéreo visível |
| 1 | pole | Poste |
| 2 | tower | Torre de transmissão ou telecomunicação |
| 3 | building | Prédio, casa, galpão ou outra construção |
| 4 | bird | Pássaro |

Defina um critério consistente: poste é um suporte individual; torre é uma estrutura
de torre. Para fios, marque cada trecho visível segundo a mesma regra em todas as
imagens. Uma caixa grande ao redor de vários fios mistura fundo e objetos.
Não invente contornos onde o fio não pode ser distinguido na imagem.

Se quiser manter pessoas e outras classes, adicione os nomes no data.yaml **e exemplos
anotados dessas classes**. O treinamento customizado não preserva automaticamente
todas as classes do modelo genérico.

## 2. Preparar as imagens

Use fotos/frames com perspectiva, câmera e ambiente semelhantes ao uso real.
Varie local, fundo, iluminação, distância aparente e tamanho dos objetos.
Inclua imagens negativas revisadas (sem objetos das classes escolhidas).

Separe por **gravação/local**, por exemplo 70% treino, 20% validação e 10% teste.
Não coloque frames vizinhos do mesmo vídeo nos três grupos: isso mascara erros.
O teste final deve ficar reservado para avaliar o modelo depois das decisões de treino.

Anote caixas em uma ferramenta como [CVAT](https://docs.cvat.ai/docs/manual/advanced/formats/format-yolo-ultralytics/)
e exporte **Ultralytics YOLO Detection**, com imagens e labels. Copie o conteúdo
para as pastas abaixo; confira a ordem das classes no YAML exportado.
Não basta copiar fotografias: cada objeto precisa de anotação.

```text
datasets/obstaculos/
  data.yaml
  images/train/   ← imagens de treino
  images/val/     ← imagens de validação
  images/test/    ← imagens de teste final
  labels/train/  ← arquivos TXT correspondentes
  labels/val/
  labels/test/
```

Exemplo: `images/train/cena01.jpg` corresponde a `labels/train/cena01.txt`.
Cada linha do TXT descreve um objeto, com coordenadas normalizadas:

```text
0 0.50 0.40 0.80 0.02
```

Isso significa classe 0, centro X/Y e largura/altura. É apenas um exemplo de formato,
não uma anotação para copiar nas suas imagens. A ferramenta gera os números.
Para uma imagem negativa revisada, crie o TXT vazio. Nosso verificador exige isso
para distinguir negativos intencionais de imagens esquecidas sem anotação.

## 3. Conferir antes de treinar

No terminal, na raiz do projeto:

```powershell
.\treinar.bat --check
```

Confere imagens legíveis, TXT correspondente, cinco campos por objeto, IDs de classe,
coordenadas normalizadas, presença de cada classe em treino/validação, labels sem
imagem e imagens idênticas em grupos diferentes. Não detecta frames apenas parecidos
nem verifica se as caixas foram desenhadas corretamente: revise visualmente.

Com as pastas vazias, é esperado informar que o dataset ainda não está pronto.
Não baixa pesos nem inicia o YOLO. O conjunto test pode ficar vazio nesta etapa;
é obrigatório para o comando de avaliação final.

## 4. Treinar

Depois de a conferência passar:

```powershell
.\treinar.bat
```

Defaults: pesos locais `yolo11n.pt`, 100 épocas, imagem 960, batch 4, device automático,
workers 0 para Windows, seed 42 e FP32. São valores iniciais para experimentar,
não garantia de qualidade. A GPU exige PyTorch com CUDA disponível.
O programa não instala GPU/CUDA e não inicia treinamento no servidor WebRTC.

Exemplos:

```powershell
# Experimento curto para verificar o processo; não produz um modelo confiável por si só.
.\treinar.bat --epochs 3 --imgsz 640 --batch 2

# Quando existir CUDA compatível:
.\treinar.bat --device cuda:0 --epochs 100 --imgsz 960 --batch 4

# Outro dataset organizado com images/<split> e labels/<split>:
.\treinar.bat --data C:/dados/obstaculos/data.yaml --check
```

Se faltar memória, reduza batch primeiro. Reduzir imgsz pode eliminar fios/pássaros
pequenos da imagem. CPU funciona, mas pode tornar o treino muito demorado.
O helper resolve `path` em relação ao arquivo data.yaml e grava um YAML absoluto
em `runs/training/`, para não depender da pasta atual nem das configurações globais
do Ultralytics. Use o helper para este YAML; não o passe diretamente ao CLI sem
ajustar o caminho absoluto.

## 5. Avaliar o resultado

O terminal informa o caminho exato de `best.pt`. O primeiro treino normalmente usa:

```text
runs/training/obstaculos/weights/best.pt
```

Execuções seguintes ganham outro nome, sem sobrescrever treinos anteriores.
Após preencher images/test e labels/test:

```powershell
.\treinar.bat --evaluate runs/training/obstaculos/weights/best.pt
```

Compare recall (objetos encontrados), precisão (acertos entre alertas) e mAP por classe.
Inspecione especialmente fios/pássaros pequenos, objetos perdidos e falsos alertas.
Depois teste vídeos inéditos no runtime, com o tamanho de inferência que será usado.
Treinar em 960 e executar em 512 pode alterar a capacidade de reconhecer detalhes.

## 6. Usar no dashboard

Depois de avaliar, copie os pesos para `models/best.pt`. Preserve uma versão anterior
se esse arquivo já existir. No dashboard → Settings → Advanced:

```text
model_path: models/best.pt
monitored_classes: wire, pole, tower, building, bird
```

Aplique. A próxima inferência carrega o modelo; a lista de classes disponíveis aparece
em SETTINGS após o carregamento. O treinamento não altera esse ajuste por conta própria.

Fios precisam de validação específica: uma caixa diagonal/extensa não representa
distância física, e seu tamanho pode enganar o cálculo atual de proximidade.
Detectar o fio não prova que o sistema estime corretamente seu risco de colisão.

As imagens, labels e resultados ficam fora do Git. Guarde backups próprios.
Formato e argumentos: [dataset YOLO](https://docs.ultralytics.com/datasets/detect/) e
[treinamento Ultralytics](https://docs.ultralytics.com/modes/train/).
