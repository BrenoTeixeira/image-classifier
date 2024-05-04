[![en](https://img.shields.io/badge/lang-en-red.svg)](README.en.md)

# Como utilizar os scripts

Nesse documento, iremos entender como utilizar os scripts e passar algumas recomendações.

* [1.0 Estrutura do Diretório](#10-estrutura-do-diretório-dos-dados)
* [2.0 Treinamento](#20-treinamento)
* [3.0 Previsões](#30-previsão)


## 1.0 Estrutura do Diretório dos Dados

Para que os scripts funcionem, é fundamental que os dados estejam em uma estrutura correta.

Devemos ter uma pasta com as pastas de test, train e valid. Em cada uma dessas pastas, teremos pastas cujos nomes serão os índices correspondentes à espécie das flores armazenadas em cada pasta.

    └── 📁data
        └── 📁flowers
            └── 📁test 
                └── 📁1 (indice da espécie pink primrose)
                    └── image_06743.jpg (imagem de uma pink primrose)
                └── 📁10 (indice da espécie globe thistle)
                    └── image_07090.jpg (imagem de uma globe thistle)
                ...
            └── 📁train
                └── 📁1
                    └── image_06743.jpg
                └── 📁10
                    └── image_07090.jpg
                ...
            └── 📁valid
                └── 📁1
                    └── image_06743.jpg
                └── 📁10
                    └── image_07090.jpg
                ...

## 2.0 Treinamento

Para o script de treinamento do modelo, temos um parâmetro obrigatório e 8 opcionais. A seguir discutiremos cada um deles.

- data_dir: Esse parâmetro é obrigatório e se trata do caminho para o diretório onde estão os dados. Ex: data/flowers. O script automaticamente completa o caminho para as pastas de test, train e valid. Garanta que os nomes das pastas estejam corretos, caso contrário um erro ocorrerá.

- learning_rate: Parâmetro opcional (valor padrão 0.003).

- save_dir: Parâmetro opcional (valor padrão 'checkpoint.pth'). Caminho para salvar o checkpoint do modelo. Quando resume_training for ativado, esse caminho será utilizado para carregar o checkpoint para reconstruir o modelo já treinado.

- arch: Parâmetro opcional (valor padrão 'resnet'). Com esse parâmetro você pode escolher entre três estruturas pré-treinadas: 'resnet', 'vgg' e 'alexnet'.

- hidden_units: Parâmetro opcional (valor padrão 2048). Número de features de entrada no primeiro layer do classificador.

- epochs: Parâmetro opcional (valor padrão 5). Define o número de iterações de treinamento.

- gpu (Flag): Parâmetro opcional (valor padrão False). Esse parâmetro, quando True, permite o uso de GPU para treinar o modelo aumentando a velocidade de processamento. É necessário que a GPU esteja disponível em sua máquina, caso contrário a CPU será utilizada.

- resume_training (Flag): Parâmetro opcional (valor padrão False). Esse parâmetro deve ser alterado para True quando o usuário desejar continuar o treinamento de algum modelo salvo.

- no_test (Flag): Parâmetro opcional (valor padrão False). Permite que o usuário desative a etapa de teste em dados de teste não vistos no treinamento.

Agora vamos a um exemplo utilizando todos os parâmetros:

    python train.py ../data/flowers --learning_rate 0.001 --save_dir checkpint_example.pth --arch vgg --hidden_units 25088 --epochs 20 --gpu --resume_training --no_test

Repare que não é necessário passar o valor para os parâmetros gpu, resume_training e no_test. Ao passarmos no comando, eles automaticamente assumem o valor contrário do valor padrão pois foram definidos como flag. Além disso, não é necessário passar o nome do parâmetro obrigatório, apenas o seu valor.

O valor de hidden_units varia de acordo com a estrutura escolhida (arch). Você pode verificar os valores para cada estrutura com o comando help:

    python train.py --help

## 3.0 Previsão

No script de classificação de imagens, temos dois parâmetros obrigatório e 5 opcionais. A seguir discutiremos cada um deles.

- image_path: Parâmetro obrigatório. Trata-se do camaninho para o arquivo de imagem que você deseja classificar com o modelo.

- checkpoint: Parâmetro obrigatório. Caminho para o arquivo de checkpoint do modelo treinado que será utilizado para classificar a imagem.

- top_k: Parâmetro opcional (valor padrão 5). Número de classes com as maiores probabilidades a serem retornadas.

- category_name: Parâmetro opcional (valor padrão 'cat_to_name.json'). Caminho para o arquivo que mapeia os indices correspondentes às espécies de flores. Esse arquivo é necessário para retornar os nomes das classe e não os índices.

- gpu (Flag): Parâmetro opcional  (valor padrão False). Esse parâmetro, quando True, permite o uso de GPU para realizar previsões.

- save_path: Parâmetro opcional (valor padrão inference_example.jpg). Caminho para salvar a imagem gerada quando o paramêtro `plot` estiver ativado.

- plot (Flag): Parâmetro opcional (valor padrão False). O usuário deve passar este parâmetro caso deseja plotar a imagem da flor e um gráfico de barras com as top_k classes e as probabilidades correspondentes. 

Exemplo:

    python predict.py ../data/test/1/image.jpg checkpoint_example.pth --top_k 10 --category_name ../assets/cat_to_name.json --gpu --plot --save_path classes_imagens.jpg


<img src='../assets/inference_plot.jpg'/>