# Generalidades
Division de formas de compresion:
- Quantization
- Pruning
- Distilation

## Cosas que inluyen en velocidad de modelo

- Esparsidad (Prunning) en neuronas SSI se usan tecnicas especiales (buscar info)
- Esparcidad en filtros (Prunning de filtro completo)
- cuantizacion y aproximacion de bajo rango

## Notaciones

- Red tutora $T$, red estudiante (todo: buscar un mejor termino) $S$
- Feature map de una capa $F \in R^{C \times WH}$. 
- De un canal $F^{k\cdot} \in R^{WH}$, de una posicion $F^{\cdot k} \in R^{C}$. 

# Papers Base


## KD hinton (cambiar)

- Formulacion exclusiva de softmax. Podria ser problematica en el caso de clasificaciones binarias.

- Softmax de red tutora se suaviza usando temperatura.

  **Insertar eq**



# Layer/ feature level

### Like What You Like: Knowledge Distill via Neuron Selectivity Transfer Zehao
Año: DEC 2017

Type: theoretical, aplication

Application: Object detection

kw: Knowledge distillation, Knowledge Transfer, features



Abstract: In this paper, we propose a novel knowledge transfer method by treating it as a distribution matching problem. Particularly, we match the distributions ofneuron selectivity patterns between teacher and student networks. To achieve this goal, we devise a new KT loss function by minimizing the Maximum Mean Discrepancy (MMD) metric between these distributions.

- El mapa de activacion de cada posicion resulta un sampleo de como la red neuronal interpreta la imagen de entrada y centrarse en la distribucion permite ver en que se centra la red neuronal para realizar la deteccion.
- Evita hacer un match directo de los feature maps ya que esto ignora la densidad de sampleo en el espacio todo:{que significa esto}. En vez de eso busca realizar un alineamiento de las distribuciones.  

- selectivity knowledge of neurons: Cada neurona se activa bajo un patron particular encontrado en entrada $X$ bajo un una tarea particular. 

- NST: se usan dos perdidas distintas, una para los feature maps y otra para la clasificacion. Clasificacion se pena con cross entropy y feature maps con MMD. 

  $$ \mathcal{L}_{NST}(WS) =\mathcal{H}(Y_{true},ps)+\frac{\lambda}{2} \mathcal{L}_{MDD^2}(F_T, F_S) $$

- MMD: se tomó prestado desde domain adaptation.

Dos sets de samples $S_p=\{p^i\}^N_{i=1}$ $S_q=\{q^i\}^m_{i=1}$, luego, la distancia MMD es:

$$ \mathcal{L}_{MDD^2}(S_p,S_q)= \mid \mid \frac{1}{N} \sum^N_{i=1}\phi(p^i) - \frac{1}{M} \sum^M_{j=1}\phi(q^j) \mid \mid$$

donde $\phi(.)$ es una funcion explicita de mapeo. Usando el kernel trick todo:documentar se puede reformular como

$$ \mathcal{L}_{MDD^2}(S_p,S_q)= \mid \mid \frac{1}{N^2} \sum^N_{i=1}\sum^N_{i'=1} k (p^i,p^{i'}) + \frac{1}{M^2} \sum^M_{j=1}\sum^M_{j'=1} k (q^j,q^{j'}) -+ \frac{1}{MN} \sum^M_{i=1}\sum^M_{j'=1} k (p^i,q^{j})  $$

- Se aplica usando sampleos  desde $F_T$ y $F_S$ sampleando la activacion a traves de todos los canales y normalizando.

  $$p^i=\frac{f^i_T}{\mid\mid f^i_T \mid\mid_2}$$ e identicamente para q con FT

- sobre el k, se usaron kernel lineal, polinomial de $d=2$ y $c=0$ y gaussiano con $\sigma^2$ igual al ECM entre los pares. El caso lineal tiene ciertas semejanzas con Attention mapping de *Z. Sergey and K. Nikos. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer*. Caso polinomial de orden 2 da la gram matrix todo:revisar.
- En general funciona mejor que todos los puntos con los que se compara el paper. en el caso de pascal VOC 2007 funciona mejor incluso que la base (Faster R-CNN)
- No se probo con GAN, lo mencionan como una idea interesante

### FITNETS: HINTS FOR THIN DEEP NETS
Año: MAR 2015 

Type: theoretical

Application: 

kw: Knowledge distillation, Knowledge Transfer

Abstract: While depth tends to improve network performances, it also makes gradient-based training more difficult since deeper networks tend to be more non-linear. The re- cently proposed knowledge distillation approach is aimed at obtaining small and fast-to-execute models, and it has shown that a student network could imitate the soft output of a larger teacher network or ensemble of networks. In this paper, we extend this idea to allow the training of a student that is deeper and thinner than the teacher, using not only the outputs but also the intermediate represen- tations learned by the teacher as hints to improve the training process and final performance of the student. Because the student intermediate hidden layer will generally be smaller than the teacher’s intermediate hidden layer, additional pa- rameters are introduced to map the student hidden layer to the prediction of the teacher hidden layer. This allows one to train deeper students that can generalize better or run faster, a trade-off that is controlled by the chosen student capacity. For example, on CIFAR-10, a deep student network with almost 10.4 times less parameters outperforms a larger, state-of-the-art teacher network.

- Se usan representaciones intermedias (features) como "hints" de lo que una red estudiante de menor dimensionalidad debiese aprender de una red de mayor dimensionalidad. Esto mejora la capacidad de generalizacion cr a un modelo enfocado solo en el resultado final reduciendo a su vez la carga computacional.
- Definen un hint como la salida de una capa convolucional $F_{T}$, desde la cual la capa de la red estudiante debe aprender. Se supone que este aprendizaje sirve como una especie de regularizacion, por lo que recomiendan usar representaciones de la parte media de la red.
- El entrenamiento se realiza usando una perdida en la siguiente forma (respetando la notacion de arriba). $r$ es un regresor que se usa simplemente para poder ajustar el tamaño de $F_S$ al de $F_T$, este debe usar la misma funcion de activacion de $F_{T}$. 

$$ \mathcal{L}_{HT}=\frac{1}{2}\mid \mid F_T-r(F_s) \mid \mid^2$$

- Dado que un r fully connected aumenta de manera drastica la cantidad de parametros necesarios para el caso convoucional, se usa un $r$ convolucional

- Primero preentrena la red usando solo el regresor y luego se entrena completa.
- No dan muchos detalles sobre la arquitectura usada.

### Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer

Año: Dec. 2016

Type: theoretical

Application: 

kw: Knowledge distillation, Knowledge Transfer

Abstract: Attention plays a critical role in human visual experience. Furthermore, it has recently been demonstrated that attention can also play an important role in the
Attention plays a critical role in human visual experience. Furthermore, it has recently been demonstrated that attention can also play an important role in the context of applying artificial neural networks to a variety of tasks from fields such
recently been demonstrated that attention can also play an important role in the context of applying artificial neural networks to a variety of tasks from fields such as computer vision and NLP. In this work we show that, by properly defining
context of applying artificial neural networks to a variety of tasks from fields such as computer vision and NLP. In this work we show that, by properly defining attention for convolutional neural networks, we can actually use this type of in-as computer vision and NLP. In this work we show that, by properly defining attention for convolutional neural networks, we can actually use this type of in- formation in order to significantly improve the performance of a student CNN
attention for convolutional neural networks, we can actually use this type of in- formation in order to significantly improve the performance of a student CNN network by forcing it to mimic the attention maps of a powerful teacher network.
formation in order to significantly improve the performance of a student CNN network by forcing it to mimic the attention maps of a powerful teacher network. To that end, we propose several novel methods of transferring attention, show-network by forcing it to mimic the attention maps of a powerful teacher network. To that end, we propose several novel methods of transferring attention, show- ing consistent improvement across a variety of datasets and convolutional neu-To that end, we propose several novel methods of transferring attention, show- ing consistent improvement across a variety of datasets and convolutional neu- ral network architectures. Code and models for our experiments are available at
ing consistent improvement across a variety of datasets and convolutional neu- ral network architectures. Code and models for our experiments are available at https://github.com/szagoruyko/attention-transfer.

- Se toma prestado el concepto de atencion desde la percepcion humana. Se distinguen 2 tripos de procesos de percepcion; atencionales y no atencionales. Los primeros permiten observar generalidades de una escena y recolectar informacion de alto nivel. Desde este proceso se logra navegar en ceirtos detalles de una escena.
- En el contexto de CNNs, se considera la atencion como mapas espaciales que permiten codificar donde enfocar mas el procesamiento. Estos mapas se pueden definir con respecto a varias capas de la red y segun en que se basen se dividen en 2 tipos, de activacion y de gradiente.
- Se realiza un estudio sobre como los mapas de atencion varían segun arquitecturas y como estos mapas pueden ser transferidos a redes estudiantes desde una red tutora ya entrenada.