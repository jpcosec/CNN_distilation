# Generalidades
Division de formas de compresion:
- Quantization
- Pruning
- Distilation

## Sobre la convergencia de las CNNs, el overfitting y por que conviene imitar el conocimiento de una red y no entrenar el modelo desde 0

todo: reexplicar

Por que entrenar intentando ajustarse al conocimiento de una red grande en vez de usar una red pequeña

En escencia una red neuronal fully connected de solo una capa debiese ser suficiente para poder aproximar cualquier funcion con un nivel de presicion arbitrario, controlado solo por la cantidad de parametros del modelo. todo:revisar

De esta manera, el uso de arquitecturas deep como las convoluciones, las recurrencias o cualquier otra lo que hacen es descubrir patrones dentro del dato de entrada, cosa que antes debía ser diseñada a mano por un experto. La red es capaz de combinar y aproximar estos, sin necesariamente dar un significado a cada uno.

Un detalle fundamental que muchas veces se pasa por alto es que al momento de analizar datos naturales como son fotos o lenguaje hablado se cuenta con un conjunto limitado de ejemplos, mientras que los ejemplos naturales son muchisimos mas.

Dado el modo de entrenamiento de la red y la carencia de significado que se da a los patrones, es un peligro siempre presente el sobre ajustarse a los datos sobre los que se entrenó, perdiendo la capacidad de ajuste al dato real. La forma en que se logra combatir este efecto actualmente es mediante la introduccion de artificios matematicos que permitan suavizar la funcion, permitiéndole predecir mejor en su vecindad; es decir, regularizando.

El entrenamiento de una CNN consiste basicamente en resolver un problema de optimizacion complejo, el cual se resuelve mediante backpropagation. Esto ultimo equivale a modificar los pesos de la red dando pequeños pasos en una direccion que permitan descender por el gradiente del error. Si bien la tecnica ha mostrado ser efectiva en muchos casos, esta no dista mucho de bajar un cerro con los ojos cerrados. 

Factores que influyen a una mejor convergencia

- Capacidad de aprendizaje de la red (numero de parámetros, estructura de la misma (modulos convolucionales)).
- Tecnicas de regularizacion (data agumentation, normalizacion por batch, aleatorizacion).
- Una cantidad masiva del orden de cientos de miles de datos.



## Cosas que inluyen en velocidad de modelo

- Esparsidad (Prunning) en neuronas SSI se usan tecnicas especiales (buscar info)
- Esparcidad en filtros (Prunning de filtro completo)
- cuantizacion y aproximacion de bajo rango

## Notaciones

- Red tutora $T$, red estudiante (todo: buscar un mejor termino) $S$
- Feature map o salida de una capa $F \in \mathcal{R}^{C \times WH}$, se considerará el valor despues de la activación (no linealidad $\sigma(.)$). 
- De un canal $F^{k\cdot} \in \mathcal{R}^{WH}$, de una posicion $F^{\cdot k} \in \mathcal{R}^{C}$. 

# Papers Base


## KD hinton (cambiar)

- Formulacion exclusiva de softmax. Podria ser problematica en el caso de clasificaciones binarias.

- Softmax de red tutora se suaviza usando temperatura.

  **Insertar eq**



# Layer/ feature level

### Like What You Like: Knowledge Distill via Neuron Selectivity Transfer Zehao
Año: DEC 2017

Applicacion: Imágenes

tipo: layer level distillation, CNN



Abstract: In this paper, we propose a novel knowledge transfer method by treating it as a distribution matching problem. Particularly, we match the distributions ofneuron selectivity patterns between teacher and student networks. To achieve this goal, we devise a new KT loss function by minimizing the Maximum Mean Discrepancy (MMD) metric between these distributions.

- El mapa de activacion de cada posicion resulta un sampleo de como la red neuronal interpreta la imagen de entrada y centrarse en la distribucion permite ver en que se centra la red neuronal para realizar la deteccion.
- Evita hacer un match directo de los feature maps ya que esto ignora la densidad de sampleo en el espacio todo:{que significa esto}. En vez de eso busca realizar un alineamiento de las distribuciones.  

- selectivity knowledge of neurons: Cada neurona se activa bajo un patron particular encontrado en entrada $X$ bajo un una tarea particular. 

- NST: se usan dos perdidas distintas, una para los feature maps y otra para la clasificacion. Clasificacion se pena con cross entropy y feature maps con MMD. 

  $$ \mathcal{L}_{NST}(WS) =\mathcal{L}_{ce}(Y_{true},ps)+\frac{\lambda}{2} \mathcal{L}_{MDD^2}(F_T, F_S) $$

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

Typo: Layer level distillation, CNN,

Aplicacion: Imágenes

Abstract: While depth tends to improve network performances, it also makes gradient-based training more difficult since deeper networks tend to be more non-linear. The re- cently proposed knowledge distillation approach is aimed at obtaining small and fast-to-execute models, and it has shown that a student network could imitate the soft output of a larger teacher network or ensemble of networks. In this paper, we extend this idea to allow the training of a student that is deeper and thinner than the teacher, using not only the outputs but also the intermediate represen- tations learned by the teacher as hints to improve the training process and final performance of the student. Because the student intermediate hidden layer will generally be smaller than the teacher’s intermediate hidden layer, additional pa- rameters are introduced to map the student hidden layer to the prediction of the teacher hidden layer. This allows one to train deeper students that can generalize better or run faster, a trade-off that is controlled by the chosen student capacity. For example, on CIFAR-10, a deep student network with almost 10.4 times less parameters outperforms a larger, state-of-the-art teacher network.

- Se usan representaciones intermedias (features) como "hints" de lo que una red estudiante de menor dimensionalidad debiese aprender de una red de mayor dimensionalidad. Esto mejora la capacidad de generalizacion cr a un modelo enfocado solo en el resultado final reduciendo a su vez la carga computacional.
- Definen un hint como la salida de una capa convolucional $F_{T}$, desde la cual la capa de la red estudiante debe aprender. Se supone que este aprendizaje sirve como una especie de regularizacion, por lo que recomiendan usar representaciones de la parte media de la red.
- El entrenamiento se realiza usando una perdida en la siguiente forma (respetando la notacion de arriba). $r$ es un regresor que se usa simplemente para poder ajustar el tamaño de $F_S$ al de $F_T$, este debe usar la misma funcion de activacion de $F_{T}$. 

$$ \mathcal{L}_{HT}=\frac{1}{2}\mid \mid F_T-r(F_s) \mid \mid^2$$

$$ \mathcal{L}_{NST}(WS) =\mathcal{L}_{ce}(Y_{true},ps)+\lambda \mathcal{L}_{HT}(F_T, F_S) $$

- Dado que un r fully connected aumenta de manera drastica la cantidad de parametros necesarios para el caso convoucional, se usa un $r$ convolucional

- Primero preentrena la red usando solo el regresor y luego se entrena completa.
- No dan muchos detalles sobre la arquitectura usada.

### Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer

Año: Dec. 2016

Tipo: Layer level distillation, CNN,

Applicacion: Imágenes

Abstract: Attention plays a critical role in human visual experience. Furthermore, it has recently been demonstrated that attention can also play an important role in the
Attention plays a critical role in human visual experience. Furthermore, it has recently been demonstrated that attention can also play an important role in the context of applying artificial neural networks to a variety of tasks from fields such
recently been demonstrated that attention can also play an important role in the context of applying artificial neural networks to a variety of tasks from fields such as computer vision and NLP. In this work we show that, by properly defining
context of applying artificial neural networks to a variety of tasks from fields such as computer vision and NLP. In this work we show that, by properly defining attention for convolutional neural networks, we can actually use this type of in-as computer vision and NLP. In this work we show that, by properly defining attention for convolutional neural networks, we can actually use this type of in- formation in order to significantly improve the performance of a student CNN
attention for convolutional neural networks, we can actually use this type of in- formation in order to significantly improve the performance of a student CNN network by forcing it to mimic the attention maps of a powerful teacher network.
formation in order to significantly improve the performance of a student CNN network by forcing it to mimic the attention maps of a powerful teacher network. To that end, we propose several novel methods of transferring attention, show-network by forcing it to mimic the attention maps of a powerful teacher network. To that end, we propose several novel methods of transferring attention, show- ing consistent improvement across a variety of datasets and convolutional neu-To that end, we propose several novel methods of transferring attention, show- ing consistent improvement across a variety of datasets and convolutional neu- ral network architectures. Code and models for our experiments are available at
ing consistent improvement across a variety of datasets and convolutional neu- ral network architectures. Code and models for our experiments are available at https://github.com/szagoruyko/attention-transfer.

- Se toma prestado el concepto de atencion desde la percepcion humana. Se distinguen 2 tripos de procesos de percepcion; atencionales y no atencionales. Los primeros permiten observar generalidades de una escena y recolectar informacion de alto nivel. Desde este proceso se logra navegar en ceirtos detalles de una escena.

- En el contexto de CNNs, se considera la atencion como mapas espaciales que permiten codificar donde enfocar mas el procesamiento. Estos mapas se pueden definir con respecto a varias capas de la red y segun en que se basen se dividen en 2 tipos, de activacion y de gradiente.

- Se realiza un estudio sobre como los mapas de atencion varían segun arquitecturas y como estos mapas pueden ser transferidos a redes estudiantes desde una red tutora ya entrenada. Se centraron en arquitecturas fully convolutional (redes donde la clasificacion o regresion final se realiza sin el uso de capas densas, si no aprovechando la reduccion dimensional que dan las convoluciones). 

  

- ** Mapa de atencion basado en activaciones**

  Considerando una capa de una red y su activación $F_{T}$ se define una funcion de mapeo de activacion $ \mathcal{F}: \mathcal{R}^{C \times H \times W} \rightarrow \mathcal{R}^{ H \times W}$. Asumiendo que para una neurona particular, el valor absoluto de su activacion puede ser tomado como una medida de la importancia que da la red a un determinado input, para obtener con respecto a una posicion $h,w$ se pueden usar alguno de los siguientes estadisticos.

  1. Suma de los absolutos entre los $C$ canales: $\mathcal{F}_{sum}(A)=\sum_{i=1}^C \mid A_i \mid$
  2. Suma de potencias: $\mathcal{F}^p_{sum}(A)=\sum_{i=1}^C \mid A_i \mid ^p$
  3. Maximo de potencias: $\mathcal{F}^p_{max}(A)=\max _{i=1,c} \mid A_i \mid ^p$

  En general se nota que las redes de mejor accuracy suelen tener atencion mas marcada, y que las capas iniciales se "fijan" en detalles como ojos o narices mientras que las mas profundas se fijan en objetos de mayor nivel como caras. 

  La perdida de entrenamiento en este caso es la siguiente, donde $\frac{Q_i^j}{\left \| Q_i^j \right \|}_2 $ es simplemente una normalizacion de las activaciones :

  $$ \mathcal{L}_{NST}(WS) =\mathcal{L}_{ce}(Y_{true},ps)+\frac{\lambda}{2} \mathcal{L}_{at}(F_T, F_S) $$

  Donde $\mathcal{L}_{at}(F_T, F_S)=  \sum_{j \in \mathcal{C}} \left \|  \frac{Q_S^j}{\left \| Q_S^j \right \|}_2 + \frac{Q_T^j}{\left \| Q_T^j \right \|}_2 \right \|_p$

- ** Mapa de atencion basado en gradiente**

Para el caso de gradiente, se asume que el gradiente de la perdida de clasificacion con respecto a una entrada permite medir la "sensibilidad" de la red ante el estimulo, para esto se define el gradiente como.

$$ J_i =\frac{\partial \mathcal{L}_{ce}(W_i,x)}{\partial x}$$

La perdida toma la siguiente forma, la cual puede ser dificil para analizarse analiticamente ya que implica realizar backpropagation dos veces pero con las tecnicas modernas de diferenciacion automatica no deberia ser problema.

$$ \mathcal{L}_{NST}(WS) =\mathcal{L}_{ce}(Y_{true},ps)+\frac{\lambda}{2} \left \| J_s - J_T \right \|_2 $$

- En general ambos metodos funcionan bien

### Paraphrasing Complex Network: Network Compression via Factor Transfer

Año: 2018

Tipo: layer level

Aplicacion: 

Abstract: Many researchers have sought ways of model compression to reduce the size of a deep neural network (DNN) with minimal performance degradation in order to use DNNs in embedded systems. Among the model compression methods, a method called knowledge transfer is to train a student network with a stronger teacher network. In this paper, we propose a novel knowledge transfer method which uses convolutional operations to paraphrase teacher’s knowledge and to translate it for the student. This is done by two convolutional modules, which are called a paraphraser and a translator. The paraphraser is trained in an unsupervised manner to extract the teacher factors which are defined as paraphrased information of the teacher network. The translator located at the student network extracts the student factors and helps to translate the teacher factors by mimicking them. We observed that our student network trained with the proposed factor transfer method outperforms the ones trained with conventional knowledge transfer methods.

- Destila a nivel de features, pero proponiendo el uso de  capas intermedias en un "autoencoder fashion" que sirva de "interprete" entre el conocimiento de la red tutora y la estudiante, de manera similar al regresor de fitsnets solo que con una mayor cantidad de abstraccion entre medio, le pone de nombre "factors".  
- 

## malitos de layer level

### Accelerating Convolutional Neural Networks with Dominant Convolutional Kernel and Knowledge Pre-regression

Año: 2016 

Tipo: Layer level, model compression

Aplicacion: imagenes

Abstract: Aiming at accelerating the test time of deep convolutional neural networks (CNNs), we propose a model compression method that contains a novel dominant kernel (DK) and a new training method called knowledge pre-regression (KP). In the combined model DK2PNet, DK is presented to significantly accomplish a low-rank decomposition of convolutional kernels, while KP is employed to transfer knowledge of intermediate hidden layers from a larger teacher network to its compressed student network on the basis of a cross entropy loss function instead of previous Euclidean distance. Compared to the latest results, the experimental results achieved on CIFAR-10, CIFAR-100, MNIST, and SVHN benchmarks show that our DK2PNet method has the best performance in the light of being close to the state of the art accuracy and requiring dramatically fewer number of model parameters.

- Inentendible, intentan definir una arquitectura convolucional que decompone la convolucion y entrenar una red estudiante sobre eso pero no se entiende la primera parte.



### FEED: FEATURE-LEVEL ENSEMBLE EFFECT FOR KNOWLEDGE DISTILLATION

Año: 2019

Tipo: Layer level distillation

Aplicacion

Abstract: This paper proposes a versatile and powerful training algorithm named Feature- level Ensemble Effect for knowledge Distillation (FEED), which is inspired by the work of factor transfer. The factor transfer is one of the knowledge transfer methods that improves the performance of a student network with a strong teacher network. It transfers the knowledge of a teacher in the feature map level using high-capacity teacher network, and our training algorithm FEED is an extension of it. FEED aims to transfer ensemble knowledge, using either multiple teacher in parallel or multiple training sequences. Adapting peer-teaching framework, we introduce a couple of training algorithms that transfer ensemble knowledge to the student at the feature map level, both of which help the student network find more generalized solutions in the parameter space. Experimental results on CIFAR-100 and ImageNet show that our method, FEED, has clear performance enhancements, without introducing any additional parameters or computations at test time.

- Centran la destilacion en tanto metodo de detener el overfitting al destilar desde un ensamble.
- Comentarios de open review. Muy parecido a la destilacion de ensamble de hinton 2019 sin buenos resultados.
- In this paper, the authors present two methods, Sequential and Parallel-FEED for learning student networks that share architectures with their teacher.
- it isn't clear to me where the novelty lies in this work. Sequential-FEED appears to be identical to BANs (https://arxiv.org/abs/1805.04770) with an additional non-linear transformation on the network outputs as in https://arxiv.org/abs/1802.04977. Parallel-FEED is just an ensemble of teachers; please correct me if I'm wrong.



## Otras aplicaciones

### KNOWLEDGE DISTILLATION FOR SMALL-FOOTPRINT HIGHWAY NETWORKS

Año: 2016

Tipo: Layer level distillation

Aplicacion: Audio

Abstract: Deep learning has significantly advanced state-of-the-art of speech recognition in the past few years. However, compared to conven- tional Gaussian mixture acoustic models, neural network models are usually much larger, and are therefore not very deployable in embed- ded devices. Previously, we investigated a compact highway deep neural network (HDNN) for acoustic modelling, which is a type of depth-gated feedforward neural network. We have shown that HDNN-based acoustic models can achieve comparable recognition accuracy with much smaller number of model parameters compared to plain deep neural network (DNN) acoustic models. In this pa- per, we push the boundary further by leveraging on the knowledge distillation technique that is also known as teacher-student training, i.e., we train the compact HDNN model with the supervision of a high accuracy cumbersome model. Furthermore, we also investigate sequence training and adaptation in the context of teacher-student training. Our experiments were performed on the AMI meeting speech recognition corpus. With this technique, we significantly im- proved the recognition accuracy of the HDNN acoustic model with less than 0.8 million parameters, and narrowed the gap between this model



- Aplican destilacion para un modelo compacto de neuronas parecido a LSTM.