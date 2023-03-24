---
layout: post
title:  "QML 02: Aprendizado Supervisionado e Circuitos Quânticos Variacionais"
date:   2023-02-03 10:00:00
author: luccas_marim
categories: teoria computacao-quântica educativo
usemathjax: true
excerpt_separator: <!--more-->
---

Continuando nosso estudo sobre QML, agora passaremos a estudar modelos de circuitos quânticos variacionais. Nesta etapa, conforme é feito em Machine Learning, teremos que ramificar essa etapa em aprendizado supervisionado, não supervisionado, semi supervisionado e aprendizado por reforço. 
O foco deste artigo será apresentar metódos e comparações de aprendizado de máquina quântica supervisionada para a era NISQ utilizando, conforme o artigo QML 01, a classificação CQ (Classical data with Quantum algorithms). Para isso, usaremos as referências que contam com comparações entre modelos de circuitos quânticos variacionais utilizando como métrica os conceitos de expressabilidade e capacidade de emaranhamento do estado gerado pelo circuito. Não obstante, iremos apresentar e detalhar scripts do PennyLane que fornecem exemplos de como aplicar esta etapa, e subsequentes, de QML. 
Por fim, tentaremos abordar o assunto focando em classificação e regressão (em aprendizado supervisionado) em Quantum Machine Learning.

<!--more-->

## Introdução

Iniciaremos nosso artigo com o método mais simples possível de machine learning: aprendizado de máquina com o método de classificação. Em quantum machine learning, este processo pode ser feito por gates em um circuito que são responsáveis por alterar características dos qubits como amplitude ou até na base destes. Iremos, primeiro, definir alguns conceitos seguindo o artigo [ref.I] para que possamos entender os parâmetros úteis na comparação entre circuitos quânticos variacionais. Há de se pontuar que como este tipo de estudo é relativamente recente, existem outras nomenclaturas para este processo na literatura como "parametrized quantum circuit", "ansatz" ou ainda "variational circuits". O importante para nós é termos em mente o que o processo faz.

![circuitA](/variational-quantum-classifier.png)
Fonte: Qiskit - https://www.youtube.com/watch?v=-sxlXNz7ZxU&list=PLOFEBzvs-VvqJwybFxkTiDzhf5E11p8BI&index=10

Comecemos com o conceito de expressabilidade de um circuito quântico. Este indicativo, coletado pela referência [ref.I] trata-se de quantos estados em nosso espaço de Hilbert (podendo ser a Bloch Sphere para um qubit e.g.) podemos alcançar com nosso circuito variacional. Seguindo a referência [ref.V], temos como exemplo dois circuitos onde no primeiro aplicamos apenas Hadamard e um gate RZ no nosso qubit. Comparando com o segundo circuito, que conta com uma Hadamard, um RZ e um RY podemos perceber que nossa intuição está correta: o segundo circuito consegue alcançar mais estados e portanto possui uma gama mais vasta de opções de estado para treinamento.

![circuitA](/circuitA.png)
Fonte: Qiskit - Introduction to QML

![circuitB](/circuitB.png)
Fonte: Qiskit - Introduction to QML

Entretanto, como podemos fazer infinitas rotações apenas com RZ, nós realmente precisamos alcançar toda a Bloch Sphere? Realmente precisamos utilizar circuitos mais complicados para realizar treinamento? A resposta para esse questionamento está na referência [ref.I] ...
O importante aqui é entendermos que em questão de expressabilidade, o segundo circuito tem vantagem.

Passemos agora para outro parâmetro de comparação entre ansatz fornecido pela referência [ref.I]: o conceito de capacidade de emaranhamento.
Este parâmetro de comparação entre circuitos quânticos variacionais trata-se do quão emaranhado seu estado inicial estará após passar pelo circuito. Naturalmente, o emaranhamento é uma propriedade fundamental da mecânica quântica que usamos para ganhar alguma vantagem quando o assunto é performance. A aplicação de gates CNOT é fundamental para aumentar essa característica do circuito.
Existem métodos teóricos para calcular o quão emaranhado um sistema correspondente a um circuito é. Uma delas é a medida de Meyer-Wallach conforme comentado superficialmente pela referência [ref.V]. Conforme [ref.VI], a medida de Meyer-Wallach é calculada pela expressão:

$$
Q(\psi) = \frac{1}{n} \sum_{k=1}^{n} 2(1-Tr[\rho_k ^2 ])
$$

Onde $$\psi$$ é nosso estado, e $$Tr[\rho_k ^2 ]$$ é o traço da matriz do operador densidade ao quadrado de cada qubit individual quando decompomos do nosso estado em termos da base. Apresentamos um script em Python para o cálculo desta métrica em apêndice I. 

Voltando para Quantum Machine Learning, estas duas métricas podem ser utilizadas para verificar o quão adequado um certo ansatz é. Em referência [ref.I] podemos encontrar um estudo mais adequado sobre ambos conceitos e suas formulações matemáticas.

## Circuitos Variacionais para Machine Learning

### Introdução

A forma mais primitiva de criarmos um circuito variacional é criando cada gate de rotação no circuito. Podemos fazer isso utilizando a função ParameterVector do Qiskit que cria uma lista de parâmetros (o qual usaremos para o treinamento) de tamanho fixo. Exemplo:

```
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

n=5  #Quantidade de qubits

rho_list = ParameterVector('ρ',length=n)

qc = QuantumCircuit(n)

for i in range(n):
    qc.rz(rho_list[i], i)
    qc.ry(rho_list[i], i)

qc.draw()
```
Aqui estamos adicionando dois gates de rotação em torno dos eixos Y e Z para cada qubit do sistema. A cada iteração do sistema o valor de cada um desses paramêtros pode ser alterado.

Em [ref.VIII] temos um script utilizando o PennyLane para um circuito quântico variacional dado por:

```
def H_layer(nqubits):
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)
# Nesse layer adicionamos gates de Hadamard para usar as propriedades de superposição de nosso modelo.

def RY_layer(w):
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)
# Nesse layer adicionamos gates de rotação em torno do eixo Y.

def entangling_layer(nqubits):
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])
# Nesse layer adicionamos gates CNOT. Criamos estados emaranhados

```

Passando agora para uma abordagem mais prática, estudaremos 5 modelos de ansatz fornecidos pelo PennyLane: 

#### qml.CVNeuralNetLayers

Baseado em arquitetura do tipo rede neural, este layer contém gates que agem como interferometros, deslocadores e compressores [ref.XIV] - transformações lineares - e gates de Kerr que agem de forma a aplicar uma "não-linearidade quântica".

![CVNeuralNetLayers](/layer_cvqnn.png)
Fonte: [ref.IX]

Onde $S$ representa os gates de compressão e $D$ os de deslocamento.
Uma forma de instanciar esta função, usando 4 qubits, pesos aleatórios e um dispositivo fotônico, é:

```
from pennylane.templates.layers import CVNeuralNetLayers
from pennylane import numpy as np
import pennylane as qml

dev = qml.device('lightning.qubit', wires=4)

shapes = CVNeuralNetLayers.shape(n_layers=2, n_wires=4)
weights = [np.random.random(shape) for shape in shapes]

@qml.qnode(dev)
def circuit():
  CVNeuralNetLayers(*weights, wires=[0, 1, 2, 3])
  return qml.expval(qml.X(0))

fig, ax = qml.draw_mpl(circuit)()
fig.show()
```


#### qml.RandomLayers

Este layer é mais simples e age através de gates de rotação e CNOT aleatóriamente distribuidos pelo circuito. Para cada gate, existe um parâmetro de peso que podemos usar para realizar backpropagration posteriormente. Abaixo está um exemplo de código utilizando esta função:

```
n_wires = 5
dev = qml.device('default.qubit', wires=n_wires)

weights = np.array([[0.1, -2.1, 1.3, 0, 0]])

@qml.qnode(dev)
def circuit(weights, seed=None):
    qml.RandomLayers(weights=weights, wires=range(5), seed=seed)
    return qml.expval(qml.PauliZ(0))

print(qml.draw(circuit, expansion_strategy="device")(weights, seed=97))
```

Este layer é extremamente útil porque podermos testar combinações diferentes de layers para um mesmo problema. Entretanto, por ser aleatório, o ansatz frequentemente nos gera redundâncias como 2 CNOTs aplicados em sequência que é equivalente a um gate identidade. Abaixo está um dos circuitos gerados aleatóriamente.

![RandomLayers](/layer_rnd.png)
Fonte: [ref.X]

#### qml.StronglyEntanglingLayers

Este layer consiste, assim como qualquer um dos circuitos do RandomLayers, de gates de rotação e CNOT. A diferença é que este é um layer fixo para uma certa quantidade de qubits e inspirado em um artigo presente na referência [ref.XV]. Um exemplo de código é:

```
dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def circuit(parameters):
    qml.StronglyEntanglingLayers(weights=parameters, wires=range(4))
    return qml.expval(qml.PauliZ(0))

shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
weights = np.random.random(size=shape)

print(qml.draw_mpl(circuit, expansion_strategy="device")(weights))
```
Executando este script temos como resposta:

![RandomLayers](/layer_sec.png)
Fonte: [ref.X]


#### qml.SimplifiedTwoDesign

Este modelo consiste em combinações de gates de rotação RY e RZ controlados e também é proposto em um artigo presente em [ref.XII] pela sua importância no estudo de "barren plateaus" em otimização.
O template começa com um bloco de rotações RY e layers que combinam RY com RZ controlado.

![RandomLayers](/simplified_two_design.png)
Fonte: [ref.XI]

```
import pennylane as qml
from math import pi

n_wires = 3
dev = qml.device('default.qubit', wires=n_wires)

@qml.qnode(dev)
def circuit(init_weights, weights):
    qml.SimplifiedTwoDesign(initial_layer_weights=init_weights, weights=weights, wires=range(n_wires))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

a = 0.
init_weights = [pi, pi, pi]
weights_layer1 = [[a, pi],
                  [a, pi]]
weights_layer2 = [[pi, a]
                  [pi, a]]
weights = [weights_layer1, weights_layer2]

fig, ax = qml.draw_mpl(circuit)(init_weights,weights)
fig.show()
```

#### qml.BasicEntanglerLayers

Este modelo consiste em rotações de apenas um parâmetro para cada qubit e gates CNOT. Na parte de CNOTs, é utilizado um ansatz onde todos os qubits precisam estar emaranhados. Abaixo um script exemplo deste layer com entrada o vetor $[\pi,\pi,\pi]$:

```
import pennylane as qml
from math import pi

n_wires = 3
dev = qml.device('default.qubit', wires=n_wires)

@qml.qnode(dev)
def circuit(weights):
    qml.BasicEntanglerLayers(weights=weights, wires=range(n_wires))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]


print(qml.draw_mpl(circuit, expansion_strategy="device")([[pi,pi,pi]]))
```

Abaixo o output:
![RandomLayers](/basicEntanglerLayers.png)
Fonte: [ref.XI]


### Outros Modelos de Circuitos Quânticos Variacionais (Qiskit)

Nesta etapa faremos um estudo baseado nas referências [ref.VIII], [...] e [...].

O Qiskit nos fornece, a saber, dois circuitos quânticos variacionais principais para QML: ZZFeatureMap e o NLocal.

O ZZFeatureMap pode ser visualizado usando:

```
from qiskit.circuit.library import ZZFeatureMap
qc_zz = ZZFeatureMap(3, reps=1, insert_barriers=True)
qc_zz.decompose().draw()
```

Podemos notar que o circuito gerado por esse script tem uma camada inicial de gates de Hadamard e uma segunda camada contando com gates CNOT e gates de rotação simples X,Y e Z representados pelo gate P. Esta função também aparece em scripts fornecidos pela comunidade para Encoding (ou Embedding) de dados.

Um exemplo de utilização do NLocal é:

```
from qiskit.circuit.library import NLocal
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


# rotation block:
rot = QuantumCircuit(2)
params = ParameterVector('r', 2)
rot.ry(params[0], 0)
rot.rz(params[1], 1)

# entanglement block:
ent = QuantumCircuit(4)
params = ParameterVector('e', 3)
ent.crx(params[0], 0, 1)
ent.crx(params[1], 1, 2)
ent.crx(params[2], 2, 3)

qc_nlocal = NLocal(num_qubits=6, rotation_blocks=rot,
                   entanglement_blocks=ent, entanglement='linear',
                   skip_final_rotation_layer=True, insert_barriers=True)

qc_nlocal.decompose().draw()
```
Que conta com um bloco inicial de gates de rotação simples e um segundo bloco de rotações controladas.


## Métodos Kernel

## Testes de expressabilidade e capacidade de emaranhamento

Prosseguindo nosso estudo, iremos calcular a capacidade de emaranhamento de qubits gerados por 5 circuitos aleatórios do layer do PennyLane qml.RandomLayers. Uma abordagem extremamente parecida mas um pouco mais simples pode ser encontrada em [ref.VI]

Trabalharemos com 8 qubits. Inicialmente, criamos nosso circuito utilizando qml.RandomLayers e geramos uma matriz com pesos (parâmetros) aleatórios: 

```
import pennylane as qml
from pennylane import numpy as np
from random import randint, uniform
from math import pi

n_wires = 8
n_layers = 2

dev = qml.device('default.qubit', wires=n_wires)


weights = np.array([[uniform(0,2*pi), 
                     uniform(0,2*pi), 
                     uniform(0,2*pi), 
                     uniform(0,2*pi),
                     uniform(0,2*pi), 
                     uniform(0,2*pi), 
                     uniform(0,2*pi), 
                     uniform(0,2*pi),
                     ]])

print("Pesos iniciais: {}".format(weights))

@qml.qnode(dev)
def circuit(weights, seed=None):
    qml.RandomLayers(weights=weights, wires=range(n_wires), seed=seed)
    for i in range(n_layers):
        weights = np.array([[uniform(0,2*pi), 
            uniform(0,2*pi), 
            uniform(0,2*pi), 
            uniform(0,2*pi),
            uniform(0,2*pi), 
            uniform(0,2*pi), 
            uniform(0,2*pi), 
            uniform(0,2*pi),
                     ]])
        print("Pesos atualizados: {} em iteração: {}".format(weights,i+2))

        qml.RandomLayers(weights=weights, wires=range(n_wires), seed=seed)

    return qml.expval(qml.PauliZ(0))


seed = randint(1,9999)
print(qml.draw(circuit, expansion_strategy="device")(weights, seed))
print()
print("Circuito correspondente ao seed: {}".format(seed))

```

O intervalo dos valores da matriz de peso foi escolhida de forma a tentar abranger todas as possíveis rotações de $0$ até $2 \pi$. A escolha do intervalo para a seed foi aleatória: de $1$ até $9999$.
Note que da forma que foi criado, o layer é inserido 3 vezes com pesos aleatórios. Entretanto, nada nos impede de, ao invés de parâmetros aleatórios, colocarmos pesos definidos por alguma certa função custo.

Testaremos agora a entangling capability do estado gerado por este sistema.



## Conclusão



## Apêndice

#### Apêndice I: Meyer-Wallach Measure em Python

Um script para calcular o quão emaranhado é um estado, baseado na referência [ref. VI] é este abaixo. Nele, é executado dois estados: um de Bell (maximamente emaranhado em sistemas de 2 qubits) e outro estado puro (minimamente emaranhado em sistemas de 2 qubits).

```
import numpy as np
import qutip

def compute_Q_ptrace(ket, N):
    """Computes Meyer-Wallach measure using alternative interpretation, i.e. as
    an average over the entanglements of each qubit with the rest of the system
    (see https://arxiv.org/pdf/quant-ph/0305094.pdf).
   
    Args:
    =====
    ket : numpy.ndarray or list
        Vector of amplitudes in 2**N dimensions
    N : int
        Number of qubits
​
    Returns:
    ========
    Q : float
        Q value for input ket
    """
    ket = qutip.Qobj(ket, dims=[[2]*(N), [1]*(N)]).unit()
    print('KET=  ', ket)
    entanglement_sum = 0
    for k in range(N):
        print('value of n', k, 'PTrace: ',ket.ptrace([k])**2 )
        rho_k_sq = ket.ptrace([k])**2
        entanglement_sum += rho_k_sq.tr()  
   
    Q = 2*(1 - (1/N)*entanglement_sum)
    return Q

if __name__ == "__main__":

    # Test #1: bell state (Q should be 1)
    n_qubits = 2
    test_state = np.zeros(2**n_qubits)
    test_state[0] = 1
    test_state[-1] = 1
    test_state /= np.linalg.norm(test_state)
    print('test state:',test_state.shape)

    print('Test #1 (Q=1):')
    Q_value = compute_Q_ptrace(ket=test_state, N=n_qubits)
    print('Q = {}\n'.format(Q_value))

    # Test #2: product state (Q should be 0)
    n_qubits = 4
    test_state = np.zeros(2**n_qubits)
    test_state[0] = 1
    test_state[1] = 1
    test_state /= np.linalg.norm(test_state)

    print('Test #2 (Q=0):')
    Q_value = compute_Q_ptrace(ket=test_state, N=n_qubits)
    print('Q = {}\n'.format(Q_value))

```


### Referências

I. https://arxiv.org/abs/1905.10876

II. https://pennylane.ai/qml/demos/tutorial_variational_classifier.html

III. https://pennylane.ai/qml/demos/tutorial_qubit_rotation.html

IV. https://arxiv.org/abs/2012.09265

V. https://learn.qiskit.org/course/machine-learning/parameterized-quantum-circuits

VI. https://born-2learn.github.io/posts/2021/01/effect-of-entanglement-on-model-training/#:~:text=Meyer%20Wallach%20Entropy%20measure%20It%20is%20a%20single,the%20kth%20qubit%20after%20tracing%20out%20the%20rest.

VII. Vojtech Havlicek, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow and Jay M. Gambetta, Supervised learning with quantum enhanced feature spaces, Nature 567, 209-212 (2019), doi.org:10.1038/s41586-019-0980-2, arXiv:1804.11326.

VIII. https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html

IX. https://docs.pennylane.ai/en/stable/code/api/pennylane.CVNeuralNetLayers.html

X. https://docs.pennylane.ai/en/stable/code/api/pennylane.RandomLayers.html

XI. https://docs.pennylane.ai/en/stable/code/api/pennylane.StronglyEntanglingLayers.html

XII. https://docs.pennylane.ai/en/stable/code/api/pennylane.SimplifiedTwoDesign.html

XIII. https://docs.pennylane.ai/en/stable/code/api/pennylane.BasicEntanglerLayers.html

XIV. https://strawberryfields.ai/photonics/demos/run_gate_visualization.html

XV. https://arxiv.org/abs/1804.00633