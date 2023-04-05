---
layout: post
title:  "QML 02: Aprendizado Supervisionado e Circuitos Quânticos Variacionais"
date:   2023-02-03 10:00:00
author: luccas_marim
categories: teoria computacao-quântica educativo
usemathjax: true
excerpt_separator: <!--more-->
---

Continuando nosso estudo sobre QML, agora passaremos a estudar modelos de circuitos quânticos variacionais. Nesta etapa, conforme é feito em Machine Learning, teremos que ramificar em aprendizado supervisionado, não supervisionado, semi supervisionado e aprendizado por reforço. 
O foco deste artigo será apresentar metódos e comparações de aprendizado de máquina quântica supervisionada para a era NISQ utilizando, conforme o artigo QML 01, a classificação CQ (Classical data with Quantum algorithms). Para isso, usaremos as referências que contam com comparações entre modelos de circuitos quânticos variacionais utilizando como métrica os conceitos de expressabilidade e capacidade de emaranhamento do estado gerado pelo circuito. Não obstante, iremos apresentar e detalhar scripts do PennyLane que fornecem exemplos de como aplicar esta etapa, e subsequentes, de QML. 

<!--more-->

## Introdução

Iniciaremos nosso artigo com o método mais simples possível de machine learning: aprendizado supervisionado de máquina com o método de classificação. Em quantum machine learning, este processo pode ser feito por gates em um circuito que são responsáveis por alterar características dos qubits como amplitude ou até na base destes. Iremos, primeiro, definir alguns conceitos seguindo o artigo [ref.I] para que possamos entender os parâmetros úteis na comparação entre circuitos quânticos variacionais. Há de se pontuar que como este tipo de estudo é relativamente recente, existem outras nomenclaturas para este processo na literatura como "parametrized quantum circuit", "ansatz" ou ainda "variational circuits". O importante para nós é termos em mente o que o processo faz.

![circuitA](/variational-quantum-classifier.png)
Fonte: Qiskit - https://www.youtube.com/watch?v=-sxlXNz7ZxU&list=PLOFEBzvs-VvqJwybFxkTiDzhf5E11p8BI&index=10

Comecemos com o conceito de expressabilidade de um circuito quântico. Esta métrica, coletado da referência [ref.I] trata-se de quantos estados em nosso espaço de Hilbert (podendo ser a Bloch Sphere para um qubit e.g.) podemos alcançar com nosso circuito variacional. Seguindo a referência [ref.V], temos como exemplo dois circuitos onde no primeiro aplicamos apenas Hadamard e um gate RZ no nosso qubit. Comparando com o segundo circuito, que conta com uma Hadamard, um RZ e um RY podemos perceber que nossa intuição está correta: o segundo circuito consegue alcançar mais estados e portanto possui uma gama mais vasta de opções de estado para treinamento.

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

### Modelos de Circuitos Quânticos Variacionais PennyLane

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


### Modelos de Circuitos Quânticos Variacionais Qiskit

Nesta etapa faremos um estudo baseado nas referências [ref.VIII], [...] e [...]. O Qiskit nos fornece, a saber, sete modelos de circuitos quânticos variacionais principais para QML: NLocal, TwoLocal, PauliTwoDesign, RealAmplitudes, EfficientSU2, ExcitationPreserving e QAOAAnsatz.

#### NLocal

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

#### TwoLocal

Este modelo de VQC consiste de blocos alternados de rotação e emaranhamento. Os blocos de rotação são compostos por gates de rotação de qubits iguais (no sentido do eixo) aplicados em todos os qubits. O bloco de emaranhamento usa portas de dois qubits para emaranhar os qubits de acordo com uma estratégia escolhida pelo usuário. Estes gates podem ser escolhidos pelo programador e portanto este modelo é altamente modificável.
O conjunto de argumentos deste modelo pode ser consultado em [ref.XVI]. Um exemplo de visualização deste é:

```
from qiskit.circuit.library import TwoLocal

circuit = TwoLocal(3, 'ry', 'cx', 'linear', reps=2, insert_barriers=True)
circuit.decompose().draw('mpl')
```

Que tem como output o circuito exemplo:

![TwoLocal](/twolocal.png)
Fonte: [Autor]

Este é apenas um exemplo de circuito gerado pelo TwoLocal. Existem várias combinações possíveis de gates para este modelo. Neste caso estamos repetindo o mesmo modelo 2 vezes em reps=2 e explicitamos o uso dos gates CNOT e RY.

#### PauliTwoDesign

Este modelo aparece frequentemente na literatura de QML conforme [ref.XVII]. Assim como o TwoLocal, este modelo consiste de camadas de rotações e emaranhamento. As rotações são feitas em torno de eixos aleatórios e o emaranhamento é realizado por gates CZ emparelhadas. Um exemplo de código para visualização do PauliTwoDesign é:

```
from qiskit.circuit.library import PauliTwoDesign

circuit = PauliTwoDesign(4, reps=2, seed=5, insert_barriers=True)
circuit.decompose().draw('mpl')
```

O output:
![PauliTwoDesign](/PauliTwoDesign.png)
Fonte: [Autor]

#### RealAmplitudes

Este modelo, além de utilizado como VQC, tem várias aplicações em química. O RealAmplitudes consiste também de camadas alternadas de rotações e emaranhamento. Como o TwoLocal, podemos escolher o padrão dos blocos mas temos também a opção de selecionar um conjunto predefinido. Diferente dos 3 modelos acima, este pode não ter blocos finais de rotações. Um exemplo de visualização do RealAmplitudes é:

```
from qiskit.circuit.library import RealAmplitudes

circuit = RealAmplitudes(3, entanglement='full', reps=2) 
circuit.decompose().draw('mpl')
```

Output:
![RealAmplitudes](/RealAmplitudes.png)
Fonte: [Autor]

Este modelo tem esse nome porque os estados quânticos resultados terão apenas amplitudes reais (não complexas). 

#### EfficientSU2

Este modelo tem nome alternativo Hardware Efficient SU(2) e este consiste também de camadas de rotação e emaranhamento. O modelo tem esse nome porque tem um padrão heurístico que pode ser usado para preparar funções de onda de teste para VQA ou circuitos de classificação para QML. Um script para visualização deste é:

```
from qiskit.circuit.library import EfficientSU2

circuit = EfficientSU2(3, reps=2)
circuit.decompose().draw('mpl')
```

Cujo output é:
![EfficientSU2](/EfficientSU2.png)
Fonte: [Autor]

Assim como os gates de Pauli, este modelo tem gates que pertencem ao conjunto SU(2), grupo unitário especial de grau 2, que consiste de matrizes unitárias de determinante 1.

#### ExcitationPreserving

```
from qiskit.circuit.library import ExcitationPreserving

circuit = ExcitationPreserving(3, reps=1, insert_barriers=True, entanglement='linear')
circuit.decompose().draw('mpl')
```

O output é:
![ExcitationPreserving](/ExcitationPreserving.png)
Fonte: [Autor]

#### QAOAAnsatz

```
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit.circuit.library import HGate, RXGate, YGate, RYGate, RZGate

mixer = QuantumCircuit(2)
cost_function = QuantumCircuit(2)
ckt = QAOAAnsatz(cost_operator=cost_function, reps=1)

circuit.decompose().decompose().draw('mpl')
```

Cujo output é:
![QAOAAnsatz](/QAOAAnsatz.png)
Fonte: [Autor]





De forma complementar, temos o ZZFeatureMap que é geralmente utilizado em data encoding:

```
from qiskit.circuit.library import ZZFeatureMap
qc_zz = ZZFeatureMap(3, reps=1, insert_barriers=True)
qc_zz.decompose().draw()
```

Podemos notar que o circuito gerado por esse script tem uma camada inicial de gates de Hadamard e uma segunda camada contando com gates CNOT e gates de rotação simples X,Y e Z representados pelo gate P. Esta função também aparece em scripts fornecidos pela comunidade para Encoding (ou Embedding) de dados.

## Testes de expressabilidade e capacidade de emaranhamento

### Modelos do Pennylane

#### RandomLayers

Prosseguindo nosso estudo, iremos calcular a capacidade de emaranhamento de qubits gerados por 5 circuitos aleatórios do layer do PennyLane qml.RandomLayers. Uma abordagem parecida pode ser encontrada em [ref.VI].

Importemos os módulos necessários e alguns adicionais:

```
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from random import randint, uniform
from math import pi
import qutip
```

Trabalharemos, sem perda de generalidade, com 2 qubits e 2 layers para cada circuito fornecido pelo RandomLayers. Inicialmente, geramos uma matriz com pesos (parâmetros) aleatórios. Criamos um vetor de matrizes porque o parametro de pesos desta função aceita matrizes como entrada.
Usaremos uma função chamada function para analisar algumas propriedades do nosso sistema.

```
def function():

    n_layers = 2
    n_wires = 2


    weights = []
    for i in range(n_layers):
        row = []
        for j in range(n_wires):
            column = []
            for k in range(n_wires):   #matriz quadrada
                column.append(uniform(0,2*pi))
            row.append(column)
        weights.append(row)

    for i in range(n_layers):
        print("Layer: {}".format(i))
        print("Matriz de pesos: {}".format(weights[i]))
        print("\n")

```

A seguir, escolhemos um valor aleatório entre 1 e $9999$ para criar o ansatz fornecido pelo RandomLayers. Criamos nosso circuito e duas repetições deste:

```
    seed = randint(1, 9999)


    dev = qml.device('default.qubit', wires=n_wires)

    @qml.qnode(dev)
    def circuit(weights, seed=None):
        for i in range(n_layers):
            qml.RandomLayers(weights=weights[i], wires=range(n_wires), seed=seed)
        return qml.state()
```

Printamos então nosso sistema utilizando o simulador do PennyLane

```
    print(qml.draw(circuit, expansion_strategy="device")(weights, seed))
    print("\nCircuito correspondente ao seed: {}".format(seed))

    ket = circuit(weights, seed)
    print("\nEstado resultado do circuito: ", ket)
```

Para finalizar nossa função principal, calculamos o grau de emaranhamento do estado gerado pelo circuito. Para tanto, usamos o script da referência [ref.VI]:

```
    def compute_Q_ptrace(ket, N):

        ket = qutip.Qobj(ket, dims=[[2]*(N), [1]*(N)]).unit()    #Transforma nosso KET em uma matriz
        #print('KET=  ', ket)   Podemos verificar se a conversão esta correta
        entanglement_sum = 0
        for k in range(N):
            #print('value of n', k, 'PTrace: ',ket.ptrace([k])**2 )
            rho_k_sq = ket.ptrace([k])**2
            entanglement_sum += rho_k_sq.tr()  
   
        Q = 2*(1 - (1/N)*entanglement_sum)
        return Q

    Q = compute_Q_ptrace(ket, n_wires)
    print("\nCircuito gerando estado com grau de emaranhamento: ", Q)
    print("\n\n==========================================================================\n\n")

    return Q
```

Dessa forma, temos um script que nos mostra o grau de emaranhamento de estados gerados por circuitos aleatórios do RandomLayers. Com o objetivo de comparar os resultados, podemos chamar nossa função principal dessa forma:

```
iteracoes = 5

k=[]
for i in range(iteracoes):
    print("=============")
    print("= Ansatz: {} =".format(i))
    print("=============\n")
    k.append(function())

for i in range(iteracoes):
    print("\n Grau de Emaranhamento {} do ansatz {}".format(k[i],i), end='')


print("\n\n Análise usando Pandas: \n")
dicionario = {"Grau de Emaranhamento": k}
df = pd.DataFrame(dicionario)

df.describe()
```

Note que da forma que foi criado, cada layer do RandomLayers é inserido 2 vezes com pesos aleatórios. Entretanto, nada nos impede de, ao invés de parâmetros aleatórios, colocarmos pesos definidos por alguma certa função custo posteriormente.
Executando este código para um número grande de iterações, digamos, 50 vezes e deixando o sistema com apenas 2 qubits, obtemos como resposta uma média baixa (e.g. 0.01) de grau de emaranhamento. Aumentando a quantidade de qubits, aumentamos essa média conforme esperado visto que teremos mais opções de estados emaranhados.


#### CVNeuralNetLayers

Infelizmente ainda não conseguimos criar uma rotina de testes para este layer o qual funciona com arquitetura fotônica. Tentamos utilizar dois dispositivos do PennyLane: 'default.qubit' e 'lightning.qubit' e ambos apresentaram erros de compilação com mensagem:

```
Gate Beamsplitter not supported on device default.qubit.autograd
```

Por ser baseado em fotônica, procuramos na documentação dos dispositivos do plugin do StrawBerry Fields mas, até o momento, nenhum destes contém os métodos qml.state e qml.show. Desta forma, a saber, não existe maneira de retornar um estado de nossa função principal e portanto não podemos rodar a função que calcula o grau de emaranhamento do estado gerado por este layer.

#### StronglyEntanglingLayers

Este layer, diferente do CVNeuralNetLayers e similar ao RandomLayers, aceita o uso do dispositivo 'default.qubit'. A rotina criada para testar este layer é identica a rotina do RandomLayers:

Importamos as bibliotecas:

```
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import qutip
```

Criamos a função principal que será responsável por instanciar o dispositivo e criar o circuito quântico. Nesta etapa, diferente do que fizemos em RandomLayers, podemos utilizar o método "shape" e a biblioteca numpy para criar aleatóriamente um vetor de matrizes que corresponderá aos pesos do nosso VQC. Neste trecho também imprimimos o vetor de matrizes, o circuito correspondente e o estado final do circuito e por fim calculamos o grau de emaranhamento do estado resultado pelo circuito.

```
def function():

    n_layers = 2
    n_wires=4

    dev = qml.device('default.qubit', wires=4)

    @qml.qnode(dev)
    def circuit(parameters):
        qml.StronglyEntanglingLayers(weights=parameters, wires=range(4))
        return qml.state()

    shape = qml.StronglyEntanglingLayers.shape(n_layers, n_wires)   #Formato do vetor de matrizes
    weights = np.random.random(size=shape)

    print("\nVetor de matrizes peso: \n", weights)
    print("\n")

    print(qml.draw(circuit, expansion_strategy="device")(weights))



    ket = circuit(weights)
    print("\nEstado resultado do circuito: ", ket)

    def compute_Q_ptrace(ket, N):

        ket = qutip.Qobj(ket, dims=[[2]*(N), [1]*(N)]).unit()
        entanglement_sum = 0
        for k in range(N):
            rho_k_sq = ket.ptrace([k])**2
            entanglement_sum += rho_k_sq.tr()  
   
        Q = 2*(1 - (1/N)*entanglement_sum)
        return Q

    Q = compute_Q_ptrace(ket, n_wires)
    print("\nCircuito gerando estado com grau de emaranhamento: ", Q)
    print("\n\n==========================================================================\n\n")


    return Q
```

Para executar a função principal, de forma identica ao RandomLayers, chamamos a função diversas vezes e armazenamos o grau de emaranhamento de cada modelo através do vetor k. Imprimimos este último e usamos o Pandas para retornar estatísticas úteis desde vetor k.

```
iteracoes = 50

k=[]
for i in range(iteracoes):
    print("=============")
    print("= Modelo: {} =".format(i))
    print("=============")
    k.append(function())

for i in range(iteracoes):
    print("\n Grau de Emaranhamento {} do ansatz {}".format(k[i],i), end='')


dicionario = {"Grau de Emaranhamento": k}
df = pd.DataFrame(dicionario)

df.describe()
```


#### BasicEntanglerLayers

Similar ao RandomLayers e o StronglyEntanglingLayers, o BasicEntanglerLayers também pode ser executado no dispositivo default.qubit do PennyLane. Dessa forma, podemos replicar todo nosso código alterando apenas o layer. Assim, uma rotina para calcular o grau de emaranhamento de um estado gerado por este circuito é:

```
import pennylane as qml
import pandas as pd
import qutip
from pennylane import numpy as np


def function():

    n_wires = 4
    n_layers = 2

    shape = qml.BasicEntanglerLayers.shape(n_layers, n_wires)
    weights = np.random.random(size=shape)

    print("\nVetor de matrizes de peso: ", weights)


    dev = qml.device('default.qubit', wires=n_wires)

    @qml.qnode(dev)
    def circuit(weights):
        qml.BasicEntanglerLayers(weights=weights, wires=range(n_wires))
        return qml.state()

    print("\n")
    print(qml.draw(circuit, expansion_strategy="device")(weights))

    ket = circuit(weights)
    print("\nEstado resultado do circuito: ", ket)

    def compute_Q_ptrace(ket, N):

        ket = qutip.Qobj(ket, dims=[[2]*(N), [1]*(N)]).unit()
        entanglement_sum = 0
        for k in range(N):
            rho_k_sq = ket.ptrace([k])**2
            entanglement_sum += rho_k_sq.tr()  
   
        Q = 2*(1 - (1/N)*entanglement_sum)
        return Q

    Q = compute_Q_ptrace(ket, n_wires)
    print("\nCircuito gerando estado com grau de emaranhamento: ", Q)
    print("\n\n==========================================================================\n\n")

    return Q

iteracoes = 4

k=[]
for i in range(iteracoes):
    print("=============")
    print("= Modelo: {} =".format(i))
    print("=============")
    k.append(function())

for i in range(iteracoes):
    print("\n Grau de Emaranhamento {} do ansatz {}".format(k[i],i), end='')


dicionario = {"Grau de Emaranhamento": k}
df = pd.DataFrame(dicionario)

df.describe()
```

#### SimplifiedTwoDesign

O layer SimplifiedTwoDesign também nos fornece o dispositivo 'default.qubit' podemos portanto usar os métodos qml.state() e qml.show(). Dessa forma, novamente, podemos repetir o escopo do código do RandomLayer. Aqui, entretanto, também precisamos caracterizar explicitamente nosso vetor de matrizes de pesos (parâmetros). Uma rotina que utiliza esse layer e nos retorna o grau de emaranhamento de alguns circuitos (variamos apenas as matrizes de pesos) é:

```
import pennylane as qml
import pandas as pd
import qutip
from pennylane import numpy as np
from random import uniform
from math import pi
    
def function():
  n_wires = 4
  n_layers = 2

  init_weights = [uniform(0,2*pi), uniform(0,2*pi), uniform(0,2*pi), uniform(0,2*pi)]

  weights_layer1 = [[uniform(0,2*pi), uniform(0,2*pi)],
                    [uniform(0,2*pi), uniform(0,2*pi)],
                    [uniform(0,2*pi), uniform(0,2*pi)]]
  weights_layer2 = [[uniform(0,2*pi), uniform(0,2*pi)],
                    [uniform(0,2*pi), uniform(0,2*pi)],
                    [uniform(0,2*pi), uniform(0,2*pi)]]
  weights_layer3 = [[uniform(0,2*pi), uniform(0,2*pi)],
                    [uniform(0,2*pi), uniform(0,2*pi)],
                    [uniform(0,2*pi), uniform(0,2*pi)]]   

  weights = [weights_layer1, weights_layer2, weights_layer3]

  print("\nVetor de matrizes de peso:", weights)

  dev = qml.device('default.qubit', wires=n_wires)

  @qml.qnode(dev)
  def circuit(init_weights, weights):
    qml.SimplifiedTwoDesign(initial_layer_weights=init_weights, weights=weights, wires=range(n_wires))
    return qml.state()
    

  print("\n")
  print(qml.draw(circuit, expansion_strategy="device")(init_weights, weights))

  ket = circuit(init_weights, weights)
  print("\nEstado resultado do circuito: ", ket)

  def compute_Q_ptrace(ket, N):
    ket = qutip.Qobj(ket, dims=[[2]*(N), [1]*(N)]).unit()
    entanglement_sum = 0
    for k in range(N):
      rho_k_sq = ket.ptrace([k])**2
      entanglement_sum += rho_k_sq.tr()  
   
    Q = 2*(1 - (1/N)*entanglement_sum)
    return Q

  Q = compute_Q_ptrace(ket, n_wires)
  print("\nCircuito gerando estado com grau de emaranhamento: ", Q)
  print("\n\n==========================================================================\n\n")

  return Q

iteracoes = 4

k=[]
for i in range(iteracoes):
    print("=============")
    print("= Modelo: {} =".format(i))
    print("=============\n")
    k.append(function())

for i in range(iteracoes):
    print("\n Grau de Emaranhamento {} do ansatz {}".format(k[i],i), end='')

dicionario = {"Grau de Emaranhamento": k}
df = pd.DataFrame(dicionario)


df.describe()
```

Entretanto, por algum motivo, esta rotina não compila corretamente no VSCode. Recomendamos o uso do Google Colabs para testes.


### Modelos do Qiskit

#### NLocal

Para estudar este layer do Qiskit repetiremos, estruturalmente, a mesma rotina dos layers fornecidos pelo PennyLane. Inicialmente importamos nossas bibliotecas:

```
from qiskit.circuit.library import NLocal
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

from random import uniform
from math import pi
from pennylane import numpy as np
import pandas as pd

import qutip
```

Note que desta vez precisamos importar métodos auxiliares do Qiskit e também o numpy do pennylane que será posteriormente usado apenas para transformar um tipo de dado (statevector) do qiskit para um tensor do numpy. Prosseguimos com a definição da nossa função principal:

```
def function():

    n_wires = 6
    n_layers = 2

    # rotation block:
    rot = QuantumCircuit(2)
    params = ParameterVector('r', 2)
    params = (uniform(0,2*pi), uniform(0,2*pi))
    print("\nParâmetros de rotação: {}".format(params))
    rot.ry(params[0], 0)
    rot.rz(params[1], 1)

    # entanglement block:
    ent = QuantumCircuit(4)
    params = ParameterVector('e', 3)
    params = (uniform(0,2*pi), uniform(0,2*pi), uniform(0,2*pi))
    print("Parâmetros de rotação/emaranhamento: {}".format(params))
    ent.crx(params[0], 0, 1)
    ent.crx(params[1], 1, 2)
    ent.crx(params[2], 2, 3)

    qc_nlocal = NLocal(num_qubits=6, rotation_blocks=rot,
                   entanglement_blocks=ent, reps=n_layers, entanglement='linear',
                   skip_final_rotation_layer=True, insert_barriers=True)

    print("\n")
    print("Circuito gerado: {}".format(qc_nlocal.decompose().draw()))

    backend = Aer.get_backend('statevector_simulator')
    ket = execute(qc_nlocal,backend).result().get_statevector()

    ket = np.array(ket)    #Transformamos nosso StateVector em um tensor do numpy (do pennylane)
    
    print("\n")
    print("Estado gerado pelo modelo: {}".format(ket))

    def compute_Q_ptrace(ket, N):

        ket = qutip.Qobj(ket, dims=[[2]*(N), [1]*(N)]).unit()

        entanglement_sum = 0
        for k in range(N):
            rho_k_sq = ket.ptrace([k])**2
            entanglement_sum += rho_k_sq.tr()  
   
        Q = 2*(1 - (1/N)*entanglement_sum)
        return Q

    Q = compute_Q_ptrace(ket, n_wires)
    print("\nCircuito gerando estado com grau de emaranhamento: ", Q)
    print("\n\n==========================================================================\n\n")

    return Q

```

Note que para utilizar o NLocal, mudamos a forma com que caracterizamos nosso vetor de matrizes de pesos (ou parâmetros). Desta vez criamos dois blocos: rotation block e entangling block especificando cada um destes com parâmetros diferentes. O motivo disto é meramente porque a implementação do Qiskit recomendou desta forma e os argumentos da função NLocal sugerem esta estruturação. É interessante deixar deste formato porque temos mais proximidade com os vetores paramétricos.
Após a criação do circuito, executamos ele e guardamos o estado resultado na variável ket que é transformada em um tensor do numpy/pennylane. Esta conversão é necessária para a execução desta rotina da forma com que foi feita.
Por fim, imprimimos o estado, o circuito e o grau de emaranhamento do estado gerado bem como os pesos correspondentes do circuito.
Prosseguimos com as repetição do modelo, conforme é feito exatamente nos outros layers.


```
iteracoes = 2

k=[]
for i in range(iteracoes):
    print("=============")
    print("= Modelo: {} =".format(i))
    print("=============\n")
    k.append(function())

for i in range(iteracoes):
    print("\n Grau de Emaranhamento {} do modelo {}".format(k[i],i), end='')

dicionario = {"Grau de Emaranhamento": k}
df = pd.DataFrame(dicionario)


df.describe()

```

#### TwoLocal
#### PauliTwoDesign
#### RealAmplitudes
#### EfficientSU2
#### ExcitationPreserving
#### QAOAAnsatz


## Conclusão
[escrever]
Comparar os métodos e resultados superficialmente

## Apêndice

#### Apêndice I: Métrica utilizada para o cálculo do grau de emaranhamento: Meyer-Wallach Measure em Python

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

#### Apêndice II: Métrica utilizada para o cálculo de expressabilidade em Python

### Futuro

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

XVI. https://qiskit.org/documentation/stubs/qiskit.circuit.library.TwoLocal.html#qiskit.circuit.library.TwoLocal

XVII. https://qiskit.org/documentation/stubs/qiskit.circuit.library.PauliTwoDesign.html#qiskit.circuit.library.PauliTwoDesign

XVIII. https://qiskit.org/documentation/stubs/qiskit.circuit.library.RealAmplitudes.html#qiskit.circuit.library.RealAmplitudes

XIX. https://qiskit.org/documentation/stubs/qiskit.circuit.library.EfficientSU2.html#qiskit.circuit.library.EfficientSU2

XX. https://qiskit.org/documentation/stubs/qiskit.circuit.library.ExcitationPreserving.html#qiskit.circuit.library.ExcitationPreserving

XXI. https://qiskit.org/documentation/stubs/qiskit.circuit.library.QAOAAnsatz.html#qiskit.circuit.library.QAOAAnsatz