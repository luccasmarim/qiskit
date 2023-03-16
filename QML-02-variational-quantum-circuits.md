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

Comecemos com o conceito de expressabilidade de um circuito quântico. Este indicativo, coletado pela referência [ref.I] trata-se quantos estados em nosso espaço de Hilbert (nesse caso a Bloch Sphere) podemos alcançar com nosso circuito variacional. Seguindo a referência [ref.V], temos como exemplo dois circuitos onde no primeiro aplicamos apenas Hadamard e um gate RZ no nosso qubit. Comparando com o segundo circuito, que conta com uma Hadamard, um RZ e um RY podemos perceber que nossa intuição está correta: o segundo circuito consegue alcançar mais estados e portanto possui uma gama mais vasta de opções.

![circuitA](/circuitA.png)
Fonte: Qiskit - Introduction to QML

![circuitB](/circuitB.png)
Fonte: Qiskit - Introduction to QML

Entretanto, como podemos fazer infinitas rotações apenas com RZ, nós realmente precisamos alcançar toda a Bloch Sphere? Realmente precisamos utilizar circuitos mais complicados para realizar treinamento? A resposta para esse questionamento está na referência [ref.I] ...
O importante aqui é entendermos que em questão de expressabilidade, o segundo circuito tem vantagem.

Passemos agora para outro parametro de comparação entre ansatz fornecido pela referência [ref.I]: o conceito de capacidade de emaranhamento.
Este parâmetro de comparação entre circuitos quânticos variacionais trata-se do quão emaranhado seu estado inicial estará após passar pelo circuito. Naturalmente, o emaranhamento é uma propriedade fundamental da mecânica quântica que usamos para ganhar alguma vantagem quando o assunto é velocidade de processamento. A aplicação de gates CNOT é fundamental para aumentar essa característica do circuito.
Existem métodos teóricos para calcular o quão emaranhado um sistema correspondente a um circuito é. Uma delas é a medida de Meyer-Wallach conforme comentado superficialmente pela referência [ref.V]. Conforme [ref.VI], a medida de Meyer-Wallach é calculada pela expressão:

$$
Q(\psi) = \frac{1}{n} \sum_{k=1}^{n} 2(1-Tr[\rho_k ^2 ])
$$

Onde $$\psi$$ é nosso estado, e $$Tr[\rho_k ^2 ]$$ é o traço da matriz do operador densidade ao quadrado de cada qubit individual quando decompomos do nosso estado em termos da base. Apresentamos um script em Python para o calculo desta métrica em apêndice I. 

Voltando para Quantum Machine Learning, estas duas métricas são utilizadas para verificar o quão adequado um certo ansatz é. Em referência [ref.I] podemos encontrar um estudo mais adequado sobre ambos conceitos e suas formulações matemáticas.

## Circuitos Variacionais para Machine Learning

Nesta etapa faremos um estudo baseado nas referências [ref.VIII], [...] e [...].
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

O Qiskit nos fornece, a saber dois circuitos quânticos variacionais principais para QML: ZZFeatureMap e o NLocal.

O ZZFeatureMap pode ser visualizado usando:

```
from qiskit.circuit.library import ZZFeatureMap
qc_zz = ZZFeatureMap(3, reps=1, insert_barriers=True)
qc_zz.decompose().draw()
```
Podemos notar que o circuito gerado por esse script tem uma camada inicial de gates de Hadamard e uma segunda camada contando com gates CNOT e gates de rotação simples X,Y e Z representados pelo gate P.

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

[Testar os circuitos??]


## Métodos Kernel

## Exemplos de Implementação (PennyLane)






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