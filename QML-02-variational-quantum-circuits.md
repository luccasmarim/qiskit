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
Aqui estamos adicionando dois gates de rotação em torno dos eixos Y e Z para cada qubit do sistema. A cada iteração do sistema o valor de cada um desses paramêtros

Passando agora para uma abordagem mais prática, estudaremos 5 modelos de ansatz fornecidos pelo PennyLane: 

#### qml.CVNeuralNetLayers

Baseado em arquitetura do tipo rede neural, este layer contém gates que agem como interferometros, deslocadores e compressores [ref.XIV] - transformações lineares - e gates de Kerr que age de forma a aplicar uma "não-linearidade quântica".

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

O código fonte desta classe está no apêndice II.


#### qml.RandomLayers

Este layer é mais simples e age através de gates de rotação e CNOT aleatóriamente distribuidos pelo circuito. Para cada gate, existe um parametro de peso que podemos usar para. Abaixo está um exemplo:

```
@qml.qnode(dev)
def circuit(weights, seed=None):
    qml.RandomLayers(weights=weights, wires=range(4), seed=seed)
    return qml.expval(qml.PauliZ(0))

print(qml.draw(circuit, expansion_strategy="device")(weights, seed=97))
```

Este layer é extremamente útil por podermos testar combinações diferentes de layers para um mesmo problema. Entretanto, por ser aleatório, o ansatz frequentemente nos gera redundâncias como 2 CNOTs aplicados em sequência que é equivalente a um gate identidade. Abaixo está um dos circuitos gerados aleatóriamente.

![RandomLayers](/layer_rnd.png)
Fonte: [ref.X]

#### qml.StronglyEntanglingLayers

Este layer consiste, assim como qualquer um dos circuitos do RandomLayers, de gates de rotação e CNOT. A diferença é que este é fixo e inspirado em um artigo presente na referência [ref.XV]. Um exemplo de código é:

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
O template começa com um bloco de rotações RY

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

init_weights = [pi, pi, pi]
weights_layer1 = [[0., pi],
                  [0., pi]]
weights_layer2 = [[pi, 0.],
                  [pi, 0.]]
weights = [weights_layer1, weights_layer2]
```
[CÓDIGO NÃO ESTÁ IMPRIMINDO O CIRCUITO]

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


### Outros Circuitos Quânticos Variacionais (Qiskit)

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


## Testes de expressabilidade e capacidade de emaranhamento




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

#### Apêndice II: Código fonte do CVNeuralNetLayers

```
# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Contains the CVNeuralNetLayers template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access,arguments-differ
import pennylane as qml
from pennylane.operation import Operation, AnyWires


[docs]class CVNeuralNetLayers(Operation):
    r"""A sequence of layers of a continuous-variable quantum neural network,
    as specified in `Killoran et al. (2019) <https://doi.org/10.1103/PhysRevResearch.1.033063>`_.

    The layer consists
    of interferometers, displacement and squeezing gates mimicking the linear transformation of
    a neural network in the x-basis of the quantum system, and uses a Kerr gate
    to introduce a 'quantum' nonlinearity.

    The layers act on the :math:`M` modes given in ``wires``,
    and include interferometers of :math:`K=M(M-1)/2` beamsplitters. The different weight parameters
    contain the weights for each layer. The number of layers :math:`L` is therefore derived
    from the first dimension of ``weights``.

    This example shows a 4-mode CVNeuralNet layer with squeezing gates :math:`S`, displacement gates :math:`D` and
    Kerr gates :math:`K`. The two big blocks are interferometers of type
    :mod:`pennylane.Interferometer`:

    .. figure:: ../../_static/layer_cvqnn.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    .. note::
       The CV neural network architecture includes :class:`~pennylane.ops.Kerr` operations.
       Make sure to use a suitable device, such as the :code:`strawberryfields.fock`
       device of the `PennyLane-SF <https://github.com/XanaduAI/pennylane-sf>`_ plugin.

    Args:
        theta_1 (tensor_like): shape :math:`(L, K)` tensor of transmittivity angles for first interferometer
        phi_1 (tensor_like): shape :math:`(L, K)` tensor of phase angles for first interferometer
        varphi_1 (tensor_like): shape :math:`(L, M)` tensor of rotation angles to apply after first interferometer
        r (tensor_like): shape :math:`(L, M)` tensor of squeezing amounts for :class:`~pennylane.ops.Squeezing` operations
        phi_r (tensor_like): shape :math:`(L, M)` tensor of squeezing angles for :class:`~pennylane.ops.Squeezing` operations
        theta_2 (tensor_like): shape :math:`(L, K)` tensor of transmittivity angles for second interferometer
        phi_2 (tensor_like): shape :math:`(L, K)` tensor of phase angles for second interferometer
        varphi_2 (tensor_like): shape :math:`(L, M)` tensor of rotation angles to apply after second interferometer
        a (tensor_like): shape :math:`(L, M)` tensor of displacement magnitudes for :class:`~pennylane.ops.Displacement` operations
        phi_a (tensor_like): shape :math:`(L, M)` tensor of displacement angles for :class:`~pennylane.ops.Displacement` operations
        k (tensor_like): shape :math:`(L, M)` tensor of kerr parameters for :class:`~pennylane.ops.Kerr` operations
        wires (Iterable): wires that the template acts on

    .. details::
        :title: Usage Details

        **Parameter shapes**

        A list of shapes for the 11 input parameter tensors can be computed by the static method
        :meth:`~.CVNeuralNetLayers.shape` and used when creating randomly
        initialised weights:

        .. code-block:: python

            shapes = CVNeuralNetLayers.shape(n_layers=2, n_wires=2)
            weights = [np.random.random(shape) for shape in shapes]

            def circuit():
              CVNeuralNetLayers(*weights, wires=[0, 1])
              return qml.expval(qml.X(0))

    """

    num_wires = AnyWires
    grad_method = None

    def __init__(
        self,
        theta_1,
        phi_1,
        varphi_1,
        r,
        phi_r,
        theta_2,
        phi_2,
        varphi_2,
        a,
        phi_a,
        k,
        wires,
        do_queue=True,
        id=None,
    ):
        n_wires = len(wires)
        # n_if -> theta and phi shape for Interferometer
        n_if = n_wires * (n_wires - 1) // 2

        # check that first dimension is the same
        weights_list = [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]
        shapes = [qml.math.shape(w) for w in weights_list]
        first_dims = [s[0] for s in shapes]
        if len(set(first_dims)) > 1:
            raise ValueError(
                f"The first dimension of all parameters needs to be the same, got {first_dims}"
            )

        # check second dimensions
        second_dims = [s[1] for s in shapes]
        expected = [n_if] * 2 + [n_wires] * 3 + [n_if] * 2 + [n_wires] * 4
        if not all(e == d for e, d in zip(expected, second_dims)):
            raise ValueError("Got unexpected shape for one or more parameters.")

        self.n_layers = shapes[0][0]

        super().__init__(
            theta_1,
            phi_1,
            varphi_1,
            r,
            phi_r,
            theta_2,
            phi_2,
            varphi_2,
            a,
            phi_a,
            k,
            wires=wires,
            do_queue=do_queue,
            id=id,
        )

    @property
    def num_params(self):
        return 11

[docs]    @staticmethod
    def compute_decomposition(
        theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k, wires
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.CVNeuralNetLayers.decomposition`.

        Args:

            theta_1 (tensor_like): shape :math:`(L, K)` tensor of transmittivity angles for first interferometer
            phi_1 (tensor_like): shape :math:`(L, K)` tensor of phase angles for first interferometer
            varphi_1 (tensor_like): shape :math:`(L, M)` tensor of rotation angles to apply after first interferometer
            r (tensor_like): shape :math:`(L, M)` tensor of squeezing amounts for :class:`~pennylane.ops.Squeezing` operations
            phi_r (tensor_like): shape :math:`(L, M)` tensor of squeezing angles for :class:`~pennylane.ops.Squeezing` operations
            theta_2 (tensor_like): shape :math:`(L, K)` tensor of transmittivity angles for second interferometer
            phi_2 (tensor_like): shape :math:`(L, K)` tensor of phase angles for second interferometer
            varphi_2 (tensor_like): shape :math:`(L, M)` tensor of rotation angles to apply after second interferometer
            a (tensor_like): shape :math:`(L, M)` tensor of displacement magnitudes for :class:`~pennylane.ops.Displacement` operations
            phi_a (tensor_like): shape :math:`(L, M)` tensor of displacement angles for :class:`~pennylane.ops.Displacement` operations
            k (tensor_like): shape :math:`(L, M)` tensor of kerr parameters for :class:`~pennylane.ops.Kerr` operations
            wires (Any or Iterable[Any]): wires that the operator acts on

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> theta_1 = torch.tensor([[0.4]])
        >>> phi_1 = torch.tensor([[-0.3]])
        >>> varphi_1 = = torch.tensor([[1.7, 0.1]])
        >>> r = torch.tensor([[-1., -0.2]])
        >>> phi_r = torch.tensor([[0.2, -0.2]])
        >>> theta_2 = torch.tensor([[1.4]])
        >>> phi_2 = torch.tensor([[-0.4]])
        >>> varphi_2 = torch.tensor([[0.1, 0.2]])
        >>> a = torch.tensor([[0.1, 0.2]])
        >>> phi_a = torch.tensor([[-1.1, 0.2]])
        >>> k = torch.tensor([[0.1, 0.2]])
        >>> qml.CVNeuralNetLayers.compute_decomposition(theta_1, phi_1, varphi_1, r, phi_r, theta_2,
        ...                                             phi_2, varphi_2, a, phi_a, k, wires=["a", "b"])
        [Interferometer(tensor([0.4000]), tensor([-0.3000]), tensor([1.7000, 0.1000]), wires=['a', 'b']),
        Squeezing(tensor(-1.), tensor(0.2000), wires=['a']),
        Squeezing(tensor(-0.2000), tensor(-0.2000), wires=['b']),
        Interferometer(tensor([1.4000]), tensor([-0.4000]), tensor([0.1000, 0.2000]), wires=['a', 'b']),
        Displacement(tensor(0.1000), tensor(-1.1000), wires=['a']),
        Displacement(tensor(0.2000), tensor(0.2000), wires=['b']),
        Kerr(tensor(0.1000), wires=['a']),
        Kerr(tensor(0.2000), wires=['b'])]
        """
        op_list = []
        n_layers = qml.math.shape(theta_1)[0]
        for m in range(n_layers):
            op_list.append(
                qml.Interferometer(
                    theta=theta_1[m],
                    phi=phi_1[m],
                    varphi=varphi_1[m],
                    wires=wires,
                )
            )

            for i in range(len(wires)):  # pylint:disable=consider-using-enumerate
                op_list.append(qml.Squeezing(r[m, i], phi_r[m, i], wires=wires[i]))

            op_list.append(
                qml.Interferometer(
                    theta=theta_2[m],
                    phi=phi_2[m],
                    varphi=varphi_2[m],
                    wires=wires,
                )
            )

            for i in range(len(wires)):  # pylint: disable=consider-using-enumerate
                op_list.append(qml.Displacement(a[m, i], phi_a[m, i], wires=wires[i]))

            for i in range(len(wires)):  # pylint:disable=consider-using-enumerate
                op_list.append(qml.Kerr(k[m, i], wires=wires[i]))

        return op_list

[docs]    @staticmethod
    def shape(n_layers, n_wires):
        r"""Returns a list of shapes for the 11 parameter tensors.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of wires

        Returns:
            list[tuple[int]]: list of shapes
        """
        # n_if -> theta and phi shape for Interferometer
        n_if = n_wires * (n_wires - 1) // 2

        shapes = (
            [(n_layers, n_if)] * 2
            + [(n_layers, n_wires)] * 3
            + [(n_layers, n_if)] * 2
            + [(n_layers, n_wires)] * 4
        )

        return shapes
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