# Escopo e objetivo da revisao

## O que o repositorio faz

O trabalho aborda controle em tempo real de uma residencia com recursos energeticos distribuidos, incluindo:

- carga residencial,
- geracao fotovoltaica com curtailment,
- bateria estacionaria (BESS),
- veiculo eletrico (EV),
- interacao com rede sob tarifas variaveis.

Ha duas abordagens de aprendizado:

- Imitation Learning (behavior cloning) com professor MILP.
- Reinforcement Learning off-policy (SAC com dois criticos, alvo target, ajuste de entropia e variavel dual).

As arquiteturas de actor comparadas incluem:

- MLP,
- GRU,
- TCN,
- Attention,
- Attention com memoria.

Tambem existe camada de seguranca (safety projection) para projetar a acao na regiao factivel segundo estados fisicos (SoC e disponibilidade do EV).

## Objetivo da revisao para a Introducao

A revisao deve sustentar quatro pontos:

1. Relevancia do problema de HEMS com PV+BESS+EV sob tarifas dinamicas.
2. Limites de abordagens classicas puras (MILP/MPC) para operacao online em tempo real.
3. Vantagens e riscos de RL off-policy com restricoes e parcial observabilidade.
4. Lacuna para comparacao sistematica de combinacao abordagem+arquitetura com avaliacao operacional e robustez.
