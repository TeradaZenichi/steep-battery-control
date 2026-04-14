# Eixos de busca e strings

Use as strings abaixo em Scopus, Web of Science, IEEE Xplore e Google Scholar.

## Eixo 1: HEMS com DER

Objetivo: contextualizar problema e aplicacoes.

String base:

`("home energy management" OR HEMS OR "residential energy management") AND (PV OR photovoltaic) AND (battery OR BESS) AND (EV OR "electric vehicle") AND ("dynamic tariff" OR "time-of-use" OR "real-time pricing")`

## Eixo 2: Otimizacao classica (MILP/MPC)

Objetivo: mostrar baseline e limitacoes para controle online.

String base:

`(HEMS OR "residential energy management") AND (MILP OR "mixed-integer linear programming" OR MPC OR "model predictive control") AND ("real-time" OR online)`

## Eixo 3: RL para gerenciamento energetico

Objetivo: mapear estado da arte em DRL para energia residencial/microgrid.

String base:

`("reinforcement learning" OR "deep reinforcement learning") AND ("home energy management" OR microgrid OR "demand response") AND (PV OR battery OR EV)`

## Eixo 4: Off-policy vs on-policy

Objetivo: justificar escolha de SAC e posicionar frente a PPO/A2C/TD3/DDPG.

String base:

`("Soft Actor-Critic" OR SAC OR TD3 OR DDPG OR PPO OR A2C) AND ("energy management" OR microgrid OR "demand response")`

## Eixo 5: Safe RL e constrained control

Objetivo: fundamentar safety projection/camada de acao factivel.

String base:

`("safe reinforcement learning" OR "constrained reinforcement learning" OR "action projection" OR "safety layer") AND ("power systems" OR "energy management" OR microgrid)`

## Eixo 6: Parcial observabilidade e modelos temporais

Objetivo: justificar historico e arquiteturas sequenciais (GRU/TCN/Attention).

String base:

`(POMDP OR "partial observability") AND ("reinforcement learning" OR control) AND (GRU OR LSTM OR TCN OR Transformer OR Attention)`

## Eixo 7: Imitation Learning + RL

Objetivo: sustentar pretreino/uso de IL para reduzir custo de tuning.

String base:

`("imitation learning" OR "behavior cloning") AND ("reinforcement learning") AND ("pretraining" OR "warm start" OR "hybrid learning") AND (energy OR microgrid OR HEMS)`

## Eixo 8: Metodologia de avaliacao e robustez

Objetivo: embasar avaliacao estatistica e operacao em multiplos cenarios.

String base:

`("evaluation" OR robustness OR stability) AND ("reinforcement learning") AND (bootstrap OR Wilcoxon OR "statistical significance")`

## Filtros sugeridos

- Janela temporal inicial: 2019-2026.
- Idioma: ingles.
- Tipo: journals + conferencias principais.
- Prioridade: estudos com ambiente comparavel (PV/BESS/EV/tarifas) e restricoes explicitas.
