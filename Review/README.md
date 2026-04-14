# Review

Esta pasta centraliza todo o material da revisao bibliografica do paper.

## Estrutura

- `00_scope_e_objetivo.md`: escopo tecnico do trabalho no repositorio e foco da revisao.
- `01_eixos_de_busca.md`: eixos de busca e strings prontas para bases academicas.
- `02_criterios_inclusao_exclusao.md`: criterios para triagem dos artigos.
- `03_matriz_extracao_template.csv`: template para extracao padronizada dos estudos.
- `04_backlog_papers.md`: backlog para registrar artigos candidatos e status.
- `scopus_advanced_search.py`: coleta metadados no Scopus via API.
- `wos_advanced_search.py`: coleta metadados no Web of Science Starter via API.
- `run_review_searches.ps1`: executa Scopus + WoS e consolida os resultados.

## Uso recomendado

1. Execute as buscas com as strings de `01_eixos_de_busca.md`.
2. Registre candidatos em `04_backlog_papers.md`.
3. Aplique os criterios de `02_criterios_inclusao_exclusao.md`.
4. Para os artigos aprovados, preencha `03_matriz_extracao_template.csv`.

## Execucao automatizada

1. Defina chaves de API no ambiente:
   - `ELS_API_KEY` para Scopus.
   - `WOS_API_KEY` para Web of Science.
2. Rode:
   - `powershell -ExecutionPolicy Bypass -File Review/run_review_searches.ps1`
