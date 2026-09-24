# Testes unitários do AgroSafe Vision

Foram adicionados **32 testes**, priorizando cálculo de risco, proximidade,
histórico de objetos, alertas, configurações, telemetria e missões.
A suíte completa passou com **68 testes**, incluindo o teste de integração WebRTC
que já existia. Esse total é uma contagem de testes, não uma medida de cobertura
de linhas ou de precisão do modelo de IA.

## Como executar

Na raiz do projeto, com as dependências instaladas no ambiente Python:

```powershell
# Suíte completa, incluindo a integração WebRTC local
.\.venv\Scripts\python.exe -m unittest discover -v

# Apenas os 32 testes novos, sem conexão WebRTC
.\.venv\Scripts\python.exe -m unittest tests.test_safety_unit tests.test_processing tests.test_app_unit -v

# Um teste específico
.\.venv\Scripts\python.exe -m unittest tests.test_app_unit.AlertTests.test_alert_cooldown_suppresses_duplicates_but_not_escalation -v
```

O resultado esperado termina em `OK`. `FAIL` indica uma expectativa não atendida;
`ERROR` indica que houve uma exceção durante o teste ou sua preparação.

## Como os testes são isolados

Usamos `unittest`, da biblioteca padrão, e `unittest.mock` para substituir as
dependências externas. Os testes de processamento recebem imagens sintéticas e
detecções predefinidas; as regras reais de proximidade e risco continuam sendo
executadas. Nenhum peso é baixado e não é necessário ter câmera ou GPU.

[tests/support.py](../tests/support.py) importa o módulo de banco com uma URL
SQLite em memória e bloqueia a leitura do `.env` durante essa importação. Nos
testes da aplicação, `SessionLocal` é substituído por uma sessão simulada: os
objetos enviados para persistência são inspecionados sem executar SQL. O
`lifespan` do FastAPI não é iniciado. Configurações, estado e mocks são
restaurados ao terminar cada teste, inclusive em caso de falha.

Os testes dos handlers chamam as funções Python diretamente, sem servidor HTTP.
Eles verificam regras e respostas das funções, mas não validam o transporte HTTP
nem a persistência real no PostgreSQL. O teste WebRTC existente é de integração
e usa uma conexão local com frames sintéticos.

## Os cinco testes mais importantes

### 1. Classificação crítica nos limites de proximidade, corredor e aproximação

**Teste:** `RiskBoundaryTests.test_critical_requires_proximity_corridor_and_approach`
em [test_safety_unit.py](../tests/test_safety_unit.py).

**Por que importa:** uma alteração nos comparadores pode antecipar ou atrasar a
classificação de um obstáculo como crítico.

**Como funciona:** chama `CollisionRiskEngine.candidate()` com proximidade 78,
aproximação de 8 pontos por segundo e interseção de 35% com o corredor. Espera
`CRITICAL`. Depois reduz separadamente cada valor para 77,9, 7,9 e 34,9%,
esperando `HIGH` em cada cenário.

Esses valores mantêm o perigo combinado abaixo de 102, o limiar da outra regra
que também pode produzir `CRITICAL`. Assim, o teste verifica especificamente a
condição que exige os três critérios simultâneos. A regra alternativa tem seu
próprio teste: `test_combined_danger_can_be_critical_without_approach`.

### 2. Recarga do modelo sem reaproveitar o histórico anterior

**Teste:** `ProcessingTests.test_model_reload_discards_confirmation_and_proximity_history`
em [test_processing.py](../tests/test_processing.py).

**Por que importa:** depois de recarregar o detector, o mesmo ID pode representar
outro objeto. Reutilizar seu histórico poderia gerar aproximação ou risco falsos.

**Como funciona:** um detector simulado retorna o objeto 7 em dois frames,
confirmando risco `CRITICAL`. O teste altera a geração do detector para simular
uma recarga e retorna uma caixa maior com o mesmo ID.

No primeiro frame após a recarga, espera `SAFE` e aproximação zero: a confirmação
e a comparação temporal devem começar novamente. No segundo frame, espera
`CRITICAL`, comprovando que a confirmação continua funcionando após a limpeza.

### 3. Intervalo entre alertas sem bloquear aumento de gravidade

**Teste:** `AlertTests.test_alert_cooldown_suppresses_duplicates_but_not_escalation`
em [test_app_unit.py](../tests/test_app_unit.py).

**Por que importa:** um objeto presente em muitos frames não deve gerar eventos
repetidos continuamente, mas uma piora do risco precisa aparecer imediatamente.

**Como funciona:** controla `time.perf_counter()` e publica detecções do mesmo
objeto, usando o intervalo padrão de 2,5 segundos:

| Instante | Risco | Resultado esperado |
| --- | --- | --- |
| 10 s | HIGH | Primeiro alerta |
| 11 s | HIGH | Repetição suprimida |
| 11,5 s | CRITICAL | Novo alerta imediato pela mudança de gravidade |
| 13,999 s | CRITICAL | Repetição ainda suprimida |
| 14 s | CRITICAL | Novo alerta ao completar 2,5 segundos |

O teste verifica os contadores durante a sequência e os três eventos finais.
O relógio simulado permite verificar o limite exato sem esperar de verdade.

### 4. Configuração inválida não altera o estado ativo

**Teste:** `SettingsAndMissionTests.test_invalid_settings_preserve_entire_configuration_and_active_detections`
em [test_app_unit.py](../tests/test_app_unit.py).

**Por que importa:** um pedido inválido não pode aplicar parcialmente mudanças
ao processamento em andamento nem apagar detecções válidas.

**Como funciona:** guarda a configuração completa e uma detecção ativa. Envia
pedidos com limiares fora de ordem ou iguais, tamanho de imagem incompatível e
campo desconhecido. Um dos pedidos combina uma confiança válida com um limiar
inválido, verificando também que o campo válido não é aplicado parcialmente.

Para cada pedido, espera status 422, a mesma instância de configuração em
`app.settings` e `config.settings`, todos os valores anteriores e as mesmas
detecções. Também verifica que nenhuma sessão de banco é aberta e nenhum evento
é emitido.

### 5. Resultados vencidos não aparecem como ameaça atual

**Teste:** `AlertTests.test_snapshot_hides_expired_missing_or_reconfigured_results`
em [test_app_unit.py](../tests/test_app_unit.py).

**Por que importa:** o dashboard não deve apresentar como atual uma ameaça
calculada a partir de um frame antigo ou de uma configuração substituída.

**Como funciona:** deixa uma detecção `HIGH` no estado e simula uma trilha de
vídeo ativa. Executa três cenários: resultado com dois segundos de idade
(o limite é um segundo), ausência de resultado e resultado associado a outra
instância de configuração.

Em todos os cenários, `snapshot()` deve retornar uma lista de detecções vazia,
`primary_threat=None` e zero objetos. O teste também verifica que a latência
medida quando a inferência terminou é preservada, mesmo quando a idade atual
do resultado aumenta. Um teste complementar verifica que resultados ainda
válidos permanecem visíveis e são ordenados por gravidade e proximidade.

## Demais cenários adicionados

| Área | Exemplos verificados |
| --- | --- |
| Risco | Limites SAFE/LOW/MEDIUM/HIGH, confirmação por objeto, redução de risco com um frame extra e área de interseção |
| Proximidade | Suavização, tempo decorrido, mudança de resolução, janela de histórico e IDs reaparecendo |
| Processamento | Contrato das detecções, desaparecimento de objetos, mudança de configuração e objetos sem tracking |
| Alertas | Intervalo independente por objeto, contagem por gravidade e falhas do detector ou banco |
| Configuração | Aplicação válida com override explícito, remoção de resultados antigos e dados enviados à persistência |
| Telemetria | Coordenadas e sensores com valor zero, ausência de dados e associação à missão atual |
| Missões | Limpeza dos alertas ao iniciar, ID retornado pela sessão, duração e contagem final ao encerrar |

Os testes anteriores continuam cobrindo seleção de dispositivo, fallback de GPU
para CPU, classes do detector, descarte de frames, timestamps, presets e
validação do dataset de treinamento.
