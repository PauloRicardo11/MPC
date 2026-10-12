"""
============================================================
Módulo: Simulacao_MPC_2R2C.py

Descrição:
    Funções destinadas à simulação do controlador MPC baseado
    no modelo térmico 2R2C segundo o princípio de horizonte
    deslizante (receding horizon).

    Em cada instante de amostragem:

        1. é construída a previsão de temperatura exterior;
        2. o problema MPC é resolvido para o horizonte Np;
        3. apenas a primeira ação ótima é aplicada;
        4. o estado térmico é atualizado durante uma amostra;
        5. o horizonte é deslocado e o processo é repetido.

    O horizonte de controlo Nc e a parametrização da potência
    em blocos são tratados no módulo MPC_2R2C. A reotimização
    é realizada a cada período de amostragem, pelo que os blocos
    reduzem a dimensão interna do problema de otimização sem
    impor uma permanência rígida da potência aplicada durante
    vários períodos sucessivos.

    O módulo regista:
        - estados térmicos;
        - temperatura interior;
        - temperatura exterior;
        - referência térmica;
        - potência aplicada;
        - estado de convergência;
        - custo ótimo;
        - número de iterações;
        - número de avaliações;
        - tempo computacional de cada otimização.

Autor:
    Paulo Ribeiro

Projeto:
    Plataforma de Processamento de Dados Experimentais
    Dissertação de Mestrado
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from Codigos.MPC_2R2C import (
    ConfiguracaoMPC2R2C,
    preparar_referencia_temperatura,
    prever_trajetoria_mpc,
    resolver_mpc_2r2c,
    validar_configuracao_mpc,
    validar_estado_inicial_mpc,
    validar_matrizes_discretas_mpc,
)


# ============================================================
# Estruturas de resultados
# ============================================================


@dataclass(frozen=True)
class ResultadoSimulacaoMPC2R2C:
    """
    Resultado completo de uma simulação MPC em horizonte
    deslizante.

    Atributos
    ---------
    estados : np.ndarray
        Estados térmicos ao longo da simulação. Possui dimensão
        (N + 1, 2), incluindo o estado inicial.

    temperatura_interior_c : np.ndarray
        Temperatura interior associada aos estados. Possui
        dimensão (N + 1,).

    temperatura_exterior_c : np.ndarray
        Temperatura exterior aplicada em cada instante.
        Possui dimensão (N,).

    referencia_temperatura_c : np.ndarray
        Referência térmica aplicada em cada instante.
        Possui dimensão (N,).

    potencia_aplicada_w : np.ndarray
        Primeira ação ótima efetivamente aplicada em cada
        instante. Possui dimensão (N,).

    sucesso_otimizacao : np.ndarray
        Indicador de convergência do problema MPC em cada
        instante.

    estado_otimizador : np.ndarray
        Código de estado devolvido pelo SLSQP em cada instante.

    custo_total : np.ndarray
        Valor ótimo da função objetivo em cada instante.

    numero_iteracoes : np.ndarray
        Número de iterações do SLSQP em cada instante.

    numero_avaliacoes : np.ndarray
        Número de avaliações da função objetivo em cada instante.

    tempo_otimizacao_s : np.ndarray
        Tempo computacional de cada otimização [s].

    mensagens_otimizador : tuple[str, ...]
        Mensagens devolvidas pelo algoritmo de otimização.
    """

    estados: np.ndarray
    temperatura_interior_c: np.ndarray

    temperatura_exterior_c: np.ndarray
    referencia_temperatura_c: np.ndarray
    potencia_aplicada_w: np.ndarray

    sucesso_otimizacao: np.ndarray
    estado_otimizador: np.ndarray

    custo_total: np.ndarray

    numero_iteracoes: np.ndarray
    numero_avaliacoes: np.ndarray
    tempo_otimizacao_s: np.ndarray

    mensagens_otimizador: tuple[str, ...]

    @property
    def numero_passos(self) -> int:
        """
        Número de períodos simulados.
        """

        return int(
            self.potencia_aplicada_w.size
        )

    @property
    def todas_otimizacoes_convergiram(self) -> bool:
        """
        Indica se todas as otimizações convergiram.
        """

        return bool(
            np.all(
                self.sucesso_otimizacao
            )
        )

    @property
    def taxa_convergencia_percentagem(self) -> float:
        """
        Percentagem de otimizações que convergiram.
        """

        if self.numero_passos == 0:
            return 0.0

        return float(
            100.0
            * np.mean(
                self.sucesso_otimizacao
            )
        )

    @property
    def tempo_otimizacao_total_s(self) -> float:
        """
        Tempo computacional acumulado das otimizações [s].
        """

        return float(
            np.sum(
                self.tempo_otimizacao_s
            )
        )

    @property
    def tempo_otimizacao_medio_s(self) -> float:
        """
        Tempo médio de otimização por período [s].
        """

        if self.numero_passos == 0:
            return 0.0

        return float(
            np.mean(
                self.tempo_otimizacao_s
            )
        )


# ============================================================
# Validações auxiliares
# ============================================================


def _validar_serie_numerica(
    valores,
    nome: str,
) -> np.ndarray:
    """
    Converte e valida uma série numérica unidimensional.
    """

    try:

        valores = np.asarray(
            valores,
            dtype=float,
        )

    except (TypeError, ValueError) as erro:

        raise TypeError(
            f"A série '{nome}' deve conter valores numéricos."
        ) from erro

    if valores.ndim != 1:

        raise ValueError(
            f"A série '{nome}' deve ser unidimensional."
        )

    if valores.size == 0:

        raise ValueError(
            f"A série '{nome}' não pode ser vazia."
        )

    if not np.all(
        np.isfinite(
            valores
        )
    ):

        raise ValueError(
            f"A série '{nome}' contém valores não finitos."
        )

    return valores


def _validar_numero_passos(
    numero_passos,
    numero_amostras_disponiveis: int,
) -> int:
    """
    Valida o número de períodos a simular.
    """

    if numero_passos is None:

        return int(
            numero_amostras_disponiveis
        )

    if not isinstance(
        numero_passos,
        (int, np.integer),
    ):

        raise TypeError(
            "O número de passos da simulação deve ser inteiro."
        )

    numero_passos = int(
        numero_passos
    )

    if numero_passos <= 0:

        raise ValueError(
            "O número de passos da simulação deve ser "
            "estritamente positivo."
        )

    if numero_passos > numero_amostras_disponiveis:

        raise ValueError(
            "O número de passos solicitado é superior ao número "
            "de amostras de temperatura exterior disponíveis."
        )

    return numero_passos


def _preparar_serie_referencia(
    temperatura_referencia,
    numero_passos: int,
    configuracao: ConfiguracaoMPC2R2C,
) -> np.ndarray:
    """
    Prepara a referência térmica utilizada durante a simulação.

    A referência pode ser:
        - None, utilizando o valor definido na configuração;
        - escalar;
        - vetor com pelo menos numero_passos amostras.
    """

    if temperatura_referencia is None:

        return np.full(
            numero_passos,
            configuracao.temperatura_referencia,
            dtype=float,
        )

    if np.isscalar(
        temperatura_referencia
    ):

        temperatura_referencia = float(
            temperatura_referencia
        )

        if not np.isfinite(
            temperatura_referencia
        ):

            raise ValueError(
                "A temperatura de referência deve ser finita."
            )

        return np.full(
            numero_passos,
            temperatura_referencia,
            dtype=float,
        )

    referencia = _validar_serie_numerica(
        temperatura_referencia,
        "temperatura_referencia",
    )

    if referencia.size < numero_passos:

        raise ValueError(
            "O vetor de referência possui menos amostras do "
            "que o número de passos da simulação."
        )

    return referencia[
        :numero_passos
    ].copy()


def _construir_janela_previsao(
    serie: np.ndarray,
    indice_inicial: int,
    comprimento: int,
) -> np.ndarray:
    """
    Constrói uma janela futura com comprimento fixo.

    Quando a série termina antes do final do horizonte, o último
    valor disponível é mantido constante. Esta extensão apenas
    afeta os instantes finais de uma série finita.
    """

    indice_final = (
        indice_inicial
        + comprimento
    )

    if indice_final <= serie.size:

        return serie[
            indice_inicial:indice_final
        ].copy()

    janela_disponivel = serie[
        indice_inicial:
    ]

    if janela_disponivel.size == 0:

        valor_final = float(
            serie[-1]
        )

        return np.full(
            comprimento,
            valor_final,
            dtype=float,
        )

    numero_em_falta = (
        comprimento
        - janela_disponivel.size
    )

    extensao = np.full(
        numero_em_falta,
        janela_disponivel[-1],
        dtype=float,
    )

    return np.concatenate(
        (
            janela_disponivel,
            extensao,
        )
    )


def _preparar_estimativa_inicial_deslizante(
    potencia_prevista_anterior_w: np.ndarray | None,
    potencia_anterior_w: float,
    horizonte_predicao: int,
) -> np.ndarray:
    """
    Constrói a estimativa inicial para a otimização seguinte.

    Quando existe uma solução anterior, a sequência ótima é
    deslocada uma amostra para a esquerda e o último valor é
    repetido. Caso contrário, utiliza-se a potência anteriormente
    aplicada em todo o horizonte.
    """

    if potencia_prevista_anterior_w is None:

        return np.full(
            horizonte_predicao,
            potencia_anterior_w,
            dtype=float,
        )

    potencia_prevista_anterior_w = np.asarray(
        potencia_prevista_anterior_w,
        dtype=float,
    )

    if potencia_prevista_anterior_w.shape != (
        horizonte_predicao,
    ):

        raise ValueError(
            "A sequência de potência da solução anterior possui "
            "dimensão incompatível com o horizonte de predição."
        )

    potencia_inicial_w = np.empty(
        horizonte_predicao,
        dtype=float,
    )

    potencia_inicial_w[:-1] = (
        potencia_prevista_anterior_w[1:]
    )

    potencia_inicial_w[-1] = (
        potencia_prevista_anterior_w[-1]
    )

    return potencia_inicial_w


# ============================================================
# Simulação MPC em horizonte deslizante
# ============================================================


def simular_mpc_horizonte_deslizante(
    matrizes,
    estado_inicial: np.ndarray,
    temperatura_exterior_c: np.ndarray,
    configuracao: ConfiguracaoMPC2R2C,
    temperatura_referencia=None,
    potencia_inicial_w: float = 0.0,
    numero_passos: int | None = None,
    maximo_iteracoes: int = 300,
    utilizar_warm_start: bool = True,
    interromper_em_falha: bool = True,
) -> ResultadoSimulacaoMPC2R2C:
    """
    Executa uma simulação MPC em horizonte deslizante.

    Parameters
    ----------
    matrizes
        Matrizes discretas do modelo térmico 2R2C.

    estado_inicial : np.ndarray
        Estado inicial [T_int, T_m].

    temperatura_exterior_c : np.ndarray
        Série de temperatura exterior [°C]. Os valores futuros
        são utilizados como previsão conhecida ao longo de Np.

    configuracao : ConfiguracaoMPC2R2C
        Configuração do controlador MPC.

    temperatura_referencia : float ou np.ndarray, optional
        Referência térmica. Quando omitida, utiliza-se o valor
        definido na configuração.

    potencia_inicial_w : float, default=0.0
        Potência aplicada imediatamente antes do início da
        simulação [W].

    numero_passos : int, optional
        Número de períodos a simular. Quando omitido, utiliza-se
        toda a série de temperatura exterior.

    maximo_iteracoes : int, default=300
        Número máximo de iterações de cada problema MPC.

    utilizar_warm_start : bool, default=True
        Quando True, a solução ótima anterior é deslocada e
        utilizada como estimativa inicial da otimização seguinte.

    interromper_em_falha : bool, default=True
        Quando True, uma falha de convergência interrompe a
        simulação. Quando False, é aplicada a primeira ação
        devolvida pelo otimizador e a falha fica registada.

    Returns
    -------
    ResultadoSimulacaoMPC2R2C
        Resultado completo da simulação em horizonte deslizante.
    """

    validar_matrizes_discretas_mpc(
        matrizes
    )

    validar_configuracao_mpc(
        configuracao
    )

    estado_atual = validar_estado_inicial_mpc(
        estado_inicial
    ).copy()

    temperatura_exterior_c = (
        _validar_serie_numerica(
            temperatura_exterior_c,
            "temperatura_exterior_c",
        )
    )

    numero_passos = _validar_numero_passos(
        numero_passos,
        temperatura_exterior_c.size,
    )

    referencia_simulacao_c = (
        _preparar_serie_referencia(
            temperatura_referencia,
            numero_passos,
            configuracao,
        )
    )

    potencia_anterior_w = float(
        potencia_inicial_w
    )

    if not np.isfinite(
        potencia_anterior_w
    ):

        raise ValueError(
            "A potência inicial deve ser finita."
        )

    if (
        potencia_anterior_w
        < configuracao.potencia_minima_w
        or potencia_anterior_w
        > configuracao.potencia_maxima_w
    ):

        raise ValueError(
            "A potência inicial encontra-se fora dos limites "
            "definidos na configuração do MPC."
        )

    if not isinstance(
        maximo_iteracoes,
        (int, np.integer),
    ):

        raise TypeError(
            "O número máximo de iterações deve ser inteiro."
        )

    maximo_iteracoes = int(
        maximo_iteracoes
    )

    if maximo_iteracoes <= 0:

        raise ValueError(
            "O número máximo de iterações deve ser "
            "estritamente positivo."
        )

    if not isinstance(
        utilizar_warm_start,
        (bool, np.bool_),
    ):

        raise TypeError(
            "utilizar_warm_start deve ser booleano."
        )

    if not isinstance(
        interromper_em_falha,
        (bool, np.bool_),
    ):

        raise TypeError(
            "interromper_em_falha deve ser booleano."
        )

    horizonte_predicao = (
        configuracao.horizonte_predicao
    )

    estados = np.empty(
        (
            numero_passos + 1,
            2,
        ),
        dtype=float,
    )

    potencia_aplicada_w = np.empty(
        numero_passos,
        dtype=float,
    )

    sucesso_otimizacao = np.empty(
        numero_passos,
        dtype=bool,
    )

    estado_otimizador = np.empty(
        numero_passos,
        dtype=int,
    )

    custo_total = np.empty(
        numero_passos,
        dtype=float,
    )

    numero_iteracoes = np.empty(
        numero_passos,
        dtype=int,
    )

    numero_avaliacoes = np.empty(
        numero_passos,
        dtype=int,
    )

    tempo_otimizacao_s = np.empty(
        numero_passos,
        dtype=float,
    )

    mensagens_otimizador = []

    estados[0] = estado_atual

    potencia_prevista_anterior_w = None

    for indice in range(
        numero_passos
    ):

        previsao_temperatura_exterior_c = (
            _construir_janela_previsao(
                temperatura_exterior_c,
                indice,
                horizonte_predicao,
            )
        )

        previsao_referencia_c = (
            _construir_janela_previsao(
                referencia_simulacao_c,
                indice,
                horizonte_predicao,
            )
        )

        if utilizar_warm_start:

            estimativa_inicial_w = (
                _preparar_estimativa_inicial_deslizante(
                    potencia_prevista_anterior_w,
                    potencia_anterior_w,
                    horizonte_predicao,
                )
            )

        else:

            estimativa_inicial_w = np.full(
                horizonte_predicao,
                potencia_anterior_w,
                dtype=float,
            )

        instante_inicial = perf_counter()

        resultado_mpc = resolver_mpc_2r2c(
            matrizes=matrizes,
            estado_inicial=estado_atual,
            temperatura_exterior_prevista=(
                previsao_temperatura_exterior_c
            ),
            potencia_anterior_w=(
                potencia_anterior_w
            ),
            configuracao=configuracao,
            temperatura_referencia=(
                previsao_referencia_c
            ),
            potencia_inicial_w=(
                estimativa_inicial_w
            ),
            maximo_iteracoes=(
                maximo_iteracoes
            ),
        )

        tempo_otimizacao_s[indice] = (
            perf_counter()
            - instante_inicial
        )

        sucesso_otimizacao[indice] = (
            resultado_mpc.sucesso
        )

        estado_otimizador[indice] = (
            resultado_mpc.estado_otimizador
        )

        custo_total[indice] = (
            resultado_mpc
            .componentes_custo
            .custo_total
        )

        numero_iteracoes[indice] = (
            resultado_mpc.numero_iteracoes
        )

        numero_avaliacoes[indice] = (
            resultado_mpc.numero_avaliacoes
        )

        mensagens_otimizador.append(
            resultado_mpc.mensagem
        )

        if (
            not resultado_mpc.sucesso
            and interromper_em_falha
        ):

            raise RuntimeError(
                "A otimização MPC não convergiu no passo "
                f"{indice}.\n"
                f"Estado: {resultado_mpc.estado_otimizador}\n"
                f"Mensagem: {resultado_mpc.mensagem}"
            )

        potencia_atual_w = float(
            resultado_mpc.primeira_acao_w
        )

        potencia_aplicada_w[indice] = (
            potencia_atual_w
        )

        (
            estados_um_passo,
            _,
        ) = prever_trajetoria_mpc(
            matrizes=matrizes,
            estado_inicial=estado_atual,
            temperatura_exterior_prevista=np.array(
                [
                    temperatura_exterior_c[indice]
                ],
                dtype=float,
            ),
            potencia_w=np.array(
                [
                    potencia_atual_w
                ],
                dtype=float,
            ),
        )

        estado_atual = (
            estados_um_passo[-1].copy()
        )

        estados[indice + 1] = (
            estado_atual
        )

        potencia_anterior_w = (
            potencia_atual_w
        )

        potencia_prevista_anterior_w = (
            resultado_mpc.potencia_otima_w.copy()
        )

    temperatura_interior_c = (
        estados[:, 0].copy()
    )

    return ResultadoSimulacaoMPC2R2C(
        estados=estados,
        temperatura_interior_c=(
            temperatura_interior_c
        ),
        temperatura_exterior_c=(
            temperatura_exterior_c[
                :numero_passos
            ].copy()
        ),
        referencia_temperatura_c=(
            referencia_simulacao_c.copy()
        ),
        potencia_aplicada_w=(
            potencia_aplicada_w
        ),
        sucesso_otimizacao=(
            sucesso_otimizacao
        ),
        estado_otimizador=(
            estado_otimizador
        ),
        custo_total=(
            custo_total
        ),
        numero_iteracoes=(
            numero_iteracoes
        ),
        numero_avaliacoes=(
            numero_avaliacoes
        ),
        tempo_otimizacao_s=(
            tempo_otimizacao_s
        ),
        mensagens_otimizador=tuple(
            mensagens_otimizador
        ),
    )


# ============================================================
# Validação do resultado da simulação
# ============================================================


def validar_resultado_simulacao_mpc(
    resultado: ResultadoSimulacaoMPC2R2C,
    configuracao: ConfiguracaoMPC2R2C,
) -> None:
    """
    Valida estrutural e numericamente o resultado da simulação.
    """

    if not isinstance(
        resultado,
        ResultadoSimulacaoMPC2R2C,
    ):

        raise TypeError(
            "O resultado deve ser uma instância de "
            "ResultadoSimulacaoMPC2R2C."
        )

    validar_configuracao_mpc(
        configuracao
    )

    numero_passos = (
        resultado.numero_passos
    )

    if resultado.estados.shape != (
        numero_passos + 1,
        2,
    ):

        raise ValueError(
            "Os estados da simulação apresentam dimensão "
            "incorreta."
        )

    if resultado.temperatura_interior_c.shape != (
        numero_passos + 1,
    ):

        raise ValueError(
            "A temperatura interior apresenta dimensão incorreta."
        )

    vetores_numero_passos = {
        "temperatura_exterior_c":
            resultado.temperatura_exterior_c,

        "referencia_temperatura_c":
            resultado.referencia_temperatura_c,

        "potencia_aplicada_w":
            resultado.potencia_aplicada_w,

        "sucesso_otimizacao":
            resultado.sucesso_otimizacao,

        "estado_otimizador":
            resultado.estado_otimizador,

        "custo_total":
            resultado.custo_total,

        "numero_iteracoes":
            resultado.numero_iteracoes,

        "numero_avaliacoes":
            resultado.numero_avaliacoes,

        "tempo_otimizacao_s":
            resultado.tempo_otimizacao_s,
    }

    for nome, vetor in (
        vetores_numero_passos.items()
    ):

        if vetor.shape != (
            numero_passos,
        ):

            raise ValueError(
                f"O vetor '{nome}' apresenta dimensão incorreta."
            )

    vetores_finitos = (
        resultado.estados,
        resultado.temperatura_interior_c,
        resultado.temperatura_exterior_c,
        resultado.referencia_temperatura_c,
        resultado.potencia_aplicada_w,
        resultado.custo_total,
        resultado.tempo_otimizacao_s,
    )

    if not all(
        np.all(
            np.isfinite(
                vetor
            )
        )
        for vetor in vetores_finitos
    ):

        raise ValueError(
            "O resultado da simulação contém valores não finitos."
        )

    if np.any(
        resultado.potencia_aplicada_w
        < configuracao.potencia_minima_w
        - 1.0e-6
    ):

        raise ValueError(
            "A simulação viola o limite inferior de potência."
        )

    if np.any(
        resultado.potencia_aplicada_w
        > configuracao.potencia_maxima_w
        + 1.0e-6
    ):

        raise ValueError(
            "A simulação viola o limite superior de potência."
        )

    if len(
        resultado.mensagens_otimizador
    ) != numero_passos:

        raise ValueError(
            "O número de mensagens do otimizador é incompatível "
            "com o número de passos da simulação."
        )


# ============================================================
# Apresentação do resultado
# ============================================================


def mostrar_resumo_simulacao_mpc(
    resultado: ResultadoSimulacaoMPC2R2C,
    configuracao: ConfiguracaoMPC2R2C,
) -> None:
    """
    Apresenta um resumo numérico da simulação MPC.
    """

    validar_resultado_simulacao_mpc(
        resultado,
        configuracao,
    )

    print(
        "=" * 70
    )

    print(
        "SIMULAÇÃO MPC EM HORIZONTE DESLIZANTE"
    )

    print(
        "=" * 70
    )

    print(
        f"Número de passos simulados: "
        f"{resultado.numero_passos}"
    )

    print(
        f"Taxa de convergência: "
        f"{resultado.taxa_convergencia_percentagem:.2f} %"
    )

    print()

    print(
        "Temperatura interior:"
    )

    print(
        f"Inicial = "
        f"{resultado.temperatura_interior_c[0]:.3f} °C"
    )

    print(
        f"Final = "
        f"{resultado.temperatura_interior_c[-1]:.3f} °C"
    )

    print(
        f"Mínima = "
        f"{resultado.temperatura_interior_c.min():.3f} °C"
    )

    print(
        f"Máxima = "
        f"{resultado.temperatura_interior_c.max():.3f} °C"
    )

    print()

    print(
        "Potência aplicada:"
    )

    print(
        f"Mínima = "
        f"{resultado.potencia_aplicada_w.min():.2f} W"
    )

    print(
        f"Máxima = "
        f"{resultado.potencia_aplicada_w.max():.2f} W"
    )

    print(
        f"Média = "
        f"{resultado.potencia_aplicada_w.mean():.2f} W"
    )

    print()

    print(
        "Esforço computacional:"
    )

    print(
        f"Iterações médias = "
        f"{resultado.numero_iteracoes.mean():.2f}"
    )

    print(
        f"Avaliações médias = "
        f"{resultado.numero_avaliacoes.mean():.2f}"
    )

    print(
        f"Tempo médio por otimização = "
        f"{resultado.tempo_otimizacao_medio_s:.4f} s"
    )

    print(
        f"Tempo acumulado de otimização = "
        f"{resultado.tempo_otimizacao_total_s:.4f} s"
    )

    print(
        "=" * 70
    )
