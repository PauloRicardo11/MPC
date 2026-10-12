"""
============================================================
Módulo: MPC_2R2C.py

Descrição:
    Implementação inicial de um controlador preditivo baseado
    em modelo para o sistema térmico 2R2C desenvolvido no
    âmbito da dissertação.

    O controlador utiliza o modelo discreto:

        x[k+1] = Ad x[k] + Bd u[k]

        y[k] = Cd x[k] + Dd u[k]

    com:

        x = [T_int, T_m]

        u = [T_ext, P_el]

    onde:

        T_ext
            Temperatura exterior, considerada uma perturbação
            conhecida ou prevista ao longo do horizonte.

        P_el
            Potência elétrica equivalente do sistema AVAC,
            utilizada como variável manipulada pelo MPC.

    A função objetivo considera quatro componentes:

        1. erro relativamente à temperatura de referência;
        2. utilização de potência elétrica;
        3. variação da potência elétrica;
        4. violação da banda de conforto.

    A banda de conforto é tratada através de uma penalização
    suave, garantindo que o problema de otimização permanece
    resolúvel mesmo quando as condições térmicas tornam
    impossível cumprir integralmente os limites definidos.

    Não é imposta uma restrição rígida sobre a variação de
    potência, uma vez que os dados experimentais demonstraram
    a existência de transições elevadas durante os arranques e
    paragens do equipamento.

    Para reduzir a dimensão do problema de otimização, é
    utilizado um horizonte de controlo Nc inferior ao horizonte
    de predição Np. Na configuração inicial, Np = 120 e Nc = 12,
    pelo que cada ação de controlo é mantida durante 10 minutos.

    As variáveis de decisão do SLSQP são normalizadas para o
    intervalo [0, 1] e posteriormente convertidas para potência
    elétrica [W]. Esta transformação melhora o condicionamento
    numérico sem alterar os limites físicos nem a função objetivo.

Método de otimização:
    scipy.optimize.minimize
    método SLSQP

Autor:
    Paulo Ribeiro

Projeto:
    Plataforma de Processamento de Dados Experimentais
    Dissertação de Mestrado
============================================================
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


# ============================================================
# Estruturas de dados
# ============================================================


@dataclass(frozen=True)
class ConfiguracaoMPC2R2C:
    """
    Configuração do controlador MPC.

    Atributos
    ---------
    horizonte_predicao : int
        Número de passos do horizonte de predição.

    horizonte_controlo : int
        Número de ações de controlo independentes ao longo do
        horizonte de predição.

    temperatura_referencia : float
        Temperatura interior de referência [°C].

    temperatura_minima : float
        Limite inferior da banda de conforto [°C].

    temperatura_maxima : float
        Limite superior da banda de conforto [°C].

    potencia_minima_w : float
        Potência elétrica mínima admissível [W].

    potencia_maxima_w : float
        Potência elétrica máxima admissível [W].

    peso_temperatura : float
        Peso associado ao erro de seguimento da temperatura.

    peso_potencia : float
        Peso associado à utilização da potência elétrica.

    peso_variacao_potencia : float
        Peso associado às alterações da potência elétrica.

    peso_violacao_conforto : float
        Peso associado à violação da banda de conforto.
    """

    horizonte_predicao: int
    horizonte_controlo: int

    temperatura_referencia: float

    temperatura_minima: float
    temperatura_maxima: float

    potencia_minima_w: float
    potencia_maxima_w: float

    peso_temperatura: float
    peso_potencia: float
    peso_variacao_potencia: float
    peso_violacao_conforto: float


@dataclass(frozen=True)
class ComponentesCustoMPC2R2C:
    """
    Componentes individuais da função objetivo do MPC.
    """

    custo_temperatura: float
    custo_potencia: float
    custo_variacao_potencia: float
    custo_violacao_conforto: float
    custo_total: float


@dataclass(frozen=True)
class ResultadoMPC2R2C:
    """
    Resultado de uma otimização MPC.

    Atributos
    ---------
    potencia_otima_w : np.ndarray
        Sequência ótima de potência ao longo do horizonte.

    estados_previstos : np.ndarray
        Estados previstos, incluindo o estado inicial.

    temperatura_prevista : np.ndarray
        Temperatura interior prevista ao longo do horizonte.

    componentes_custo : ComponentesCustoMPC2R2C
        Componentes da função objetivo.

    sucesso : bool
        Estado de convergência do algoritmo.

    estado_otimizador : int
        Código de estado devolvido pelo otimizador.

    mensagem : str
        Mensagem devolvida pelo algoritmo.

    numero_iteracoes : int
        Número de iterações executadas.

    numero_avaliacoes : int
        Número de avaliações da função objetivo.

    primeira_acao_w : float
        Primeira ação ótima do horizonte, correspondente à
        ação que seria efetivamente aplicada ao sistema.
    """

    potencia_otima_w: np.ndarray

    estados_previstos: np.ndarray

    temperatura_prevista: np.ndarray

    componentes_custo: ComponentesCustoMPC2R2C

    sucesso: bool
    estado_otimizador: int
    mensagem: str

    numero_iteracoes: int
    numero_avaliacoes: int

    primeira_acao_w: float


# ============================================================
# Configuração inicial recomendada
# ============================================================


def criar_configuracao_mpc_padrao(
) -> ConfiguracaoMPC2R2C:
    """
    Cria a configuração inicial do controlador MPC.

    Os valores definidos resultam da caracterização
    experimental e das análises realizadas anteriormente.

    Returns
    -------
    ConfiguracaoMPC2R2C
        Configuração inicial do controlador.
    """

    configuracao = ConfiguracaoMPC2R2C(
        horizonte_predicao=120,
        horizonte_controlo=12,
        temperatura_referencia=20.0,
        temperatura_minima=19.5,
        temperatura_maxima=20.5,
        potencia_minima_w=0.0,
        potencia_maxima_w=4760.0,
        peso_temperatura=1.0,
        peso_potencia=0.05,
        peso_variacao_potencia=0.10,
        peso_violacao_conforto=100.0,
    )

    validar_configuracao_mpc(
        configuracao
    )

    return configuracao


# ============================================================
# Validação da configuração
# ============================================================


def validar_configuracao_mpc(
    configuracao: ConfiguracaoMPC2R2C,
) -> None:
    """
    Valida a configuração do controlador MPC.
    """

    if not isinstance(
        configuracao,
        ConfiguracaoMPC2R2C,
    ):
        raise TypeError(
            "A configuração deve ser uma instância de "
            "ConfiguracaoMPC2R2C."
        )

    if not isinstance(
        configuracao.horizonte_predicao,
        (int, np.integer),
    ):
        raise TypeError(
            "O horizonte de predição deve ser inteiro."
        )

    if configuracao.horizonte_predicao <= 0:
        raise ValueError(
            "O horizonte de predição deve ser "
            "estritamente positivo."
        )

    if not isinstance(
        configuracao.horizonte_controlo,
        (int, np.integer),
    ):
        raise TypeError(
            "O horizonte de controlo deve ser inteiro."
        )

    if configuracao.horizonte_controlo <= 0:
        raise ValueError(
            "O horizonte de controlo deve ser "
            "estritamente positivo."
        )

    if (
        configuracao.horizonte_controlo
        > configuracao.horizonte_predicao
    ):
        raise ValueError(
            "O horizonte de controlo não pode ser superior "
            "ao horizonte de predição."
        )

    if (
        configuracao.horizonte_predicao
        % configuracao.horizonte_controlo
        != 0
    ):
        raise ValueError(
            "O horizonte de predição deve ser divisível pelo "
            "horizonte de controlo para permitir blocos de "
            "igual duração."
        )

    valores_numericos = {
        "temperatura_referencia":
            configuracao.temperatura_referencia,

        "temperatura_minima":
            configuracao.temperatura_minima,

        "temperatura_maxima":
            configuracao.temperatura_maxima,

        "potencia_minima_w":
            configuracao.potencia_minima_w,

        "potencia_maxima_w":
            configuracao.potencia_maxima_w,

        "peso_temperatura":
            configuracao.peso_temperatura,

        "peso_potencia":
            configuracao.peso_potencia,

        "peso_variacao_potencia":
            configuracao.peso_variacao_potencia,

        "peso_violacao_conforto":
            configuracao.peso_violacao_conforto,
    }

    for nome, valor in valores_numericos.items():

        if not isinstance(
            valor,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError(
                f"O parâmetro '{nome}' deve ser numérico."
            )

        if not np.isfinite(
            float(valor)
        ):
            raise ValueError(
                f"O parâmetro '{nome}' deve ser finito."
            )

    if (
        configuracao.temperatura_minima
        >= configuracao.temperatura_maxima
    ):
        raise ValueError(
            "A temperatura mínima deve ser inferior "
            "à temperatura máxima."
        )

    if not (
        configuracao.temperatura_minima
        <= configuracao.temperatura_referencia
        <= configuracao.temperatura_maxima
    ):
        raise ValueError(
            "A temperatura de referência deve estar "
            "dentro da banda de conforto."
        )

    if configuracao.potencia_minima_w < 0:
        raise ValueError(
            "A potência mínima não pode ser negativa."
        )

    if (
        configuracao.potencia_maxima_w
        <= configuracao.potencia_minima_w
    ):
        raise ValueError(
            "A potência máxima deve ser superior "
            "à potência mínima."
        )

    pesos = (
        configuracao.peso_temperatura,
        configuracao.peso_potencia,
        configuracao.peso_variacao_potencia,
        configuracao.peso_violacao_conforto,
    )

    if any(
        peso < 0
        for peso in pesos
    ):
        raise ValueError(
            "Os pesos da função objetivo não podem "
            "ser negativos."
        )

    if configuracao.peso_temperatura <= 0:
        raise ValueError(
            "O peso da temperatura deve ser "
            "estritamente positivo."
        )

    if configuracao.peso_violacao_conforto <= 0:
        raise ValueError(
            "O peso da violação de conforto deve ser "
            "estritamente positivo."
        )


# ============================================================
# Validação das matrizes discretas
# ============================================================


def validar_matrizes_discretas_mpc(
    matrizes,
) -> None:
    """
    Valida as matrizes discretas utilizadas pelo MPC.
    """

    atributos = (
        "Ad",
        "Bd",
        "Cd",
        "Dd",
    )

    for atributo in atributos:

        if not hasattr(
            matrizes,
            atributo,
        ):
            raise TypeError(
                "A estrutura das matrizes discretas "
                f"não contém '{atributo}'."
            )

    matrizes_dict = {
        "Ad": np.asarray(
            matrizes.Ad,
            dtype=float,
        ),
        "Bd": np.asarray(
            matrizes.Bd,
            dtype=float,
        ),
        "Cd": np.asarray(
            matrizes.Cd,
            dtype=float,
        ),
        "Dd": np.asarray(
            matrizes.Dd,
            dtype=float,
        ),
    }

    dimensoes_esperadas = {
        "Ad": (2, 2),
        "Bd": (2, 2),
        "Cd": (1, 2),
        "Dd": (1, 2),
    }

    for nome, matriz in matrizes_dict.items():

        if matriz.shape != dimensoes_esperadas[nome]:

            raise ValueError(
                f"A matriz {nome} deve possuir dimensão "
                f"{dimensoes_esperadas[nome]}."
            )

        if not np.all(
            np.isfinite(
                matriz
            )
        ):
            raise ValueError(
                f"A matriz {nome} contém valores "
                "não finitos."
            )


# ============================================================
# Validação do estado inicial
# ============================================================


def validar_estado_inicial_mpc(
    estado_inicial: np.ndarray,
) -> np.ndarray:
    """
    Valida o estado inicial utilizado pelo MPC.
    """

    try:

        estado_inicial = np.asarray(
            estado_inicial,
            dtype=float,
        )

    except (TypeError, ValueError) as erro:

        raise TypeError(
            "O estado inicial deve conter apenas "
            "valores numéricos."
        ) from erro

    if estado_inicial.shape != (2,):

        raise ValueError(
            "O estado inicial deve possuir dimensão (2,)."
        )

    if not np.all(
        np.isfinite(
            estado_inicial
        )
    ):
        raise ValueError(
            "O estado inicial contém valores não finitos."
        )

    return estado_inicial


# ============================================================
# Validação da previsão de temperatura exterior
# ============================================================


def validar_previsao_temperatura_exterior(
    temperatura_exterior_prevista: np.ndarray,
    horizonte_predicao: int,
) -> np.ndarray:
    """
    Valida a previsão de temperatura exterior.
    """

    try:

        temperatura_exterior_prevista = np.asarray(
            temperatura_exterior_prevista,
            dtype=float,
        )

    except (TypeError, ValueError) as erro:

        raise TypeError(
            "A previsão de temperatura exterior deve "
            "conter valores numéricos."
        ) from erro

    if temperatura_exterior_prevista.shape != (
        horizonte_predicao,
    ):
        raise ValueError(
            "A previsão de temperatura exterior deve "
            f"possuir dimensão ({horizonte_predicao},)."
        )

    if not np.all(
        np.isfinite(
            temperatura_exterior_prevista
        )
    ):
        raise ValueError(
            "A previsão de temperatura exterior contém "
            "valores não finitos."
        )

    return temperatura_exterior_prevista


# ============================================================
# Preparação da referência térmica
# ============================================================


def preparar_referencia_temperatura(
    temperatura_referencia,
    horizonte_predicao: int,
) -> np.ndarray:
    """
    Constrói o vetor de referência de temperatura.

    Pode ser fornecido:

        - um único valor;
        - um vetor com comprimento igual ao horizonte.
    """

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
            horizonte_predicao,
            temperatura_referencia,
            dtype=float,
        )

    try:

        temperatura_referencia = np.asarray(
            temperatura_referencia,
            dtype=float,
        )

    except (TypeError, ValueError) as erro:

        raise TypeError(
            "A referência de temperatura deve ser "
            "numérica."
        ) from erro

    if temperatura_referencia.shape != (
        horizonte_predicao,
    ):
        raise ValueError(
            "O vetor de referência deve possuir dimensão "
            f"({horizonte_predicao},)."
        )

    if not np.all(
        np.isfinite(
            temperatura_referencia
        )
    ):
        raise ValueError(
            "A referência contém valores não finitos."
        )

    return temperatura_referencia


# ============================================================
# Previsão da trajetória
# ============================================================


def prever_trajetoria_mpc(
    matrizes,
    estado_inicial: np.ndarray,
    temperatura_exterior_prevista: np.ndarray,
    potencia_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propaga o modelo discreto ao longo do horizonte.

    Parameters
    ----------
    matrizes
        Matrizes discretas do modelo.

    estado_inicial : np.ndarray
        Estado atual [T_int, T_m].

    temperatura_exterior_prevista : np.ndarray
        Temperatura exterior prevista.

    potencia_w : np.ndarray
        Sequência candidata de potência elétrica.

    Returns
    -------
    tuple
        Estados previstos e temperatura interior prevista.
    """

    validar_matrizes_discretas_mpc(
        matrizes
    )

    estado_inicial = validar_estado_inicial_mpc(
        estado_inicial
    )

    temperatura_exterior_prevista = np.asarray(
        temperatura_exterior_prevista,
        dtype=float,
    )

    potencia_w = np.asarray(
        potencia_w,
        dtype=float,
    )

    if temperatura_exterior_prevista.ndim != 1:
        raise ValueError(
            "A temperatura exterior prevista deve ser "
            "unidimensional."
        )

    if potencia_w.ndim != 1:
        raise ValueError(
            "A sequência de potência deve ser "
            "unidimensional."
        )

    if (
        temperatura_exterior_prevista.shape
        != potencia_w.shape
    ):
        raise ValueError(
            "A temperatura exterior e a potência devem "
            "possuir o mesmo comprimento."
        )

    if potencia_w.size == 0:
        raise ValueError(
            "O horizonte de previsão não pode ser vazio."
        )

    if not np.all(
        np.isfinite(
            potencia_w
        )
    ):
        raise ValueError(
            "A sequência de potência contém valores "
            "não finitos."
        )

    Ad = np.asarray(
        matrizes.Ad,
        dtype=float,
    )

    Bd = np.asarray(
        matrizes.Bd,
        dtype=float,
    )

    Cd = np.asarray(
        matrizes.Cd,
        dtype=float,
    )

    horizonte = potencia_w.size

    estados = np.empty(
        (
            horizonte + 1,
            2,
        ),
        dtype=float,
    )

    temperatura_prevista = np.empty(
        horizonte,
        dtype=float,
    )

    estados[0] = estado_inicial

    for indice in range(
        horizonte
    ):

        entrada = np.array(
            [
                temperatura_exterior_prevista[indice],
                potencia_w[indice],
            ],
            dtype=float,
        )

        estados[indice + 1] = (
            Ad @ estados[indice]
            + Bd @ entrada
        )

        temperatura_prevista[indice] = float(
            (
                Cd
                @ estados[indice + 1]
            )[0]
        )

    return (
        estados,
        temperatura_prevista,
    )


# ============================================================
# Cálculo da função objetivo
# ============================================================


def calcular_componentes_custo_mpc(
    matrizes,
    estado_inicial: np.ndarray,
    temperatura_exterior_prevista: np.ndarray,
    potencia_w: np.ndarray,
    potencia_anterior_w: float,
    referencia_temperatura: np.ndarray,
    configuracao: ConfiguracaoMPC2R2C,
) -> ComponentesCustoMPC2R2C:
    """
    Calcula as componentes da função objetivo do MPC.
    """

    validar_configuracao_mpc(
        configuracao
    )

    (
        _,
        temperatura_prevista,
    ) = prever_trajetoria_mpc(
        matrizes=matrizes,
        estado_inicial=estado_inicial,
        temperatura_exterior_prevista=(
            temperatura_exterior_prevista
        ),
        potencia_w=potencia_w,
    )

    potencia_w = np.asarray(
        potencia_w,
        dtype=float,
    )

    referencia_temperatura = np.asarray(
        referencia_temperatura,
        dtype=float,
    )

    potencia_anterior_w = float(
        potencia_anterior_w
    )

    if not np.isfinite(
        potencia_anterior_w
    ):
        raise ValueError(
            "A potência anterior deve ser finita."
        )

    erro_temperatura = (
        temperatura_prevista
        - referencia_temperatura
    )

    custo_temperatura = float(
        configuracao.peso_temperatura
        * np.sum(
            erro_temperatura ** 2
        )
    )

    potencia_normalizada = (
        potencia_w
        / configuracao.potencia_maxima_w
    )

    custo_potencia = float(
        configuracao.peso_potencia
        * np.sum(
            potencia_normalizada ** 2
        )
    )

    potencia_com_anterior = np.concatenate(
        (
            np.array(
                [
                    potencia_anterior_w
                ],
                dtype=float,
            ),
            potencia_w,
        )
    )

    variacao_potencia = np.diff(
        potencia_com_anterior
    )

    variacao_normalizada = (
        variacao_potencia
        / configuracao.potencia_maxima_w
    )

    custo_variacao_potencia = float(
        configuracao.peso_variacao_potencia
        * np.sum(
            variacao_normalizada ** 2
        )
    )

    violacao_inferior = np.maximum(
        0.0,
        configuracao.temperatura_minima
        - temperatura_prevista,
    )

    violacao_superior = np.maximum(
        0.0,
        temperatura_prevista
        - configuracao.temperatura_maxima,
    )

    custo_violacao_conforto = float(
        configuracao.peso_violacao_conforto
        * np.sum(
            violacao_inferior ** 2
            + violacao_superior ** 2
        )
    )

    custo_total = float(
        custo_temperatura
        + custo_potencia
        + custo_variacao_potencia
        + custo_violacao_conforto
    )

    return ComponentesCustoMPC2R2C(
        custo_temperatura=custo_temperatura,
        custo_potencia=custo_potencia,
        custo_variacao_potencia=(
            custo_variacao_potencia
        ),
        custo_violacao_conforto=(
            custo_violacao_conforto
        ),
        custo_total=custo_total,
    )


# ============================================================
# Normalização da variável manipulada
# ============================================================


def normalizar_potencia_mpc(
    potencia_w: np.ndarray,
    configuracao: ConfiguracaoMPC2R2C,
) -> np.ndarray:
    """
    Converte valores de potência [W] para a variável normalizada
    utilizada internamente pelo otimizador.

        v = (P - P_min) / (P_max - P_min)

    A variável normalizada pertence ao intervalo [0, 1].
    """

    validar_configuracao_mpc(configuracao)

    potencia_w = np.asarray(potencia_w, dtype=float)

    if not np.all(np.isfinite(potencia_w)):
        raise ValueError(
            "A potência contém valores não finitos."
        )

    amplitude_potencia = (
        configuracao.potencia_maxima_w
        - configuracao.potencia_minima_w
    )

    potencia_normalizada = (
        potencia_w - configuracao.potencia_minima_w
    ) / amplitude_potencia

    return np.clip(
        potencia_normalizada,
        0.0,
        1.0,
    )


def desnormalizar_potencia_mpc(
    potencia_normalizada: np.ndarray,
    configuracao: ConfiguracaoMPC2R2C,
) -> np.ndarray:
    """
    Converte a variável normalizada do otimizador para potência
    elétrica [W]:

        P = P_min + v (P_max - P_min)
    """

    validar_configuracao_mpc(configuracao)

    potencia_normalizada = np.asarray(
        potencia_normalizada,
        dtype=float,
    )

    if not np.all(np.isfinite(potencia_normalizada)):
        raise ValueError(
            "A variável de decisão contém valores não finitos."
        )

    potencia_normalizada = np.clip(
        potencia_normalizada,
        0.0,
        1.0,
    )

    amplitude_potencia = (
        configuracao.potencia_maxima_w
        - configuracao.potencia_minima_w
    )

    return (
        configuracao.potencia_minima_w
        + potencia_normalizada * amplitude_potencia
    )


# ============================================================
# Parametrização do horizonte de controlo
# ============================================================


def expandir_acoes_controlo(
    acoes_controlo_w: np.ndarray,
    configuracao: ConfiguracaoMPC2R2C,
) -> np.ndarray:
    """
    Expande as ações de controlo independentes para o horizonte
    completo de predição.

    Para a configuração padrão:

        Np = 120
        Nc = 12

    cada ação é mantida constante durante 10 amostras,
    correspondentes a 10 minutos para Ts = 60 s.
    """

    validar_configuracao_mpc(
        configuracao
    )

    try:

        acoes_controlo_w = np.asarray(
            acoes_controlo_w,
            dtype=float,
        )

    except (TypeError, ValueError) as erro:

        raise TypeError(
            "As ações de controlo devem conter apenas "
            "valores numéricos."
        ) from erro

    horizonte_controlo = (
        configuracao.horizonte_controlo
    )

    if acoes_controlo_w.shape != (
        horizonte_controlo,
    ):
        raise ValueError(
            "O vetor de ações de controlo deve possuir "
            f"dimensão ({horizonte_controlo},)."
        )

    if not np.all(
        np.isfinite(
            acoes_controlo_w
        )
    ):
        raise ValueError(
            "As ações de controlo contêm valores não finitos."
        )

    tamanho_bloco = (
        configuracao.horizonte_predicao
        // configuracao.horizonte_controlo
    )

    potencia_expandida_w = np.repeat(
        acoes_controlo_w,
        tamanho_bloco,
    )

    if potencia_expandida_w.shape != (
        configuracao.horizonte_predicao,
    ):
        raise RuntimeError(
            "A expansão das ações de controlo produziu uma "
            "dimensão inesperada."
        )

    return potencia_expandida_w


# ============================================================
# Função objetivo para o otimizador
# ============================================================


def funcao_objetivo_mpc(
    acoes_controlo_normalizadas: np.ndarray,
    matrizes,
    estado_inicial: np.ndarray,
    temperatura_exterior_prevista: np.ndarray,
    potencia_anterior_w: float,
    referencia_temperatura: np.ndarray,
    configuracao: ConfiguracaoMPC2R2C,
) -> float:
    """
    Função objetivo utilizada pelo algoritmo de otimização.

    O SLSQP trabalha internamente com Nc variáveis normalizadas
    no intervalo [0, 1]. Estas são convertidas para potência [W]
    e posteriormente expandidas para as Np amostras.
    """

    acoes_controlo_w = desnormalizar_potencia_mpc(
        potencia_normalizada=acoes_controlo_normalizadas,
        configuracao=configuracao,
    )

    potencia_w = expandir_acoes_controlo(
        acoes_controlo_w=acoes_controlo_w,
        configuracao=configuracao,
    )

    componentes = calcular_componentes_custo_mpc(
        matrizes=matrizes,
        estado_inicial=estado_inicial,
        temperatura_exterior_prevista=temperatura_exterior_prevista,
        potencia_w=potencia_w,
        potencia_anterior_w=potencia_anterior_w,
        referencia_temperatura=referencia_temperatura,
        configuracao=configuracao,
    )

    return componentes.custo_total


# ============================================================
# Resolução do problema MPC
# ============================================================


def resolver_mpc_2r2c(
    matrizes,
    estado_inicial: np.ndarray,
    temperatura_exterior_prevista: np.ndarray,
    potencia_anterior_w: float,
    configuracao: ConfiguracaoMPC2R2C,
    temperatura_referencia=None,
    potencia_inicial_w: np.ndarray | None = None,
    maximo_iteracoes: int = 300,
) -> ResultadoMPC2R2C:
    """
    Resolve um problema MPC para o instante atual.

    O SLSQP otimiza Nc ações normalizadas no intervalo [0, 1].
    Depois, essas ações são convertidas para potência [W] e
    expandidas para o horizonte completo de Np amostras.

    A interface externa permanece em unidades físicas [W].
    """

    validar_matrizes_discretas_mpc(matrizes)
    validar_configuracao_mpc(configuracao)

    estado_inicial = validar_estado_inicial_mpc(
        estado_inicial
    )

    horizonte = configuracao.horizonte_predicao
    horizonte_controlo = configuracao.horizonte_controlo
    tamanho_bloco = horizonte // horizonte_controlo

    temperatura_exterior_prevista = (
        validar_previsao_temperatura_exterior(
            temperatura_exterior_prevista,
            horizonte,
        )
    )

    potencia_anterior_w = float(potencia_anterior_w)

    if not np.isfinite(potencia_anterior_w):
        raise ValueError(
            "A potência anterior deve ser finita."
        )

    if (
        potencia_anterior_w
        < configuracao.potencia_minima_w
        or potencia_anterior_w
        > configuracao.potencia_maxima_w
    ):
        raise ValueError(
            "A potência anterior encontra-se fora dos limites definidos."
        )

    if temperatura_referencia is None:
        temperatura_referencia = (
            configuracao.temperatura_referencia
        )

    referencia_temperatura = (
        preparar_referencia_temperatura(
            temperatura_referencia,
            horizonte,
        )
    )

    if potencia_inicial_w is None:
        acoes_iniciais_w = np.full(
            horizonte_controlo,
            potencia_anterior_w,
            dtype=float,
        )
    else:
        try:
            potencia_inicial_w = np.asarray(
                potencia_inicial_w,
                dtype=float,
            )
        except (TypeError, ValueError) as erro:
            raise TypeError(
                "A potência inicial deve conter valores numéricos."
            ) from erro

        if not np.all(np.isfinite(potencia_inicial_w)):
            raise ValueError(
                "A potência inicial contém valores não finitos."
            )

        if potencia_inicial_w.shape == (
            horizonte_controlo,
        ):
            acoes_iniciais_w = potencia_inicial_w.copy()

        elif potencia_inicial_w.shape == (
            horizonte,
        ):
            acoes_iniciais_w = np.mean(
                potencia_inicial_w.reshape(
                    horizonte_controlo,
                    tamanho_bloco,
                ),
                axis=1,
            )

        else:
            raise ValueError(
                "A sequência inicial de potência deve possuir "
                f"dimensão ({horizonte_controlo},) ou ({horizonte},)."
            )

        acoes_iniciais_w = np.clip(
            acoes_iniciais_w,
            configuracao.potencia_minima_w,
            configuracao.potencia_maxima_w,
        )

    if not isinstance(
        maximo_iteracoes,
        (int, np.integer),
    ):
        raise TypeError(
            "O número máximo de iterações deve ser inteiro."
        )

    maximo_iteracoes = int(maximo_iteracoes)

    if maximo_iteracoes <= 0:
        raise ValueError(
            "O número máximo de iterações deve ser estritamente positivo."
        )

    acoes_iniciais_normalizadas = (
        normalizar_potencia_mpc(
            potencia_w=acoes_iniciais_w,
            configuracao=configuracao,
        )
    )

    limites_normalizados = [
        (0.0, 1.0)
        for _ in range(horizonte_controlo)
    ]

    resultado_scipy = minimize(
        fun=funcao_objetivo_mpc,
        x0=acoes_iniciais_normalizadas,
        args=(
            matrizes,
            estado_inicial,
            temperatura_exterior_prevista,
            potencia_anterior_w,
            referencia_temperatura,
            configuracao,
        ),
        method="SLSQP",
        bounds=limites_normalizados,
        options={
            "maxiter": maximo_iteracoes,
            "ftol": 1.0e-8,
            "disp": False,
        },
    )

    acoes_otimas_normalizadas = np.asarray(
        resultado_scipy.x,
        dtype=float,
    )

    acoes_otimas_normalizadas = np.clip(
        acoes_otimas_normalizadas,
        0.0,
        1.0,
    )

    acoes_otimas_w = desnormalizar_potencia_mpc(
        potencia_normalizada=acoes_otimas_normalizadas,
        configuracao=configuracao,
    )

    potencia_otima_w = expandir_acoes_controlo(
        acoes_controlo_w=acoes_otimas_w,
        configuracao=configuracao,
    )

    (
        estados_previstos,
        temperatura_prevista,
    ) = prever_trajetoria_mpc(
        matrizes=matrizes,
        estado_inicial=estado_inicial,
        temperatura_exterior_prevista=temperatura_exterior_prevista,
        potencia_w=potencia_otima_w,
    )

    componentes_custo = calcular_componentes_custo_mpc(
        matrizes=matrizes,
        estado_inicial=estado_inicial,
        temperatura_exterior_prevista=temperatura_exterior_prevista,
        potencia_w=potencia_otima_w,
        potencia_anterior_w=potencia_anterior_w,
        referencia_temperatura=referencia_temperatura,
        configuracao=configuracao,
    )

    return ResultadoMPC2R2C(
        potencia_otima_w=potencia_otima_w.copy(),
        estados_previstos=estados_previstos.copy(),
        temperatura_prevista=temperatura_prevista.copy(),
        componentes_custo=componentes_custo,
        sucesso=bool(resultado_scipy.success),
        estado_otimizador=int(resultado_scipy.status),
        mensagem=str(resultado_scipy.message),
        numero_iteracoes=int(resultado_scipy.nit),
        numero_avaliacoes=int(resultado_scipy.nfev),
        primeira_acao_w=float(potencia_otima_w[0]),
    )


# ============================================================
# Validação do resultado
# ============================================================


def validar_resultado_mpc(
    resultado: ResultadoMPC2R2C,
    configuracao: ConfiguracaoMPC2R2C,
) -> None:
    """
    Valida estrutural e numericamente o resultado do MPC.
    """

    if not isinstance(
        resultado,
        ResultadoMPC2R2C,
    ):
        raise TypeError(
            "O resultado deve ser uma instância de "
            "ResultadoMPC2R2C."
        )

    validar_configuracao_mpc(
        configuracao
    )

    horizonte = (
        configuracao.horizonte_predicao
    )

    if resultado.potencia_otima_w.shape != (
        horizonte,
    ):
        raise ValueError(
            "A potência ótima apresenta dimensão incorreta."
        )

    if resultado.estados_previstos.shape != (
        horizonte + 1,
        2,
    ):
        raise ValueError(
            "Os estados previstos apresentam dimensão "
            "incorreta."
        )

    if resultado.temperatura_prevista.shape != (
        horizonte,
    ):
        raise ValueError(
            "A temperatura prevista apresenta dimensão "
            "incorreta."
        )

    if not np.all(
        np.isfinite(
            resultado.potencia_otima_w
        )
    ):
        raise ValueError(
            "A potência ótima contém valores não finitos."
        )

    if not np.all(
        np.isfinite(
            resultado.estados_previstos
        )
    ):
        raise ValueError(
            "Os estados previstos contêm valores não finitos."
        )

    if not np.all(
        np.isfinite(
            resultado.temperatura_prevista
        )
    ):
        raise ValueError(
            "A temperatura prevista contém valores não "
            "finitos."
        )

    if np.any(
        resultado.potencia_otima_w
        < configuracao.potencia_minima_w
        - 1.0e-6
    ):
        raise ValueError(
            "A solução viola o limite inferior de potência."
        )

    if np.any(
        resultado.potencia_otima_w
        > configuracao.potencia_maxima_w
        + 1.0e-6
    ):
        raise ValueError(
            "A solução viola o limite superior de potência."
        )


# ============================================================
# Apresentação do resultado
# ============================================================


def mostrar_resultado_mpc(
    resultado: ResultadoMPC2R2C,
    configuracao: ConfiguracaoMPC2R2C,
) -> None:
    """
    Apresenta um resumo da solução MPC.
    """

    validar_resultado_mpc(
        resultado,
        configuracao,
    )

    temperatura = (
        resultado.temperatura_prevista
    )

    potencia = (
        resultado.potencia_otima_w
    )

    custo = (
        resultado.componentes_custo
    )

    print(
        "========== RESULTADO DO MPC 2R2C ==========\n"
    )

    print(
        f"Convergência: "
        f"{resultado.sucesso}"
    )

    print(
        f"Estado do otimizador: "
        f"{resultado.estado_otimizador}"
    )

    print(
        f"Mensagem: "
        f"{resultado.mensagem}"
    )

    print(
        f"Número de iterações: "
        f"{resultado.numero_iteracoes}"
    )

    print(
        f"Número de avaliações: "
        f"{resultado.numero_avaliacoes}"
    )

    print(
        "\nAção de controlo:"
    )

    print(
        f"Primeira potência ótima = "
        f"{resultado.primeira_acao_w:.2f} W"
    )

    print(
        f"Potência mínima prevista = "
        f"{potencia.min():.2f} W"
    )

    print(
        f"Potência máxima prevista = "
        f"{potencia.max():.2f} W"
    )

    print(
        "\nTemperatura prevista:"
    )

    print(
        f"Inicial prevista = "
        f"{temperatura[0]:.3f} °C"
    )

    print(
        f"Final prevista = "
        f"{temperatura[-1]:.3f} °C"
    )

    print(
        f"Mínima prevista = "
        f"{temperatura.min():.3f} °C"
    )

    print(
        f"Máxima prevista = "
        f"{temperatura.max():.3f} °C"
    )

    print(
        "\nComponentes da função objetivo:"
    )

    print(
        f"Temperatura = "
        f"{custo.custo_temperatura:.6f}"
    )

    print(
        f"Potência = "
        f"{custo.custo_potencia:.6f}"
    )

    print(
        f"Variação da potência = "
        f"{custo.custo_variacao_potencia:.6f}"
    )

    print(
        f"Violação de conforto = "
        f"{custo.custo_violacao_conforto:.6f}"
    )

    print(
        f"Custo total = "
        f"{custo.custo_total:.6f}"
    )

    print(
        "\n=========================================="
    )