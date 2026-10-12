"""
============================================================
Módulo: Importacao.py

Descrição:
    Funções destinadas exclusivamente à leitura dos ficheiros
    de dados experimentais utilizados no projeto.

    O módulo permite importar ficheiros CSV genéricos e o
    ficheiro basic_data.csv produzido pelo analisador Mi550.

    As operações de validação, limpeza, transformação e
    exportação são realizadas noutros módulos da plataforma.

Autor:
    Paulo Ribeiro

Projeto:
    Plataforma de Processamento de Dados Experimentais
============================================================
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd

from Codigos.Constantes import (
    CODIFICACAO_ALTERNATIVA,
    CODIFICACAO_CSV,
    SEPARADOR_DECIMAL,
)


def verificar_ficheiro_csv(
    caminho_ficheiro: str | Path,
) -> Path:
    """
    Verifica se um caminho corresponde a um ficheiro CSV válido
    para leitura.

    Esta função realiza apenas as verificações necessárias à
    importação. Não valida o conteúdo experimental.

    Parameters
    ----------
    caminho_ficheiro:
        Caminho para o ficheiro CSV.

    Returns
    -------
    pathlib.Path
        Caminho absoluto e validado.

    Raises
    ------
    TypeError
        Caso o caminho possua um tipo inválido.

    FileNotFoundError
        Caso o ficheiro não exista.

    ValueError
        Caso o caminho não corresponda a um ficheiro, não possua
        extensão CSV ou esteja vazio.
    """

    if not isinstance(caminho_ficheiro, (str, Path)):
        raise TypeError(
            "O caminho do ficheiro deve ser uma string "
            "ou um objeto pathlib.Path."
        )

    caminho = Path(caminho_ficheiro).expanduser().resolve()

    if not caminho.exists():
        raise FileNotFoundError(
            f"O ficheiro não foi encontrado: {caminho}"
        )

    if not caminho.is_file():
        raise ValueError(
            "O caminho indicado não corresponde a um ficheiro: "
            f"{caminho}"
        )

    if caminho.suffix.lower() != ".csv":
        raise ValueError(
            f"O ficheiro '{caminho.name}' não possui a extensão .csv."
        )

    if caminho.stat().st_size == 0:
        raise ValueError(
            f"O ficheiro '{caminho.name}' encontra-se vazio."
        )

    return caminho


def detetar_separador(
    caminho_ficheiro: str | Path,
    codificacao: str = CODIFICACAO_CSV,
    numero_linhas: int = 10,
) -> str:
    """
    Deteta o separador predominante num ficheiro CSV.

    São considerados os seguintes separadores:

    - vírgula;
    - ponto e vírgula;
    - tabulação.

    Parameters
    ----------
    caminho_ficheiro:
        Caminho para o ficheiro CSV.

    codificacao:
        Codificação utilizada para a leitura preliminar.

    numero_linhas:
        Número máximo de linhas não vazias a analisar.

    Returns
    -------
    str
        Separador identificado.

    Raises
    ------
    TypeError
        Caso os parâmetros possuam tipos inválidos.

    ValueError
        Caso os parâmetros sejam inválidos ou não seja possível
        identificar o separador.
    """

    caminho = verificar_ficheiro_csv(caminho_ficheiro)

    if not isinstance(codificacao, str):
        raise TypeError(
            "A codificação deve ser uma string."
        )

    if not codificacao:
        raise ValueError(
            "A codificação não pode ser uma string vazia."
        )

    if not isinstance(numero_linhas, int):
        raise TypeError(
            "O número de linhas a analisar deve ser um valor inteiro."
        )

    if numero_linhas <= 0:
        raise ValueError(
            "O número de linhas a analisar deve ser superior a zero."
        )

    linhas_analisadas: list[str] = []

    with caminho.open(
        mode="r",
        encoding=codificacao,
        errors="strict",
    ) as ficheiro:
        for linha in ficheiro:
            linha = linha.strip()

            if not linha:
                continue

            linhas_analisadas.append(linha)

            if len(linhas_analisadas) >= numero_linhas:
                break

    if not linhas_analisadas:
        raise ValueError(
            f"O ficheiro '{caminho.name}' não contém linhas utilizáveis."
        )

    separadores_possiveis = (",", ";", "\t")

    contagens = {
        separador: sum(
            linha.count(separador)
            for linha in linhas_analisadas
        )
        for separador in separadores_possiveis
    }

    separador_identificado = max(
        contagens,
        key=contagens.get,
    )

    if contagens[separador_identificado] == 0:
        raise ValueError(
            "Não foi possível identificar o separador do ficheiro "
            f"'{caminho.name}'."
        )

    return separador_identificado


def importar_csv(
    caminho_ficheiro: str | Path,
    separador: str | None = None,
    codificacao: str = CODIFICACAO_CSV,
    decimal: str = SEPARADOR_DECIMAL,
    **parametros_leitura: Any,
) -> pd.DataFrame:
    """
    Importa um ficheiro CSV genérico para um DataFrame.

    A função verifica as condições necessárias à leitura, deteta
    automaticamente o separador quando este não é indicado e tenta
    uma codificação alternativa caso a codificação principal falhe.

    Não são realizadas validações relativas ao conteúdo experimental.

    Parameters
    ----------
    caminho_ficheiro:
        Caminho para o ficheiro CSV.

    separador:
        Separador utilizado no ficheiro. Quando não é indicado,
        é detetado automaticamente.

    codificacao:
        Codificação principal utilizada na importação.

    decimal:
        Símbolo utilizado como separador decimal.

    **parametros_leitura:
        Parâmetros adicionais encaminhados para pandas.read_csv.

    Returns
    -------
    pandas.DataFrame
        DataFrame resultante da importação.

    Raises
    ------
    TypeError
        Caso os parâmetros possuam tipos inválidos.

    ValueError
        Caso existam parâmetros incompatíveis ou não seja possível
        importar o ficheiro.
    """

    caminho = verificar_ficheiro_csv(caminho_ficheiro)

    if separador is not None:
        if not isinstance(separador, str):
            raise TypeError(
                "O separador deve ser uma string ou None."
            )

        if len(separador) != 1:
            raise ValueError(
                "O separador deve possuir exatamente um carácter."
            )

    if not isinstance(codificacao, str):
        raise TypeError(
            "A codificação deve ser uma string."
        )

    if not codificacao:
        raise ValueError(
            "A codificação não pode ser uma string vazia."
        )

    if not isinstance(decimal, str):
        raise TypeError(
            "O separador decimal deve ser uma string."
        )

    if len(decimal) != 1:
        raise ValueError(
            "O separador decimal deve possuir exatamente um carácter."
        )

    parametros_reservados = {
        "filepath_or_buffer",
        "sep",
        "encoding",
        "decimal",
    }

    parametros_em_conflito = (
        parametros_reservados
        & set(parametros_leitura)
    )

    if parametros_em_conflito:
        raise ValueError(
            "Os seguintes parâmetros devem ser definidos através "
            "dos argumentos próprios de importar_csv: "
            f"{sorted(parametros_em_conflito)}"
        )

    if separador is None:
        try:
            separador = detetar_separador(
                caminho_ficheiro=caminho,
                codificacao=codificacao,
            )

        except UnicodeDecodeError:
            separador = detetar_separador(
                caminho_ficheiro=caminho,
                codificacao=CODIFICACAO_ALTERNATIVA,
            )

    parametros: dict[str, Any] = {
        "filepath_or_buffer": caminho,
        "sep": separador,
        "encoding": codificacao,
        "decimal": decimal,
    }

    parametros.update(parametros_leitura)

    try:
        dataframe = pd.read_csv(**parametros)

    except UnicodeDecodeError:
        parametros["encoding"] = CODIFICACAO_ALTERNATIVA

        try:
            dataframe = pd.read_csv(**parametros)

        except pd.errors.EmptyDataError as erro:
            raise ValueError(
                f"O ficheiro '{caminho.name}' não contém dados legíveis."
            ) from erro

        except pd.errors.ParserError as erro:
            raise ValueError(
                "Erro ao interpretar o ficheiro "
                f"'{caminho.name}': {erro}"
            ) from erro

    except pd.errors.EmptyDataError as erro:
        raise ValueError(
            f"O ficheiro '{caminho.name}' não contém dados legíveis."
        ) from erro

    except pd.errors.ParserError as erro:
        raise ValueError(
            "Erro ao interpretar o ficheiro "
            f"'{caminho.name}': {erro}"
        ) from erro

    if dataframe.empty:
        raise ValueError(
            f"O ficheiro '{caminho.name}' foi importado, "
            "mas não contém linhas de dados."
        )

    return dataframe


def localizar_primeira_linha_dados_mi550(
    caminho_ficheiro: str | Path,
    codificacao: str = CODIFICACAO_CSV,
) -> int:
    """
    Localiza a primeira linha de dados no ficheiro do Mi550.

    O formato esperado no início de cada linha de dados é:

    ``AAAA/MM/DD,HH:MM:SS,``

    Parameters
    ----------
    caminho_ficheiro:
        Caminho para o ficheiro basic_data.csv.

    codificacao:
        Codificação utilizada na leitura preliminar.

    Returns
    -------
    int
        Índice da primeira linha de dados, com indexação iniciada
        em zero.

    Raises
    ------
    TypeError
        Caso a codificação possua um tipo inválido.

    ValueError
        Caso não seja possível localizar a primeira linha de dados.
    """

    caminho = verificar_ficheiro_csv(caminho_ficheiro)

    if not isinstance(codificacao, str):
        raise TypeError(
            "A codificação deve ser uma string."
        )

    if not codificacao:
        raise ValueError(
            "A codificação não pode ser uma string vazia."
        )

    padrao_data = re.compile(
        r"^\d{4}/\d{2}/\d{2},\d{2}:\d{2}:\d{2},"
    )

    with caminho.open(
        mode="r",
        encoding=codificacao,
        errors="replace",
    ) as ficheiro:
        for indice, linha in enumerate(ficheiro):
            if padrao_data.match(linha.strip()):
                return indice

    raise ValueError(
        "Não foi possível localizar a primeira linha de dados "
        f"no ficheiro Mi550 '{caminho.name}'."
    )


def importar_basic_data(
    caminho_ficheiro: str | Path,
    codificacao: str = CODIFICACAO_CSV,
) -> pd.DataFrame:
    """
    Importa o ficheiro basic_data.csv do analisador Mi550.

    O ficheiro contém metadados nas primeiras linhas e um
    cabeçalho distribuído por duas linhas:

    - primeira linha: grupos de grandezas e designações Date/Time;
    - segunda linha: nomes individuais das variáveis elétricas.

    A função localiza automaticamente a primeira linha de dados,
    reconstrói o cabeçalho efetivo, garante nomes de colunas
    únicos e importa todas as medições.

    A coluna adicional completamente vazia, resultante da vírgula
    terminal existente em cada linha do ficheiro, é ignorada
    durante a importação.

    Não são realizadas operações de limpeza ou transformação dos
    dados experimentais.

    Parameters
    ----------
    caminho_ficheiro:
        Caminho para o ficheiro basic_data.csv.

    codificacao:
        Codificação principal utilizada na leitura.

    Returns
    -------
    pandas.DataFrame
        DataFrame com os dados importados do Mi550.

    Raises
    ------
    TypeError
        Caso a codificação possua um tipo inválido.

    ValueError
        Caso não seja possível localizar ou interpretar os dados.
    """

    caminho = verificar_ficheiro_csv(caminho_ficheiro)

    if not isinstance(codificacao, str):
        raise TypeError(
            "A codificação deve ser uma string."
        )

    if not codificacao:
        raise ValueError(
            "A codificação não pode ser uma string vazia."
        )

    # --------------------------------------------------------
    # Localização da primeira linha de dados
    # --------------------------------------------------------

    try:
        linha_primeiros_dados = localizar_primeira_linha_dados_mi550(
            caminho_ficheiro=caminho,
            codificacao=codificacao,
        )

        codificacao_utilizada = codificacao

    except UnicodeDecodeError:
        linha_primeiros_dados = localizar_primeira_linha_dados_mi550(
            caminho_ficheiro=caminho,
            codificacao=CODIFICACAO_ALTERNATIVA,
        )

        codificacao_utilizada = CODIFICACAO_ALTERNATIVA

    if linha_primeiros_dados < 2:
        raise ValueError(
            "O ficheiro Mi550 não contém as duas linhas de "
            "cabeçalho esperadas antes dos dados."
        )

    linha_grupos = linha_primeiros_dados - 2
    linha_variaveis = linha_primeiros_dados - 1

    # --------------------------------------------------------
    # Leitura das duas linhas de cabeçalho
    # --------------------------------------------------------

    try:
        cabecalho_grupos = pd.read_csv(
            caminho,
            sep=",",
            header=None,
            skiprows=linha_grupos,
            nrows=1,
            encoding=codificacao_utilizada,
        ).iloc[0].tolist()

        cabecalho_variaveis = pd.read_csv(
            caminho,
            sep=",",
            header=None,
            skiprows=linha_variaveis,
            nrows=1,
            encoding=codificacao_utilizada,
        ).iloc[0].tolist()

    except pd.errors.EmptyDataError as erro:
        raise ValueError(
            f"O cabeçalho do ficheiro Mi550 '{caminho.name}' "
            "não contém dados legíveis."
        ) from erro

    except pd.errors.ParserError as erro:
        raise ValueError(
            "Erro ao interpretar o cabeçalho do ficheiro Mi550 "
            f"'{caminho.name}': {erro}"
        ) from erro

    # --------------------------------------------------------
    # Reconstrução dos nomes das colunas
    # --------------------------------------------------------

    nomes_colunas: list[str] = []
    nomes_utilizados: set[str] = set()

    grupo_atual: str | None = None

    for indice, nome_variavel in enumerate(cabecalho_variaveis):

        # Atualização do grupo associado à coluna
        if indice < len(cabecalho_grupos):
            nome_grupo = cabecalho_grupos[indice]

            if pd.notna(nome_grupo):
                grupo_atual = str(nome_grupo).strip()

        # ----------------------------------------------------
        # Determinação do nome base
        # ----------------------------------------------------

        if pd.notna(nome_variavel):
            nome_base = str(nome_variavel).strip()

        elif indice < len(cabecalho_grupos):
            nome_grupo = cabecalho_grupos[indice]

            if pd.notna(nome_grupo):
                nome_base = str(nome_grupo).strip()
            else:
                nome_base = f"Coluna_{indice}"

        else:
            nome_base = f"Coluna_{indice}"

        # ----------------------------------------------------
        # Garantia de unicidade do nome
        # ----------------------------------------------------

        nome_final = nome_base

        if nome_final in nomes_utilizados:

            if grupo_atual:
                nome_final = (
                    f"{nome_base} [{grupo_atual}]"
                )

            else:
                nome_final = (
                    f"{nome_base} [coluna {indice}]"
                )

        contador = 2
        nome_original = nome_final

        while nome_final in nomes_utilizados:
            nome_final = f"{nome_original} [{contador}]"
            contador += 1

        nomes_colunas.append(nome_final)
        nomes_utilizados.add(nome_final)

    # --------------------------------------------------------
    # Importação das medições
    # --------------------------------------------------------

    try:
        dataframe = pd.read_csv(
            caminho,
            sep=",",
            header=None,
            skiprows=linha_primeiros_dados,
            encoding=codificacao_utilizada,
            low_memory=False,
        )

    except pd.errors.EmptyDataError as erro:
        raise ValueError(
            f"O ficheiro Mi550 '{caminho.name}' "
            "não contém dados legíveis."
        ) from erro

    except pd.errors.ParserError as erro:
        raise ValueError(
            "Erro ao interpretar os dados do ficheiro Mi550 "
            f"'{caminho.name}': {erro}"
        ) from erro

    if dataframe.empty:
        raise ValueError(
            f"O ficheiro Mi550 '{caminho.name}' foi importado, "
            "mas não contém linhas de dados."
        )

    # --------------------------------------------------------
    # Remoção controlada da coluna terminal vazia
    # --------------------------------------------------------

    if dataframe.iloc[:, -1].isna().all():
        dataframe = dataframe.iloc[:, :-1].copy()

    # --------------------------------------------------------
    # Verificação da compatibilidade do cabeçalho
    # --------------------------------------------------------

    if dataframe.shape[1] != len(nomes_colunas):
        raise ValueError(
            "O número de colunas dos dados do Mi550 não coincide "
            "com o número de colunas reconstruídas a partir do "
            "cabeçalho. "
            f"Dados: {dataframe.shape[1]}; "
            f"Cabeçalho: {len(nomes_colunas)}."
        )

    dataframe.columns = nomes_colunas

    return dataframe