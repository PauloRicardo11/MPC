"""
============================================================
Módulo: Exportacao.py

Descrição:
    Funções destinadas à exportação controlada dos dados
    processados pela plataforma.

    O módulo permite guardar DataFrames em ficheiros CSV,
    assegurando:

    - validação do caminho de destino;
    - proteção das pastas de dados originais;
    - controlo explícito da substituição de ficheiros existentes;
    - escrita temporária antes da criação do ficheiro final;
    - preservação do DataFrame recebido.

    As operações de importação, validação, limpeza e sincronização
    são realizadas nos respetivos módulos da plataforma.

Autor:
    Paulo Ribeiro

Projeto:
    Plataforma de Processamento de Dados Experimentais
============================================================
"""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from Codigos.Constantes import (
    CODIFICACAO_CSV,
    SEPARADOR_DECIMAL,
)


def verificar_dataframe_exportacao(
    dataframe: pd.DataFrame,
) -> None:
    """
    Verifica se o objeto fornecido pode ser exportado.

    Esta função realiza apenas verificações necessárias à
    operação de exportação. A validade científica e estrutural
    dos dados deve ser confirmada previamente pelo módulo de
    validação.

    Parameters
    ----------
    dataframe:
        DataFrame a verificar.

    Raises
    ------
    TypeError
        Caso o objeto não seja um DataFrame do pandas.

    ValueError
        Caso o DataFrame esteja vazio.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "O objeto fornecido não é um DataFrame do pandas."
        )

    if dataframe.empty:
        raise ValueError(
            "O DataFrame não contém dados para exportação."
        )


def verificar_caminho_exportacao(
    caminho_ficheiro: str | Path,
    criar_diretorio: bool = False,
) -> Path:
    """
    Verifica e prepara o caminho destinado à exportação.

    Por segurança, não é permitida a exportação para qualquer
    diretório denominado ``Original``.

    Parameters
    ----------
    caminho_ficheiro:
        Caminho do ficheiro CSV a criar.

    criar_diretorio:
        Quando True, cria o diretório de destino caso este ainda
        não exista.

    Returns
    -------
    pathlib.Path
        Caminho absoluto preparado para exportação.

    Raises
    ------
    TypeError
        Caso os parâmetros possuam tipos inválidos.

    FileNotFoundError
        Caso o diretório de destino não exista e a sua criação
        não tenha sido autorizada.

    ValueError
        Caso o caminho seja inválido, não possua extensão CSV ou
        pertença a uma pasta denominada ``Original``.
    """

    if not isinstance(caminho_ficheiro, (str, Path)):
        raise TypeError(
            "O caminho do ficheiro deve ser uma string "
            "ou um objeto pathlib.Path."
        )

    if not isinstance(criar_diretorio, bool):
        raise TypeError(
            "O parâmetro 'criar_diretorio' deve ser booleano."
        )

    caminho = (
        Path(caminho_ficheiro)
        .expanduser()
        .resolve()
    )

    if caminho.suffix.lower() != ".csv":
        raise ValueError(
            "O ficheiro de exportação deve possuir "
            "a extensão .csv."
        )

    if not caminho.name.strip():
        raise ValueError(
            "O nome do ficheiro de exportação é inválido."
        )

    partes_caminho = {
        parte.lower()
        for parte in caminho.parts
    }

    if "original" in partes_caminho:
        raise ValueError(
            "A exportação para uma pasta denominada "
            "'Original' não é permitida."
        )

    diretorio = caminho.parent

    if not diretorio.exists():

        if criar_diretorio:
            diretorio.mkdir(
                parents=True,
                exist_ok=True,
            )

        else:
            raise FileNotFoundError(
                "O diretório de destino não existe: "
                f"{diretorio}"
            )

    if not diretorio.is_dir():
        raise ValueError(
            "O caminho de destino não corresponde "
            f"a um diretório: {diretorio}"
        )

    return caminho


def exportar_csv(
    dataframe: pd.DataFrame,
    caminho_ficheiro: str | Path,
    separador: str = ",",
    codificacao: str = CODIFICACAO_CSV,
    decimal: str = SEPARADOR_DECIMAL,
    incluir_indice: bool = False,
    substituir: bool = False,
    criar_diretorio: bool = False,
    formato_data_hora: str = "%Y-%m-%d %H:%M:%S",
) -> Path:
    """
    Exporta um DataFrame para CSV de forma controlada.

    A escrita é inicialmente realizada num ficheiro temporário
    localizado no mesmo diretório do destino. Apenas após uma
    escrita bem-sucedida é criado ou substituído o ficheiro
    definitivo.

    Esta estratégia reduz o risco de permanecer um ficheiro
    final incompleto caso ocorra uma falha durante a exportação.

    Parameters
    ----------
    dataframe:
        DataFrame a exportar.

    caminho_ficheiro:
        Caminho do ficheiro CSV final.

    separador:
        Separador utilizado entre campos.

    codificacao:
        Codificação utilizada no ficheiro.

    decimal:
        Símbolo utilizado como separador decimal.

    incluir_indice:
        Indica se o índice do DataFrame deve ser exportado.

    substituir:
        Autoriza a substituição de um ficheiro existente.

    criar_diretorio:
        Autoriza a criação do diretório de destino caso este
        ainda não exista.

    formato_data_hora:
        Formato utilizado para colunas datetime.

    Returns
    -------
    pathlib.Path
        Caminho absoluto do ficheiro exportado.

    Raises
    ------
    TypeError
        Caso algum parâmetro possua tipo inválido.

    FileExistsError
        Caso o ficheiro já exista e a substituição não tenha
        sido autorizada.

    ValueError
        Caso os parâmetros sejam inválidos ou a exportação falhe.
    """

    verificar_dataframe_exportacao(
        dataframe
    )

    caminho = verificar_caminho_exportacao(
        caminho_ficheiro=caminho_ficheiro,
        criar_diretorio=criar_diretorio,
    )

    if not isinstance(separador, str):
        raise TypeError(
            "O separador deve ser uma string."
        )

    if len(separador) != 1:
        raise ValueError(
            "O separador deve possuir exatamente um carácter."
        )

    if not isinstance(codificacao, str):
        raise TypeError(
            "A codificação deve ser uma string."
        )

    if not codificacao.strip():
        raise ValueError(
            "A codificação não pode ser vazia."
        )

    if not isinstance(decimal, str):
        raise TypeError(
            "O separador decimal deve ser uma string."
        )

    if len(decimal) != 1:
        raise ValueError(
            "O separador decimal deve possuir exatamente "
            "um carácter."
        )

    if separador == decimal:
        raise ValueError(
            "O separador de campos e o separador decimal "
            "não podem ser iguais."
        )

    if not isinstance(incluir_indice, bool):
        raise TypeError(
            "O parâmetro 'incluir_indice' deve ser booleano."
        )

    if not isinstance(substituir, bool):
        raise TypeError(
            "O parâmetro 'substituir' deve ser booleano."
        )

    if not isinstance(formato_data_hora, str):
        raise TypeError(
            "O formato de data e hora deve ser uma string."
        )

    if not formato_data_hora.strip():
        raise ValueError(
            "O formato de data e hora não pode ser vazio."
        )

    if caminho.exists() and not substituir:
        raise FileExistsError(
            "O ficheiro de destino já existe e a substituição "
            f"não foi autorizada: {caminho}"
        )

    caminho_temporario: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            suffix=".tmp",
            prefix=f"{caminho.stem}_",
            dir=caminho.parent,
            delete=False,
            encoding=codificacao,
        ) as ficheiro_temporario:

            caminho_temporario = Path(
                ficheiro_temporario.name
            )

        dataframe.to_csv(
            caminho_temporario,
            sep=separador,
            encoding=codificacao,
            decimal=decimal,
            index=incluir_indice,
            date_format=formato_data_hora,
        )

        os.replace(
            caminho_temporario,
            caminho,
        )

    except (OSError, TypeError, ValueError) as erro:

        if (
            caminho_temporario is not None
            and caminho_temporario.exists()
        ):
            caminho_temporario.unlink()

        raise ValueError(
            "Não foi possível exportar o DataFrame para "
            f"'{caminho}'."
        ) from erro

    return caminho