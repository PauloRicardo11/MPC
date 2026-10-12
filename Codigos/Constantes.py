"""
============================================================
Módulo: Constantes.py

Descrição:
    Constantes globais utilizadas na Plataforma de
    Processamento de Dados Experimentais.

    Este módulo contém apenas parâmetros globais reutilizados
    por diferentes módulos da plataforma.

Autor:
    Paulo Ribeiro

Projeto:
    Plataforma de Processamento de Dados Experimentais
============================================================
"""

# ============================================================
# Configuração dos ficheiros CSV
# ============================================================

CODIFICACAO_CSV = "utf-8-sig"
CODIFICACAO_ALTERNATIVA = "latin-1"

SEPARADOR_DECIMAL = "."


# ============================================================
# Configuração temporal
# ============================================================

INTERVALO_AMOSTRAGEM_TEMPERATURA_S = 60


# ============================================================
# Colunas obrigatórias dos dados de temperatura
# ============================================================

COLUNAS_TEMPERATURA_OBRIGATORIAS = (
    "tempo_s",
    "T_int",
    "T_ext",
)


# ============================================================
# Limites físicos preliminares
# ============================================================

TEMPERATURA_MINIMA_C = -20.0
TEMPERATURA_MAXIMA_C = 60.0